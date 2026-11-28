#!/usr/bin/env python3
"""One endpoint for several models, one model on the GPU at a time.

vLLM serves one model per process and the card holds one model, so this sits in front:
it reads the "model" field of each request, and when that model is not the one running it
waits for the requests in flight to end, stops the running server, starts the asked one
and forwards the request once it answers /health. Open WebUI (or any OpenAI client) sees
every chat model in /v1/models and switches by picking one.

    scripts/model_switch.py                                 # scripts/models.toml, :8080
    scripts/model_switch.py --config my.toml --port 9000 --bind 127.0.0.1
    scripts/serve-switch.sh                                 # the usual way: with the speed proxy

The servers behind it listen on 127.0.0.1 only, without a key: the key (API_KEY) is
checked here. Everything every server prints goes, prefixed with its name, into one log
(--log, .local/model-switch.log by default) next to the switch events, so one file shows
a whole session.

Endpoints of its own:
  GET  /health            200 while the switcher runs (whatever model is loaded)
  GET  /v1/models         every model in the config with listed = true (nothing starts)
  GET  /switch/status     which model runs, which is starting, requests in flight
  POST /switch/load       {"model": "clef"}: switch now (to warm a model up ahead)
  POST /switch/unload     stop the running model, free the GPU
Requests without a model field (/metrics, /tokenize...) go to the running server.
Standard library only.
"""

import argparse
import collections
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host",
       "proxy-connection", "upgrade", "te", "trailer", "authorization"}


class Log:
    """One file for the switcher and every server it starts, also echoed to stdout."""

    def __init__(self, path):
        self.lock = threading.Lock()
        self.file = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def __call__(self, source, text):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{source}] {text.rstrip()}"
        with self.lock:
            print(line, flush=True)
            if self.file:
                self.file.write(line + "\n")


class Model:
    def __init__(self, name, spec, defaults):
        self.name = name
        self.command = spec["command"]
        self.port = int(spec["port"])
        self.health = spec.get("health", "/health")
        self.aliases = list(spec.get("aliases", []))
        self.listed = bool(spec.get("listed", True))
        # Paths this model answers. A chat sent to a decision model fails here, with a
        # message, instead of after a switch.
        self.paths = list(spec.get("paths", []))
        self.start_timeout = float(spec.get("start_timeout", defaults["start_timeout"]))
        self.env = {k: str(v) for k, v in spec.get("env", {}).items()}
        # Output lines (substrings) that mean the model is ready after /health answers:
        # Inkling's start.sh warms up in the background after /health, and a request sent
        # during that warm-up shares the decode slots with it (half the speed). If none is
        # printed within ready_timeout seconds of /health, requests go through anyway.
        ready = spec.get("ready_line", [])
        self.ready_lines = [ready] if isinstance(ready, str) else list(ready)
        self.ready_timeout = float(spec.get("ready_timeout", 600))
        self.description = spec.get("description", "")

    def accepts(self, path):
        return not self.paths or any(path.startswith(p) for p in self.paths)


class Switcher:
    def __init__(self, config, log):
        defaults = {"start_timeout": 900, "stop_timeout": 60, "idle_unload": 0,
                    "home": "", "home_after": 120}
        defaults.update(config.get("switch", {}))
        self.stop_timeout = float(defaults["stop_timeout"])
        self.idle_unload = float(defaults["idle_unload"])
        # The model to go back to once another one has been idle home_after seconds (and
        # to start with), so that it is booted and warm when the next chat comes.
        self.home_name = str(defaults["home"])
        self.home_after = float(defaults["home_after"])
        self.models = {}
        self.names = {}
        for name, spec in config.get("models", {}).items():
            model = Model(name, spec, defaults)
            self.models[name] = model
            for key in [name, *model.aliases]:
                if key in self.names:
                    raise SystemExit(f"model name {key!r} is used twice in the config")
                self.names[key] = model
        if not self.models:
            raise SystemExit("the config has no [models.<name>] entry")
        self.home = self.names.get(self.home_name) if self.home_name else None
        if self.home_name and self.home is None:
            raise SystemExit(f"[switch] home = {self.home_name!r}: no such model")
        self.log = log
        self.cond = threading.Condition()
        self.active = None       # Model whose server is up
        self.target = None       # Model being started (or None)
        self.inflight = 0
        self.process = None
        self.last_used = time.monotonic()
        self.last_error = ""
        self.tail = collections.deque(maxlen=40)

    # --- requests -------------------------------------------------------------------
    def resolve(self, name):
        return self.names.get(name) if isinstance(name, str) else None

    def acquire(self, model):
        """Wait until `model` runs (switching to it if needed); count the request."""
        with self.cond:
            while True:
                if self.target is None and self.active is model:
                    self.inflight += 1
                    return
                if self.target is None:
                    self.target = model  # new requests for the old model now wait too
                    break
                self.cond.wait()
            while self.inflight:
                self.cond.wait()
            old = self.active
            self.active = None
        error = ""
        try:
            if old is not None:
                self._stop(old, f"switching to {model.name}")
            self._start(model)
        except Exception as exc:  # report it to every waiting request
            error = str(exc)
        with self.cond:
            self.target = None
            if error:
                self.last_error = error
            else:
                self.active = model
                self.inflight += 1
            self.cond.notify_all()
        if error:
            raise RuntimeError(error)

    def release(self):
        with self.cond:
            self.inflight -= 1
            self.last_used = time.monotonic()
            self.cond.notify_all()

    def unload(self, reason):
        with self.cond:
            while self.target is not None or self.inflight:
                self.cond.wait()
            old, self.active = self.active, None
            self.target = old  # hold new requests while it stops
        try:
            if old is not None:
                self._stop(old, reason)
        finally:
            with self.cond:
                self.target = None
                self.cond.notify_all()
        return old

    def status(self):
        with self.cond:
            return {"active": self.active.name if self.active else None,
                    "starting": self.target.name if self.target else None,
                    "inflight": self.inflight,
                    "idle_s": round(time.monotonic() - self.last_used, 1),
                    "last_error": self.last_error,
                    "models": {m.name: {"port": m.port, "listed": m.listed,
                                        "aliases": m.aliases, "paths": m.paths}
                               for m in self.models.values()}}

    # --- processes ------------------------------------------------------------------
    def _start(self, model):
        if port_open(model.port):
            raise RuntimeError(f"port {model.port} for {model.name} is already taken by "
                               "something this switcher did not start")
        self.log("switch", f"starting {model.name}: {model.command}")
        self.tail.clear()
        self.ready_seen = threading.Event()
        env = {**os.environ, **model.env}
        started = time.monotonic()
        self.process = subprocess.Popen(
            model.command, shell=True, cwd=REPO, env=env, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        threading.Thread(target=self._pump, args=(model, self.process, self.ready_seen),
                         daemon=True).start()
        deadline = started + model.start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = "\n".join(self.tail)
                self.process = None
                raise RuntimeError(f"{model.name} exited while starting; its last lines:\n"
                                   f"{tail}")
            if http_ok(model.port, model.health):
                break
            time.sleep(1)
        else:
            self._stop(model, "start timed out")
            raise RuntimeError(f"{model.name} did not answer {model.health} within "
                               f"{model.start_timeout:.0f} s")
        if model.ready_lines:
            self.log("switch", f"{model.name} answers {model.health} after "
                               f"{time.monotonic() - started:.0f} s; waiting for its "
                               f"warm-up ({' / '.join(model.ready_lines)})")
            ready_deadline = time.monotonic() + model.ready_timeout
            while not self.ready_seen.wait(1):
                if self.process.poll() is not None:
                    tail = "\n".join(self.tail)
                    self.process = None
                    raise RuntimeError(f"{model.name} exited while warming up; its last "
                                       f"lines:\n{tail}")
                if time.monotonic() > ready_deadline:
                    self.log("switch", f"{model.name}: no warm-up line within "
                                       f"{model.ready_timeout:.0f} s, serving anyway")
                    break
        self.log("switch", f"{model.name} ready in {time.monotonic() - started:.0f} s")

    def _pump(self, model, process, ready_seen):
        for raw in process.stdout:
            text = raw.decode("utf-8", "replace").rstrip()
            self.tail.append(text)
            self.log(model.name, text)
            if any(line in text for line in model.ready_lines):
                ready_seen.set()

    def _stop(self, model, reason):
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        self.log("switch", f"stopping {model.name} ({reason})")
        started = time.monotonic()
        signal_group(process.pid, signal.SIGTERM)
        try:
            process.wait(self.stop_timeout)
        except subprocess.TimeoutExpired:
            self.log("switch", f"{model.name} ignored SIGTERM for "
                               f"{self.stop_timeout:.0f} s: SIGKILL")
            signal_group(process.pid, signal.SIGKILL)
            process.wait()
        # The group's other members (the engine core, a helper) can outlive the leader.
        for _ in range(int(self.stop_timeout * 5)):
            if not group_alive(process.pid):
                break
            time.sleep(0.2)
        else:
            signal_group(process.pid, signal.SIGKILL)
        wait_gpu_free(self.log)
        self.log("switch", f"{model.name} stopped in {time.monotonic() - started:.0f} s")

    def idle_loop(self):
        while self.idle_unload > 0:
            time.sleep(min(30, self.idle_unload))
            with self.cond:
                idle = (self.active is not None and self.target is None
                        and not self.inflight
                        and time.monotonic() - self.last_used > self.idle_unload)
            if idle:
                self.unload(f"idle for {self.idle_unload:.0f} s")

    def home_loop(self):
        """Back to the home model once another one has been idle home_after seconds."""
        while self.home is not None:
            time.sleep(min(10, max(self.home_after, 1)))
            with self.cond:
                away = (self.active is not None and self.active is not self.home
                        and self.target is None and not self.inflight
                        and time.monotonic() - self.last_used > self.home_after)
            if away:
                self.log("switch", f"{self.active.name} idle for {self.home_after:.0f} s: "
                                   f"back to {self.home.name}")
                _preload(self, self.home)

    def shutdown(self):
        if self.active is not None or self.process is not None:
            model = self.active or self.target or next(iter(self.models.values()))
            self._stop(model, "switcher exiting")


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def group_alive(pgid):
    try:
        out = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return bool(out.stdout.strip())


def wait_gpu_free(log, seconds=30):
    """A stopped engine can hold the card's memory for a moment after it exits."""
    for i in range(seconds):
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                  "--format=csv,noheader"], capture_output=True, text=True,
                                 timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if not out.stdout.strip():
            return
        if i == 0:
            log("switch", f"waiting for the GPU to be released: {out.stdout.strip()}")
        time.sleep(1)
    log("switch", "the GPU still has compute processes after the stop; starting anyway")


def port_open(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        conn.connect()
        return True
    except OSError:
        return False
    finally:
        conn.close()


def http_ok(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        conn.request("GET", path)
        return conn.getresponse().status == 200
    except OSError:
        return False
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # a response ends when the connection closes
    switcher = None
    api_key = ""

    def log_message(self, *args):
        pass

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message, kind="invalid_request_error"):
        self._json(status, {"error": {"message": message, "type": kind, "code": status}})

    def _authorized(self):
        if not self.api_key:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {self.api_key}"

    def _handle(self, method):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._json(200, {"status": "ok"})
        if not self._authorized():
            return self._error(401, "missing or wrong API key", "authentication_error")
        sw = self.switcher
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        if method == "GET" and path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": m.name, "object": "model", "created": 0, "owned_by": "hyperinkling",
                 **({"description": m.description} if m.description else {})}
                for m in sw.models.values() if m.listed]})
        if path == "/switch/status":
            return self._json(200, sw.status())
        request = None
        if body:
            try:
                request = json.loads(body)
            except ValueError:
                request = None
        name = request.get("model") if isinstance(request, dict) else None
        if path == "/switch/unload" and method == "POST":
            old = sw.unload("asked for")
            return self._json(200, {"unloaded": old.name if old else None})
        if path == "/switch/load" and method == "POST":
            model = sw.resolve(name)
            if model is None:
                return self._error(404, f"no model {name!r} in the config")
            try:
                sw.acquire(model)
            except RuntimeError as exc:
                return self._error(503, str(exc), "server_error")
            sw.release()
            return self._json(200, {"active": model.name})
        if name is None:
            # Nothing names a model: the running server answers, if there is one.
            with sw.cond:
                model = sw.active if sw.target is None else None
            if model is None:
                return self._error(503, "no model is running; name one in the request's "
                                        "\"model\" field, or POST /switch/load")
        else:
            model = sw.resolve(name)
            if model is None:
                known = ", ".join(m.name for m in sw.models.values())
                return self._error(404, f"the model {name!r} does not exist here "
                                        f"(known: {known})", "not_found_error")
        if not model.accepts(path):
            return self._error(400, f"{model.name} does not answer {path}: it answers "
                                    f"{', '.join(model.paths)} only. {model.description}")
        try:
            sw.acquire(model)
        except RuntimeError as exc:
            return self._error(503, f"could not start {model.name}: {exc}", "server_error")
        try:
            self._forward(method, model, body)
        finally:
            sw.release()

    def _forward(self, method, model, body):
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection("127.0.0.1", model.port, timeout=3600)
        try:
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read1(65536):  # streams as the server writes
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client went away
        except OSError as error:
            try:
                self._error(502, f"{model.name} unreachable: {error}", "server_error")
            except OSError:
                pass
        finally:
            conn.close()

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(REPO / "scripts/models.toml"))
    ap.add_argument("--bind", default=os.environ.get("BIND", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    ap.add_argument("--log", default=os.environ.get("SWITCH_LOG",
                                                    str(REPO / ".local/model-switch.log")))
    ap.add_argument("--preload", default=os.environ.get("PRELOAD", ""),
                    help="model to start at once instead of on the first request")
    args = ap.parse_args()
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    log = Log(args.log)
    switcher = Switcher(config, log)
    Handler.switcher = switcher
    Handler.api_key = os.environ.get("API_KEY", "")
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    names = ", ".join(f"{m.name}" + ("" if m.listed else " (unlisted)")
                      for m in switcher.models.values())
    log("switch", f"listening on {args.bind}:{args.port}, models: {names}; "
                  f"log {args.log}" + (" (API key required)" if Handler.api_key else ""))
    if switcher.idle_unload > 0:
        threading.Thread(target=switcher.idle_loop, daemon=True).start()
    if switcher.home is not None and switcher.idle_unload <= 0:
        threading.Thread(target=switcher.home_loop, daemon=True).start()
    preload = args.preload or (switcher.home.name if switcher.home else "")
    if preload == "none":
        preload = ""
    if preload:
        model = switcher.resolve(preload)
        if model is None:
            raise SystemExit(f"PRELOAD={preload}: no such model in the config")
        threading.Thread(target=lambda: _preload(switcher, model), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        switcher.shutdown()
        log("switch", "exited")


def _preload(switcher, model):
    try:
        switcher.acquire(model)
        switcher.release()
    except RuntimeError as exc:
        switcher.log("switch", f"preload of {model.name} failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
