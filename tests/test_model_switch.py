"""scripts/model_switch.py with stand-in servers (no GPU): routing by model name, one
server at a time, waiting requests, refused paths, and the API key."""

import http.client
import json
import socket
import sys
import textwrap
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import model_switch  # noqa: E402

FAKE = textwrap.dedent('''
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    name, port, delay = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    time.sleep(delay)  # a boot
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self.reply({"served_by": name, "path": self.path})
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = json.loads(body)
            time.sleep(request.get("sleep", 0))
            self.reply({"served_by": name, "path": self.path,
                        "auth": self.headers.get("Authorization")})
    print(name, "up", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def switch(tmp_path):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    ports = {"a": free_port(), "b": free_port()}
    config = {
        "switch": {"stop_timeout": 5, "start_timeout": 20},
        "models": {
            "a": {"command": f"{sys.executable} {fake} a {ports['a']} 0.3",
                  "port": ports["a"], "aliases": ["local"]},
            "b": {"command": f"{sys.executable} {fake} b {ports['b']} 0.3",
                  "port": ports["b"], "listed": False, "paths": ["/v1/systemone"]},
        },
    }
    lines = []
    switcher = model_switch.Switcher(config, lambda src, text: lines.append((src, text)))
    handler = type("H", (model_switch.Handler,), {"switcher": switcher, "api_key": ""})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], switcher, handler, lines
    server.shutdown()
    switcher.shutdown()


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers=headers or {})
    response = conn.getresponse()
    out = response.status, json.loads(response.read() or b"null")
    conn.close()
    return out


def test_models_listed_without_starting(switch):
    port, switcher, _, _ = switch
    status, body = call(port, "GET", "/v1/models")
    assert status == 200 and [m["id"] for m in body["data"]] == ["a"]
    assert switcher.status()["active"] is None


def test_switches_by_model_and_alias(switch):
    port, switcher, _, lines = switch
    assert call(port, "POST", "/v1/chat/completions", {"model": "local"})[1]["served_by"] == "a"
    # a request without a model goes to whatever runs
    assert call(port, "GET", "/metrics")[1]["served_by"] == "a"
    assert call(port, "POST", "/v1/systemone", {"model": "b"})[1]["served_by"] == "b"
    assert switcher.status()["active"] == "b"
    assert any("stopping a" in text for _, text in lines)
    assert any(src == "b" and "up" in text for src, text in lines)  # its output is logged
    assert call(port, "GET", "/metrics")[0] == 400  # b answers /v1/systemone only


def test_refuses_unknown_model_and_path(switch):
    port, switcher, _, _ = switch
    assert call(port, "POST", "/v1/chat/completions", {"model": "nope"})[0] == 404
    status, body = call(port, "POST", "/v1/chat/completions", {"model": "b"})
    assert status == 400 and "/v1/systemone" in body["error"]["message"]
    assert switcher.status()["active"] is None  # refused before any switch


def test_switch_waits_for_requests_in_flight(switch):
    port, switcher, _, _ = switch
    call(port, "POST", "/switch/load", {"model": "a"})
    results = {}
    slow = threading.Thread(target=lambda: results.update(
        slow=call(port, "POST", "/v1/chat/completions", {"model": "a", "sleep": 1.5})))
    slow.start()
    time.sleep(0.3)
    results["b"] = call(port, "POST", "/v1/systemone", {"model": "b"})
    slow.join()
    assert results["slow"] == (200, results["slow"][1]) and results["slow"][1]["served_by"] == "a"
    assert results["b"][1]["served_by"] == "b"


def test_unload_and_api_key(switch):
    port, switcher, handler, _ = switch
    call(port, "POST", "/switch/load", {"model": "a"})
    assert call(port, "POST", "/switch/unload", {})[1] == {"unloaded": "a"}
    assert switcher.status()["active"] is None
    handler.api_key = "k"
    assert call(port, "GET", "/v1/models")[0] == 401
    assert call(port, "GET", "/health")[0] == 200
    status, body = call(port, "POST", "/v1/chat/completions", {"model": "a"},
                        {"Authorization": "Bearer k"})
    assert status == 200 and body["auth"] is None  # the key stays at the switcher


def test_failed_start_is_reported(switch, tmp_path):
    port, switcher, _, _ = switch
    switcher.models["a"].command = "echo broken config; exit 3"
    status, body = call(port, "POST", "/v1/chat/completions", {"model": "a"})
    assert status == 503 and "broken config" in body["error"]["message"]
    assert call(port, "POST", "/v1/systemone", {"model": "b"})[0] == 200  # still usable


def test_waits_for_ready_line_after_health(switch, tmp_path):
    port, switcher, _, lines = switch
    warm = tmp_path / "warm.py"
    a_port = switcher.models["a"].port
    warm.write_text(FAKE.replace('print(name, "up", flush=True)\n',
                                 'import threading\n'
                                 'threading.Timer(1.5, lambda: print("warm-up done in 1.5 s",'
                                 ' flush=True)).start()\n'))
    switcher.models["a"].command = f"{sys.executable} {warm} a {a_port} 0.1"
    switcher.models["a"].ready_lines = ["warm-up done"]
    t0 = time.monotonic()
    assert call(port, "POST", "/v1/chat/completions", {"model": "a"})[0] == 200
    assert time.monotonic() - t0 >= 1.5
    texts = [t for _, t in lines]
    assert texts.index("warm-up done in 1.5 s") < next(
        i for i, t in enumerate(texts) if t.startswith("a ready in"))


def test_goes_back_home_after_idle(switch):
    port, switcher, _, lines = switch
    switcher.home, switcher.home_after = switcher.models["a"], 1
    threading.Thread(target=switcher.home_loop, daemon=True).start()
    assert call(port, "POST", "/v1/systemone", {"model": "b"})[0] == 200
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and switcher.status()["active"] != "a":
        time.sleep(0.2)
    assert switcher.status()["active"] == "a"
    assert any("back to a" in t for _, t in lines)
