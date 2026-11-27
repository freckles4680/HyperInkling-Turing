#!/usr/bin/env python3
"""Show prefill and decode speed under every answer, in any chat front end.

A small pass-through in front of the server: the front end (Open WebUI, Unsloth, any
OpenAI client) talks to this port instead of the server's, and each streamed chat
answer ends with one line such as

    ⚡ prefill 8,185 tok in 3.4 s (2,407 tok/s) · decode 512 tok at 84.1 tok/s

    scripts/speed_proxy.py                       # listens on 0.0.0.0:8081, server on :8080
    scripts/speed_proxy.py --port 9001 --upstream http://127.0.0.1:8080 --bind 127.0.0.1

Then give the front end  http://<this machine>:8081/v1  (same API key as the server, if
any: headers pass through). Everything else (/v1/models, /health, /metrics) is forwarded
unchanged, so nothing else needs to know about it.

What the numbers are:
  prefill  the prompt tokens the server had to compute (prompt minus the ones its prefix
           cache already held, shown as "cached"), over the time to the first token. A
           follow-up turn of a chat is mostly cached, so its rate says little: judge
           prefill on a first message or a new document.
  decode   the answer's tokens (reasoning included) over the time from the first token
           to the last. Short answers are noisy.
Both are measured here, at the proxy, so they include the request's time in the
server's queue when it is busy.

The line is added to streamed chat answers only (titles and other background requests
of a front end are not streamed, and answers that call tools are left alone), and it is
removed from the history a front end sends back, so the model never sees it and the
prefix cache still hits. Standard library only.
"""

import argparse
import http.client
import json
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARK = "\n\n---\n⚡ "
STRIP = re.compile(re.escape(MARK) + r"[^\n]*\s*$")
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host",
       "proxy-connection", "upgrade", "te", "trailer"}


def strip_stats(body):
    """Our line out of the assistant turns a front end sends back."""
    for message in body.get("messages") or []:
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            message["content"] = STRIP.sub("", message["content"])


def stats_line(usage, started, first, now):
    prompt = usage.get("prompt_tokens") or 0
    done = usage.get("completion_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    wait = max(first - started, 1e-6)
    if cached is None:  # the server does not report its prefix-cache hits
        prefill = f"prompt {prompt:,} tok, first token after {wait:.1f} s"
    else:
        fresh = max(prompt - cached, 0)
        prefill = f"prefill {fresh:,} tok in {wait:.1f} s ({fresh / wait:,.0f} tok/s)"
        if cached:
            prefill += f", {cached:,} cached"
    span = now - first
    decode = f"decode {done:,} tok"
    if done > 1 and span > 0:
        decode += f" at {done / span:.1f} tok/s"
    return f"{prefill} · {decode}"


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # a response ends when the connection closes
    upstream = None

    def log_message(self, *args):
        pass

    def _forward(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        chat = method == "POST" and self.path.split("?")[0].endswith("/chat/completions")
        annotate = False
        if chat and body:
            try:
                request = json.loads(body)
                strip_stats(request)
                if request.get("stream"):
                    annotate = True
                    options = request.setdefault("stream_options", {})
                    options["include_usage"] = options["continuous_usage_stats"] = True
                body = json.dumps(request).encode()
            except (ValueError, AttributeError):
                annotate = False
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        started = time.monotonic()
        conn = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port or 80,
                                          timeout=3600)
        try:
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if annotate and response.status == 200:
                self._stream(response, started)
            else:
                while chunk := response.read(65536):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client went away
        except OSError as error:
            try:
                self.send_error(502, f"server unreachable: {error}")
            except OSError:
                pass
        finally:
            conn.close()

    def _stream(self, response, started):
        """Server-sent events, one line at a time; our line goes in just before the
        chunk that ends the answer."""
        first, usage = None, {}
        while line := response.readline():
            text = line.strip()
            if text.startswith(b"data: ") and text != b"data: [DONE]":
                try:
                    event = json.loads(text[6:])
                except ValueError:
                    event = {}
                if event.get("usage"):
                    details = usage.get("prompt_tokens_details")
                    usage = dict(event["usage"])
                    if details and not usage.get("prompt_tokens_details"):
                        usage["prompt_tokens_details"] = details  # sent once, at the start
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    produced = any(delta.get(k) for k in
                                   ("content", "reasoning_content", "reasoning", "tool_calls"))
                    if first is None and produced:
                        first = time.monotonic()
                    if choice.get("finish_reason") in ("stop", "length") and first and usage:
                        note = stats_line(usage, started, first, time.monotonic())
                        extra = {**event, "choices": [{"index": choice.get("index", 0),
                                                       "delta": {"content": MARK + note},
                                                       "finish_reason": None}]}
                        extra.pop("usage", None)
                        self.wfile.write(b"data: " + json.dumps(extra).encode() + b"\n\n")
            self.wfile.write(line)
            self.wfile.flush()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_DELETE(self):
        self._forward("DELETE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--upstream", default="http://127.0.0.1:8080", help="the vLLM server")
    ap.add_argument("--bind", default="0.0.0.0", help="127.0.0.1 = this machine only")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    Proxy.upstream = urllib.parse.urlsplit(args.upstream)
    print(f"speed proxy: {args.bind}:{args.port} -> {args.upstream}", flush=True)
    ThreadingHTTPServer((args.bind, args.port), Proxy).serve_forever()


if __name__ == "__main__":
    main()
