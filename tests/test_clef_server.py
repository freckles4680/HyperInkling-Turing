"""The HTTP side of scripts/clef_server.py with a stand-in model (no GPU, no torch)."""

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import clef_server  # noqa: E402


class Stub:
    def answer(self, request):
        if not request.get("questions"):
            raise ValueError("questions must be a non-empty object")
        return {"model": request["model"], "usage": {"input_tokens": 3, "output_tokens": 0},
                "answers": {k: {"type": "noul", "noul": 0.9} for k in request["questions"]}}


@pytest.fixture
def port():
    handler = type("H", (clef_server.Handler,), {"service": Stub(), "names": ["clef"],
                                                 "api_key": ""})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def call(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=json.dumps(body).encode() if body is not None else None)
    response = conn.getresponse()
    return response.status, json.loads(response.read())


def test_systemone(port):
    status, body = call(port, "POST", "/v1/systemone",
                        {"state": "down", "questions": {"outage": {"type": "noul"}}})
    assert status == 200 and body["model"] == "clef" and body["answers"]["outage"]["noul"] == 0.9


def test_errors(port):
    assert call(port, "POST", "/v1/systemone", {"state": "x"})[0] == 400
    status, body = call(port, "POST", "/v1/chat/completions", {"model": "clef"})
    assert status == 404 and "does not chat" in body["error"]["message"]
    assert call(port, "GET", "/v1/models")[1]["data"][0]["id"] == "clef"
    assert call(port, "GET", "/health")[0] == 200
