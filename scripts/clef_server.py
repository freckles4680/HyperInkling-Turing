#!/usr/bin/env python3
"""Serve Cloudflare Clef (a decision model) on a Turing card, with the Jev/SystemOne API.

Clef does not write text: for a state (any text or JSON, optionally images) and a set of
typed questions it returns a probability per allowed answer, in one forward pass. The
model code is the release's own joint_schema_model.py (loaded from the checkpoint); this
file adds what the 2080 Ti needs and an HTTP server:

  * fp16 compute (Turing has no bf16 tensor cores), with an overflow check
  * 4-bit NF4 or 8-bit weights through bitsandbytes (27B in bf16 is 55 GB; NF4 ~19 GB),
    LM head and vision tower kept in fp16 (the head reads the LM head as a table)
  * token embeddings in host RAM on request (CLEF_EMBED_CPU=1, 2.5 GB less on the GPU)
  * --save-quantized: write the NF4 model once, later loads skip the quantization

    POST /v1/systemone   {"model": "clef", "state": "...", "questions": {...}}
    GET  /v1/models, GET /health

Run it through scripts/clef.sh (settings, venv); docs/clef.md has the request format.
"""

import argparse
import base64
import io
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def log(text):
    print(f"{time.strftime('%H:%M:%S')} clef: {text}", flush=True)


# --- loading --------------------------------------------------------------------------
def quantization_config(quant, skip):
    import torch
    from transformers import BitsAndBytesConfig

    if quant == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True,
                                  llm_int8_skip_modules=skip)
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    return None


def backbone_of(model):
    for child in model.children():
        if hasattr(child, "save_pretrained"):
            return child
    raise RuntimeError("no transformers model inside the Clef model")


def embeddings_to_host(backbone):
    """Token embeddings in pinned host RAM: the lookup runs on the CPU, the rows go up."""
    import torch

    embed = backbone.get_input_embeddings()
    device = embed.weight.device
    embed.to("cpu")
    embed.weight.data = embed.weight.data.pin_memory()

    def before(module, args):
        return tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args)

    def after(module, args, output):
        return output.to(device, non_blocking=True)

    embed.register_forward_pre_hook(before)
    embed.register_forward_hook(after)


def headroom_hooks(backbone):
    """Largest |value| leaving each decoder layer (fp16 saturates at 65504)."""
    import torch

    seen = {"max": 0.0, "layer": -1}
    layers = None
    for name, module in backbone.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList):
            if layers is None or len(module) > len(layers):  # text layers, not vision
                layers = module
    for i, layer in enumerate(layers or []):
        def hook(module, args, output, i=i):
            out = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out):
                value = out.detach().abs().amax().float().item()
                if not value < seen["max"]:  # also catches nan
                    seen["max"], seen["layer"] = value, i
        layer.register_forward_hook(hook)
    return seen


class Clef:
    def __init__(self, args):
        import torch

        self.torch = torch
        self.args = args
        self.lock = threading.Lock()
        path = Path(args.model).resolve()
        if not (path / "joint_schema_model.py").is_file():
            raise SystemExit(f"{path} is not a Clef checkpoint (no joint_schema_model.py): "
                             "scripts/setup-clef.sh downloads it")
        sys.path.insert(0, str(path))
        import joint_schema_model as jsm

        self.jsm = jsm
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
        prequantized = "quantization_config" in json.loads((path / "config.json").read_text())
        kwargs = {"attn_implementation": "sdpa"}
        skip = ["lm_head", "visual", *[s for s in args.skip.split(",") if s]]
        if not prequantized:
            config = quantization_config(args.quant, skip)
            if config is not None:
                kwargs["quantization_config"] = config
        gpu = "no GPU"
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            gpu = f"{torch.cuda.get_device_name()} sm{cap[0]}{cap[1]}"
        log(f"torch {torch.__version__}, {gpu}, loading {path.name} "
            f"({'pre-quantized' if prequantized else args.quant}, {args.dtype})")
        started = time.monotonic()
        self.model, self.processor = jsm.load_release_model(path, device=args.device,
                                                            dtype=dtype, **kwargs)
        backbone = backbone_of(self.model)
        if args.save_quantized:
            self._memory(started)
            self.save(path, Path(args.save_quantized), backbone)
            return
        if args.embed_cpu:
            embeddings_to_host(backbone)
            torch.cuda.empty_cache()
        self.seen = headroom_hooks(backbone) if args.dtype == "float16" else None
        self._memory(started)

    def _memory(self, started):
        torch = self.torch
        note = f"loaded in {time.monotonic() - started:.0f} s"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            note += (f", {torch.cuda.memory_allocated() / 2**30:.2f} GiB on the GPU "
                     f"(card {total / 2**30:.1f} GiB, {free / 2**30:.1f} free)")
        log(note)

    def save(self, source, target, backbone):
        target.mkdir(parents=True, exist_ok=True)
        log(f"writing the quantized model to {target}")
        backbone.save_pretrained(target)
        self.processor.save_pretrained(target)
        for name in ("joint_head.safetensors", "joint_head_config.json",
                     "joint_schema_model.py", "chat_template.jinja", "LICENSE", "README.md"):
            if (source / name).is_file():
                shutil.copy2(source / name, target / name)
        log(f"done: CLEF_PATH={target} scripts/clef.sh  loads it without quantizing")

    def answer(self, request):
        torch = self.torch
        request = dict(request)
        if request.get("images"):
            request["images"] = [decode_image(i) for i in request["images"]]
        if request.get("videos"):
            raise ValueError("videos are not taken over HTTP here; send frames as images")
        with self.lock:  # one forward at a time on one card
            if self.seen is not None:
                self.seen.update(max=0.0, layer=-1)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            try:
                response = self.jsm.systemone(self.model, self.processor, request,
                                              max_length=self.args.max_len)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise MemoryError("out of GPU memory: send a shorter state, or set "
                                  "CLEF_MAX_LEN lower / CLEF_EMBED_CPU=1 in scripts/clef.sh")
            seconds = time.monotonic() - started
        tokens = response.get("usage", {}).get("input_tokens", 0)
        note = f"{tokens:,} tokens in {seconds:.2f} s ({tokens / max(seconds, 1e-6):,.0f} tok/s)"
        if torch.cuda.is_available():
            note += f", peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        if self.seen is not None:
            peak = self.seen["max"]
            note += f", fp16 peak |x| {peak:,.0f} at layer {self.seen['layer']}"
            if not peak < 65504:
                raise ArithmeticError(
                    f"fp16 overflow at layer {self.seen['layer']} (|x| = {peak}): the "
                    "answer would be wrong. CLEF_DTYPE=bfloat16 avoids it (slower on "
                    "Turing); please send .local/model-switch.log or this server's output")
        log(note)
        return response


def decode_image(item):
    """An image as a data: URL or bare base64 (files and URLs are not fetched)."""
    from PIL import Image

    if not isinstance(item, str):
        raise ValueError("each image must be a base64 string or a data: URL")
    if item.startswith("data:"):
        item = item.split(",", 1)[-1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB")
    except Exception as exc:
        raise ValueError(f"an image could not be decoded: {exc}") from exc


# --- HTTP -----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    service = None   # anything with .answer(dict) -> dict
    names = ()
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

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self._json(200 if self.service else 503, {"status": "ok"})
        if self.api_key and self.headers.get("Authorization") != f"Bearer {self.api_key}":
            return self._error(401, "missing or wrong API key", "authentication_error")
        if path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": n, "object": "model", "created": 0, "owned_by": "cloudflare"}
                for n in self.names]})
        self._error(404, f"{path}: this server answers POST /v1/systemone (Clef is a "
                         "decision model, it does not chat)", "not_found_error")

    def do_POST(self):
        path = self.path.split("?")[0]
        if self.api_key and self.headers.get("Authorization") != f"Bearer {self.api_key}":
            return self._error(401, "missing or wrong API key", "authentication_error")
        if path not in ("/v1/systemone", "/systemone"):
            return self._error(404, f"{path}: this server answers POST /v1/systemone (Clef "
                                    "is a decision model, it does not chat)",
                               "not_found_error")
        if self.service is None:
            return self._error(503, "the model is still loading", "server_error")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"null")
            if not isinstance(request, dict):
                raise ValueError("the body must be a JSON object")
            request.setdefault("model", self.names[0] if self.names else "clef")
            response = self.service.answer(request)
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(400, str(exc))
        except (MemoryError, ArithmeticError) as exc:
            return self._error(503, str(exc), "server_error")
        except Exception as exc:  # keep serving; the trace goes to the log
            import traceback

            traceback.print_exc()
            return self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
        self._json(200, response)


SELFTEST = {
    "state": "Our checkout started returning errors at 09:12 and every order since is "
             "blocked. Customers are calling. Please help ASAP.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle it?",
                       "criteria": {"billing": "Payments or invoices",
                                    "technical": "Bugs or outages"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}


def selftest(service, name, tokens):
    """The expected answers are technical / ~2 (today) / outage true."""
    response = service.answer({"model": name, **SELFTEST})
    log("self-test: " + json.dumps(response["answers"]))
    if tokens:
        filler = ("Ticket log line: the payment page timed out for a customer in region "
                  "eu-west, retried, and succeeded on the second attempt. ")
        state = filler * max(1, tokens // 28)
        response = service.answer({"model": name, "state": state,
                                   "questions": {"outage": SELFTEST["questions"]["outage"]}})
        log(f"self-test at ~{tokens} tokens: " + json.dumps(response["answers"]))


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=env("CLEF_PATH", "models/clef"))
    ap.add_argument("--name", default=env("CLEF_NAME", "clef"),
                    help="model id(s) in /v1/models, comma-separated")
    ap.add_argument("--quant", choices=["nf4", "int8", "none"], default=env("CLEF_QUANT", "nf4"))
    ap.add_argument("--dtype", choices=["float16", "bfloat16"],
                    default=env("CLEF_DTYPE", "float16"))
    ap.add_argument("--skip", default=env("CLEF_SKIP", ""),
                    help="more module names to keep unquantized, comma-separated")
    ap.add_argument("--embed-cpu", type=int, default=int(env("CLEF_EMBED_CPU", "0")))
    ap.add_argument("--max-len", type=int, default=int(env("CLEF_MAX_LEN", "16384")))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bind", default=env("BIND", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("PORT", "18090")))
    ap.add_argument("--selftest", type=int, default=int(env("CLEF_SELFTEST", "1")))
    ap.add_argument("--selftest-tokens", type=int, default=int(env("CLEF_SELFTEST_TOKENS", "0")))
    ap.add_argument("--save-quantized", default="",
                    help="write the quantized model to this directory and exit")
    args = ap.parse_args()
    names = [n for n in args.name.split(",") if n]
    service = Clef(args)
    if args.save_quantized:
        return 0
    if args.selftest:
        selftest(service, names[0], args.selftest_tokens)
    Handler.service, Handler.names = service, names
    Handler.api_key = env("API_KEY", "")
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True
    log(f"listening on http://{args.bind}:{args.port}/v1/systemone as {', '.join(names)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
