#!/usr/bin/env python3
# OpenAI-compatible OV server for DiffusionGemma 26B-A4B (block-diffusion, world-first on Arc).
# Same contract as ovserver_moe.py: env OV_MODEL/OV_DEVICE/OV_PORT/OV_MODEL_NAME/OV_EXPECT_BUS,
# GET /v1/models, POST *completions* (messages | prompt; max_tokens -> 256-token blocks).
# Single-stream: one denoise loop at a time (the sampler owns one KV cache + infer request).
import json, os, re, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL      = os.environ.get("OV_MODEL", os.path.expanduser("~/dg_ov"))
DEVICE     = os.environ.get("OV_DEVICE", "GPU.1")
PORT       = int(os.environ.get("OV_PORT", "8002"))
EXPECT_BUS = os.environ.get("OV_EXPECT_BUS", "")
MODEL_NAME = os.environ.get("OV_MODEL_NAME", "diffusiongemma-26b-a4b")
MAX_BLOCKS_CAP = int(os.environ.get("DG_MAX_BLOCKS", "6"))     # 6 x 256 = 1536 tok ceiling
MAX_CTX_CHARS  = int(os.environ.get("OV_MAX_CTX_TOKENS", "8000")) * 4

import openvino as ov
_core = ov.Core()
try: PCI = str(_core.get_property(DEVICE, "DEVICE_PCI_INFO"))
except Exception as e: PCI = "unknown:%s" % e
print("[dg] device %s PCI=%s" % (DEVICE, PCI), flush=True)
if EXPECT_BUS and EXPECT_BUS not in PCI:
    raise SystemExit("[dg] ABORT: %s PCI '%s' missing expected bus token '%s'" % (DEVICE, PCI, EXPECT_BUS))

print("[dg] loading DiffusionGemma sampler (%s on %s)..." % (MODEL, DEVICE), flush=True)
_t0 = time.time()
sys.argv = ["dg_sampler", MODEL, DEVICE, "server-mode", "1"]
sys.path.insert(0, os.path.expanduser("~"))
import dg_sampler as dg   # loads both split models, warms up, ready to generate()
print("[dg] LOADED in %.1fs -- serving %s on :%d (canvas %d, <=%d steps/block)"
      % (time.time() - _t0, MODEL_NAME, PORT, dg.CANVAS, dg.MAX_STEPS), flush=True)

GEN_LOCK = threading.Lock()

# Backstop thought-stripping (native suppression happens in dg_sampler.generate via the
# empty pre-closed thought channel; this catches anything that still leaks).
_THOUGHT_TAG  = re.compile(r'^\s*<\|?(?:thought|channel\|?>?\s*thought).*?(?:</thought>|<\s*channel\s*\|?>)\s*', re.DOTALL | re.IGNORECASE)
_THOUGHT_LINE = re.compile(r'^[\s:>*_|-]*thought[\s:>*_-]*\n', re.IGNORECASE)
# Thought-channel content that survives the label strip: the model narrating its own
# instructions/identity instead of speaking in character. Leading paragraphs matching
# these ship analysis to Discord (seen live 2026-07-11) -> drop the leading meta run.
_META_PAT = re.compile(
    r"(?i)(the user'?s? (?:input|prompt|request|message)|i(?:'| a)m gemma|as gemma ?4"
    r"|large language model|developed by google|open weights model|knowledge cutoff"
    r"|the (?:prompt|instructions?) (?:is|are|state|includes?|provided|divided)"
    r"|character (?:profile|prompt|-based prompt)|adopt the persona|maintain the persona"
    r"|identify the objective|my (?:core )?identity|structural constraints"
    r"|role-?play(?:ing)? (?:scenario|prompt)|creative writing piece"
    r"|you'?ve provided|set of instructions|roleplay as|i must clarify"
    r"|i am not the character|i remain an ai|as an ai\b|designed to process"
    r"|examples? of (?:how|what) to)")
def strip_thinking(t):
    t = (t or "").strip()
    t = _THOUGHT_TAG.sub("", t, count=1).strip()
    t = _THOUGHT_LINE.sub("", t, count=1).strip()
    paras = re.split(r"\n\s*\n", t)
    drop = 0
    for pgh in paras:                       # strip only the LEADING meta run
        if _META_PAT.search(re.sub(r"[*_`]", "", pgh)):   # markdown bold hides "**Gemma 4**"
            drop += 1
        else:
            break
    return "\n\n".join(paras[drop:]).strip()   # all-meta -> empty (silence beats identity leaks)

MIN_CTX_CHARS = int(os.environ.get("DG_MIN_CTX_CHARS", "1400"))

def cap_context(messages):
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total < MIN_CTX_CHARS:
        # tiny prompts sit outside the GPU's warmed shape corridor (the driver silently
        # corrupts or crashes there); injecting pad text overrides personas, so refuse honestly
        raise ValueError("prompt too small for the diffusion shape corridor: need >= %d chars "
                         "of persona/context, got %d" % (MIN_CTX_CHARS, total))
    if total <= MAX_CTX_CHARS:
        return messages
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    while rest and sum(len(str(m.get("content", ""))) for m in sys_msgs + rest) > MAX_CTX_CHARS:
        rest.pop(0)
    return sys_msgs + rest

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if "/v1/models" in self.path:
            self._json({"object": "list", "data": [{"id": MODEL_NAME, "object": "model",
                        "owned_by": "wondernutts", "note": "block-diffusion LLM, ~256 tok/block"}]})
        else:
            self._json({"status": "ok", "model": MODEL_NAME, "engine": "block-diffusion"})
    def do_POST(self):
        if "completions" not in self.path:
            self._json({"error": "not found"}, 404); return
        try:
            ln = int(self.headers.get("Content-Length", "0") or 0)
            req = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            self._json({"error": "bad json"}, 400); return
        msgs = req.get("messages")
        if msgs is None and "prompt" in req: msgs = [{"role": "user", "content": req["prompt"]}]
        mt = int(req.get("max_tokens") or req.get("max_new_tokens") or 512)
        blocks = max(1, min(MAX_BLOCKS_CAP, (mt + dg.CANVAS - 1) // dg.CANVAS))
        t0 = time.time()
        try:
            with GEN_LOCK:
                text = strip_thinking(dg.generate(cap_context(msgs or []), max_blocks=blocks))
        except Exception as e:
            self._json({"error": "gen failed: %s" % str(e)[:200]}, 500); return
        dt = time.time() - t0
        print("[dg] %d blocks in %.1fs (%.1f tok/s effective)" % (blocks, dt, len(text.split()) * 1.3 / max(dt, 0.1)), flush=True)
        cid = "chatcmpl-%d" % int(time.time() * 1000)
        base = {"id": cid, "model": MODEL_NAME, "created": int(time.time())}
        if bool(req.get("stream")):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            self.wfile.write(("data: " + json.dumps({**base, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}) + "\n\n").encode())
            self.wfile.write(("data: " + json.dumps({**base, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\ndata: [DONE]\n\n").encode())
        else:
            self._json({**base, "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
