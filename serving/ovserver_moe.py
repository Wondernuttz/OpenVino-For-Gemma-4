#!/usr/bin/env python3
# OpenAI-compatible OV server for the 26B MoE heretic. Single-stream (Intel's recommended mode for MoE on GPU).
# Coherent via: <bos> + Gemma chat format + repetition_penalty. Strips the leading <thought> reasoning.
# Concurrent requests FIFO-queue on a lock (no garbage, no crash). Drop-in for the SYCL llama-server.
import os, json, time, threading, re
import openvino as ov
import openvino_genai as g
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL      = os.environ.get("OV_MODEL", "/home/wondernutts/models/heretics/gemma-4-26B-A4B-heretic-int4-ov")
DEVICE     = os.environ.get("OV_DEVICE", "GPU.2")
PORT       = int(os.environ.get("OV_PORT", "8092"))
EXPECT_BUS = os.environ.get("OV_EXPECT_BUS", "")          # safety: abort if device PCI lacks this token
MODEL_NAME = os.environ.get("OV_MODEL_NAME", "gemma-4-26b-a4b-heretic")
REP_PEN    = float(os.environ.get("OV_REP_PEN", "1.2"))
THINK      = os.environ.get("OV_THINK", "0") == "1"      # 0 = suppress reasoning (fast, default), 1 = enable thinking
MAX_CTX_TOKENS = int(os.environ.get("OV_MAX_CTX_TOKENS", "20000"))   # hard input cap — drops oldest turns to prevent OOM
THINK_HEADROOM = int(os.environ.get("OV_THINK_HEADROOM", "1024"))    # when thinking, add this so client max_tokens = ANSWER length
_MAX_CTX_CHARS = MAX_CTX_TOKENS * 4                       # ~4 chars/token; well under the card's OOM ceiling at this size

core = ov.Core()
try: PCI = str(core.get_property(DEVICE, "DEVICE_PCI_INFO"))
except Exception as e: PCI = "unknown:%s" % e
print("[ov] device %s PCI=%s" % (DEVICE, PCI), flush=True)
if EXPECT_BUS and EXPECT_BUS not in PCI:
    raise SystemExit("[ov] ABORT: %s PCI '%s' missing expected bus token '%s'" % (DEVICE, PCI, EXPECT_BUS))

print("[ov] loading %s on %s ..." % (MODEL, DEVICE), flush=True)
_t0 = time.time()
pipe = g.VLMPipeline(MODEL, DEVICE, **{"DYNAMIC_QUANTIZATION_GROUP_SIZE": 0})
GEN_LOCK = threading.Lock()
TOK = pipe.get_tokenizer()
try: _RUN = g.StreamingStatus.RUNNING
except Exception: _RUN = False
class _Collector(g.StreamerBase):   # collects raw token IDs so we can decode with special tokens kept
    def __init__(self): super().__init__(); self.toks = []
    def write(self, token):
        if isinstance(token, (list, tuple)): self.toks.extend(int(t) for t in token)
        else: self.toks.append(int(token))
        return _RUN
    def end(self): pass
print("[ov] LOADED in %.1fs -- serving %s on :%d (single-stream, rep_pen=%.2f, thinking=%s, max_ctx=%d tok)" % (time.time() - _t0, MODEL_NAME, PORT, REP_PEN, "ON" if THINK else "OFF", MAX_CTX_TOKENS), flush=True)

def cap_context(messages):
    # Hard OOM guard: keep system + newest turns within the char budget; drop oldest beyond it.
    if sum(len(str(m.get("content", ""))) for m in messages) <= _MAX_CTX_CHARS:
        return messages
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest     = [m for m in messages if m.get("role") != "system"]
    budget = _MAX_CTX_CHARS - sum(len(str(m.get("content", ""))) for m in sys_msgs)
    kept = []
    for m in reversed(rest):
        c = len(str(m.get("content", "")))
        if c <= budget: kept.insert(0, m); budget -= c
        else: break
    if not kept and rest:                       # newest turn alone exceeds budget — keep its tail
        m = dict(rest[-1]); ct = m.get("content")
        if isinstance(ct, str): m["content"] = ct[-max(budget, 4000):]
        kept = [m]
    return sys_msgs + kept

def build_prompt(messages):
    # Native gemma-4 heretic format: <|turn>role ... <turn|>. Thinking suppressed via empty pre-closed channel.
    sys_txt, turns = "", []
    for m in messages:
        role = m.get("role", "user"); content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            sys_txt += content.strip() + "\n\n"; continue
        r = "model" if role == "assistant" else "user"
        turns.append((r, content.strip()))
    p = "<bos>"
    if sys_txt.strip() or THINK:
        p += "<|turn>system\n"
        if THINK: p += "<|think|>\n"
        p += sys_txt.strip() + "<turn|>\n"
    for r, c in turns:
        p += "<|turn>%s\n%s<turn|>\n" % (r, c)
    p += "<|turn>model\n"
    if not THINK:
        p += "<|channel>thought\n<channel|>"   # empty pre-closed thought channel = skip reasoning
    return p

# Backstop only — native suppression means nothing should leak; strips <thought>..</thought>, <|channel>thought..<channel|>, or a bare leading "thought" line.
_THOUGHT_TAG  = re.compile(r'^\s*<\|?(?:thought|channel\|?>?\s*thought).*?(?:</thought>|<\s*channel\s*\|?>)\s*', re.DOTALL | re.IGNORECASE)
_THOUGHT_LINE = re.compile(r'^[\s:>*_|-]*thought[\s:>*_-]*\n', re.IGNORECASE)
def strip_thinking(t):
    t = (t or "").strip()
    t = _THOUGHT_TAG.sub("", t, count=1).strip()
    t = _THOUGHT_LINE.sub("", t, count=1).strip()
    return t

def make_cfg(req):
    c = g.GenerationConfig()
    mt = int(req.get("max_tokens") or req.get("max_new_tokens") or req.get("max_completion_tokens") or 512)
    if THINK: mt += THINK_HEADROOM   # reserve room for hidden reasoning so the client's max_tokens = answer length
    c.max_new_tokens = mt
    t = req.get("temperature")
    if t is not None and float(t) > 0:
        c.do_sample = True; c.temperature = float(t)
        if req.get("top_p") is not None:
            try: c.top_p = float(req["top_p"])
            except Exception: pass
        if req.get("top_k") is not None:
            try: c.top_k = int(req["top_k"])
            except Exception: pass
    try: c.repetition_penalty = float(req.get("repetition_penalty") or req.get("repeat_penalty") or REP_PEN)
    except Exception: pass
    for k in ("frequency_penalty", "presence_penalty"):   # client-tunable from CHIM connector
        if req.get(k) is not None:
            try: setattr(c, k, float(req[k]))
            except Exception: pass
    try: c.apply_chat_template = False
    except Exception: pass
    return c

def generate(messages, cfg):
    messages = cap_context(messages)
    prompt = build_prompt(messages)
    if THINK:
        # Stream token IDs, decode with special tokens kept, drop everything up to the thought-close.
        col = _Collector()
        with GEN_LOCK:
            pipe.generate(prompt, generation_config=cfg, streamer=col)
        full = TOK.decode(col.toks, skip_special_tokens=False) if col.toks else ""
        if "<channel|>" in full:
            full = full.split("<channel|>", 1)[1]   # keep only the answer after the reasoning
        for s in ("<turn|>", "<eos>", "<pad>", "<bos>", "<|channel>"):
            full = full.replace(s, "")
        return strip_thinking(full)
    with GEN_LOCK:
        r = pipe.generate(prompt, generation_config=cfg)
    txt = None
    for a in ("texts", "m_generation_ids"):
        try:
            v = getattr(r, a)
            if v: txt = v[0]; break
        except Exception: pass
    return strip_thinking(txt if txt is not None else str(r))

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if "/v1/models" in self.path:
            self._json({"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "ov"}]})
        else:
            self._json({"status": "ok", "model": MODEL_NAME, "device": DEVICE})
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
        try:
            text = generate(msgs or [], make_cfg(req))
        except Exception as e:
            self._json({"error": "gen failed: %s" % str(e)[:200]}, 500); return
        cid = "chatcmpl-%d" % int(time.time() * 1000)
        if bool(req.get("stream")):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            self.wfile.write(("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}) + "\n\n").encode())
            self.wfile.write(("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._json({"id": cid, "object": "chat.completion", "model": MODEL_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(text) // 4, "total_tokens": 0}})

print("[ov] HTTP up on 0.0.0.0:%d" % PORT, flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
