#!/usr/bin/env python3
# Portable text server derived from the live September 6 deployment.
# Profiles: launch.py. Serialized generation; buffered SSE; no native AV.
import os, sys, glob, hashlib, json, time, threading, re, base64, binascii, io, ipaddress, socket
import urllib.parse, urllib.request
import openvino as ov
import openvino_genai as g
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL      = os.environ["OV_MODEL"]
DEVICE     = os.environ.get("OV_DEVICE", "GPU")
PORT       = int(os.environ.get("OV_PORT", "8000"))
EXPECT_BUS = os.environ.get("OV_EXPECT_BUS", "")          # safety: abort if device PCI lacks this token
MODEL_NAME = os.environ.get("OV_MODEL_NAME", "gemma-4-26b-a4b-heretic")


def _runtime_identity():
    libs = glob.glob(os.path.join(os.path.dirname(ov.__file__), "libs",
                                  "libopenvino_intel_gpu_plugin.so"))
    plugin = os.path.realpath(libs[0]) if libs else "missing"
    digest = "missing"
    if libs:
        h = hashlib.sha256()
        with open(libs[0], "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return plugin, digest


_GPU_PLUGIN, _GPU_PLUGIN_SHA256 = _runtime_identity()
print("[ov] runtime python=%s openvino=%s genai=%s" %
      (sys.executable, ov.__version__, g.__version__), flush=True)
print("[ov] runtime gpu_plugin=%s sha256=%s" %
      (_GPU_PLUGIN, _GPU_PLUGIN_SHA256), flush=True)

_OVERRIDE_PATH = os.environ.get(
    "OV_RUNTIME_OVERRIDES",
    "",
)
try:
    with open(_OVERRIDE_PATH, "r", encoding="utf-8") as _override_file:
        _RUNTIME_OVERRIDES = json.load(_override_file).get(MODEL_NAME, {})
except (OSError, ValueError, TypeError):
    _RUNTIME_OVERRIDES = {}


def _setting(name, default):
    return str(_RUNTIME_OVERRIDES.get(name, os.environ.get(name, default)))


REP_PEN    = float(_setting("OV_REP_PEN", "1.2"))
# Some creative merges lose names and scene anchors under an aggressive
# repetition penalty.  A per-model ceiling lets the registry protect those
# models even when an older CHIM connector still submits 1.2.
REP_PEN_MAX = float(_setting("OV_REP_PEN_MAX", "0") or "0")
# Optional per-model sampling defaults.  A zero temperature retains the old
# greedy behavior for models that do not opt in.  Explicit client values always
# win, including an explicit temperature of zero for deterministic tests.
TEMPERATURE_DEFAULT = float(_setting("OV_TEMPERATURE_DEFAULT", "0") or "0")
TOP_P_DEFAULT = float(_setting("OV_TOP_P_DEFAULT", "0.95") or "0.95")
TOP_K_DEFAULT = int(_setting("OV_TOP_K_DEFAULT", "50") or "50")
MIN_P_DEFAULT = float(_setting("OV_MIN_P_DEFAULT", "0") or "0")
THINK      = os.environ.get("OV_THINK", "0") == "1"      # 0 = suppress reasoning (fast, default), 1 = enable thinking
# OV_CONTEXT_TOKENS is a per-card override from ovhub's Arch-Mage slider.
# Keep it separate from the model-wide runtime_overrides.json so the same model
# can run at different windows on the two physical cards.
MAX_CTX_TOKENS = int(os.environ.get("OV_CONTEXT_TOKENS", _setting("OV_MAX_CTX_TOKENS", "20000")))
THINK_HEADROOM = int(os.environ.get("OV_THINK_HEADROOM", "1024"))    # when thinking, add this so client max_tokens = ANSWER length
_CTX_MARGIN_TOKENS = int(os.environ.get("OV_CTX_MARGIN_TOKENS", "128"))
_MAX_NEW_TOKENS = int(os.environ.get("OV_MAX_NEW_TOKENS", "4096"))
_MIN_OUTPUT_RESERVE = int(os.environ.get("OV_MIN_OUTPUT_RESERVE", "4096"))

core = ov.Core()
try: PCI = str(core.get_property(DEVICE, "DEVICE_PCI_INFO"))
except Exception as e: PCI = "unknown:%s" % e
print("[ov] device %s PCI=%s" % (DEVICE, PCI), flush=True)
if EXPECT_BUS and EXPECT_BUS not in PCI:
    raise SystemExit("[ov] ABORT: %s PCI '%s' missing expected bus token '%s'" % (DEVICE, PCI, EXPECT_BUS))

print("[ov] loading %s on %s ..." % (MODEL, DEVICE), flush=True)
_t0 = time.time()
# OV_PREFIX_CACHE=<GB> switches to the continuous-batching backend with prefix
# caching: multi-turn chats reuse the KV of earlier turns instead of
# re-prefilling the whole history. Validated per model before enabling in the
# registry. Gemma 12 still uses the plain pipeline; its text path is validated
# with DQGS 128, while its separate AV worker remains on DQGS 0.
_PFX_GB = int(_setting("OV_PREFIX_CACHE", "0") or "0")
_MAX_BATCHED_TOKENS = int(_setting("OV_MAX_BATCHED_TOKENS", "0") or "0")
_MAX_NUM_SEQS = int(_setting("OV_MAX_NUM_SEQS", "1") or "1")
# OV_DQGS: dynamic-quantization group size. 0 (the fleet default) keeps DQ OFF,
# which is the long-standing safe setting (diffusion s8xs4 kernel bug). Nonzero
# values (32/64/...) quantize activations on the fly to feed the XMX int8 path;
# opt in PER MODEL via the registry env after benching coherence + speed.
_DQGS = int(_setting("OV_DQGS", "0") or "0")
_KV_CACHE_PRECISION = _setting("OV_KV_CACHE_PRECISION", "").strip().lower()
if _KV_CACHE_PRECISION not in ("", "f16", "bf16", "u8", "i8", "u4", "i4"):
    raise SystemExit("[ov] ABORT: unsupported OV_KV_CACHE_PRECISION=%r" % _KV_CACHE_PRECISION)
print("[ov] DYNAMIC_QUANTIZATION_GROUP_SIZE=%d" % _DQGS, flush=True)
print("[ov] KV_CACHE_PRECISION=%s" % (_KV_CACHE_PRECISION or "auto"), flush=True)
_is_vlm = os.path.isfile(os.path.join(MODEL, "openvino_language_model.xml"))
_has_vision_files = (_is_vlm and
                     os.path.isfile(os.path.join(MODEL, "openvino_vision_embeddings_model.xml")) and
                     os.path.isfile(os.path.join(MODEL, "openvino_vision_embeddings_model.bin")))
_vision_enabled = False  # Portable text-only server; AV requires separate validation.
_has_vision = _has_vision_files and _vision_enabled
if _has_vision:
    _vision_reason = ""
elif _has_vision_files:
    _vision_reason = _setting(
        "OV_NATIVE_VISION_REASON",
        "This model contains a vision graph, but native vision has not passed validation in this serving runtime.",
    )
else:
    _vision_reason = "The loaded model does not contain native vision."
_has_audio = False  # No public audio request API.
_Pipe = g.VLMPipeline if _is_vlm else g.LLMPipeline
print("[ov] pipeline=%s native_vision=%s native_audio=%s" %
      ("VLM" if _is_vlm else "LLM-text", _has_vision, _has_audio), flush=True)
_PIPELINE_PROPERTIES = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": _DQGS}
if _KV_CACHE_PRECISION:
    _PIPELINE_PROPERTIES["KV_CACHE_PRECISION"] = _KV_CACHE_PRECISION
if _PFX_GB > 0:
    _sched = g.SchedulerConfig()
    _sched.enable_prefix_caching = True
    _sched.cache_size = _PFX_GB
    # Requests are serialized by GEN_LOCK.  Advertising hundreds of scheduler
    # sequences only gives the CB backend permission to reserve memory that can
    # never be used by this server.
    _sched.max_num_seqs = _MAX_NUM_SEQS
    if _MAX_BATCHED_TOKENS > 0:
        _sched.max_num_batched_tokens = _MAX_BATCHED_TOKENS
    print(
        "[ov] prefix caching ON (cache %dGB, max batch %d, max seqs %d, CB backend)"
        % (_PFX_GB, _sched.max_num_batched_tokens, _sched.max_num_seqs),
        flush=True,
    )
    pipe = _Pipe(MODEL, DEVICE, scheduler_config=_sched, **_PIPELINE_PROPERTIES)
else:
    pipe = _Pipe(MODEL, DEVICE, **_PIPELINE_PROPERTIES)
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
print("[ov] LOADED in %.1fs -- serving %s on :%d (single-stream, rep_pen=%.2f, sampler_default=t%.2f/p%.2f/k%d/min%.3f, thinking=%s, max_ctx=%d tok)" % (time.time() - _t0, MODEL_NAME, PORT, REP_PEN, TEMPERATURE_DEFAULT, TOP_P_DEFAULT, TOP_K_DEFAULT, MIN_P_DEFAULT, "ON" if THINK else "OFF", MAX_CTX_TOKENS), flush=True)

class AttachmentError(ValueError):
    pass


_MAX_IMAGE_BYTES = 14 * 1024 * 1024


def _image_source(part):
    if not isinstance(part, dict):
        return None
    if part.get("type") not in ("image", "image_url", "input_image"):
        return None
    src = part.get("image_url")
    if isinstance(src, dict):
        src = src.get("url")
    if not src:
        src = part.get("url") or part.get("image") or part.get("data")
    return src if isinstance(src, str) and src else None


def _message_budget_len(message):
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(content, list):
        return len(str(content))
    total = 0
    for part in content:
        if not isinstance(part, dict):
            total += len(str(part))
        elif part.get("type") in ("text", "input_text"):
            total += len(str(part.get("text", "")))
        elif _image_source(part):
            # Gemma-4 uses 280 soft tokens per image.  Count those tokens, not
            # a multi-megabyte base64 URL, against the text context guard.
            total += 280 * 4
    return total


def cap_context(messages, reserve_tokens=0, thinking=None):
    """Slide whole old turns out until the *tokenized* prompt fits.

    The old four-characters-per-token estimate undercounted CHIM's markdown,
    speaker tags and prompt wrappers.  A nominal 24K request could therefore
    reach OpenVINO larger than 24K and make Xe/TTM evict VRAM into host RAM.
    Count the exact rendered tokens and retain the system prompt plus the most
    recent contiguous conversation instead.
    """
    messages = list(messages or [])
    # Context length is prompt + generation.  Reserve the server's normal 4K
    # answer allowance even if a client advertises an unrealistically tiny
    # max_tokens value; otherwise it can consume the full window during
    # prefill and recreate the transient VRAM spike this guard exists to stop.
    reserved_output = max(0, reserve_tokens, _MIN_OUTPUT_RESERVE)
    token_budget = MAX_CTX_TOKENS - reserved_output - _CTX_MARGIN_TOKENS
    if token_budget < 1:
        raise ValueError("requested output leaves no room in the %d-token context" % MAX_CTX_TOKENS)

    def count(render_messages):
        encoded = TOK.encode(
            build_prompt(render_messages, thinking=thinking),
            add_special_tokens=False,
        )
        return int(encoded.input_ids.shape[-1])

    original_tokens = count(messages)
    if original_tokens <= token_budget:
        return messages

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    rest     = [m for m in messages if m.get("role") != "system"]
    if count(sys_msgs) > token_budget:
        raise ValueError("system prompt alone exceeds the safe %d-token input budget" % token_budget)

    kept = []
    for m in reversed(rest):
        candidate = sys_msgs + [m] + kept
        if count(candidate) <= token_budget:
            kept.insert(0, m)
        else:
            break

    if not kept and rest:
        # The newest turn can itself be enormous (CHIM current-context dumps).
        # Keep its tail, where the current location/instruction normally lives,
        # and binary-search the largest safe tail instead of forwarding an OOM.
        latest = dict(rest[-1])
        content = latest.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            content = str(content)
        lo, hi, best = 0, len(content), None
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = dict(latest)
            trial["content"] = content[-mid:] if mid else ""
            if count(sys_msgs + [trial]) <= token_budget:
                best = trial
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise ValueError("newest turn cannot fit beside the system prompt")
        kept = [best]

    result = sys_msgs + kept
    final_tokens = count(result)
    if final_tokens > token_budget:
        raise ValueError("context slider failed its hard token limit")
    print(
        "[ov] context slide: %d -> %d input tokens; dropped %d old turn(s); "
        "reserved %d output + %d guard"
        % (original_tokens, final_tokens, len(rest) - len(kept), reserved_output, _CTX_MARGIN_TOKENS),
        flush=True,
    )
    return result


def _read_image_bytes(source):
    if source.startswith("data:"):
        try:
            header, payload = source.split(",", 1)
            if ";base64" not in header.lower():
                raise AttachmentError("image data URL is not base64 encoded")
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as e:
            raise AttachmentError("invalid image data: %s" % str(e)[:120])
    elif source.startswith(("http://", "https://")):
        raise AttachmentError("remote image URLs are disabled")
    else:
        try:
            raw = base64.b64decode(source, validate=True)
        except (ValueError, binascii.Error) as e:
            raise AttachmentError("unsupported image source: %s" % str(e)[:120])
    if not raw:
        raise AttachmentError("empty image")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise AttachmentError("image exceeds the 14 MB limit")
    return raw


def _decode_image(source):
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError:
        raise AttachmentError("native vision decoder is not installed for this model")
    try:
        Image.MAX_IMAGE_PIXELS = 40_000_000
        with Image.open(io.BytesIO(_read_image_bytes(source))) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if max(image.size) > 4096:
                image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            # OpenVINO GenAI's VLM API expects one HWC tensor per image.  A
            # leading batch dimension is accepted by some older paths but the
            # Gemma-4 PA adapter silently returns zero output tokens for it.
            array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        return ov.Tensor(array)
    except AttachmentError:
        raise
    except Exception as e:
        raise AttachmentError("unreadable image: %s" % str(e)[:120])


def prepare_vision_messages(messages):
    prepared, images = [], []
    for message in messages:
        item = dict(message)
        content = message.get("content", "")
        if isinstance(content, list):
            rendered = []
            for part in content:
                source = _image_source(part)
                if source:
                    if not _has_vision:
                        raise AttachmentError(_vision_reason)
                    index = len(images)
                    images.append(_decode_image(source))
                    rendered.append("<ov_genai_image_%d>" % index)
                elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                    rendered.append(str(part.get("text", "")))
            item["content"] = "\n".join(x for x in rendered if x)
        prepared.append(item)
    return prepared, images


def native_vision_caption(source, prompt):
    """Run the loaded model's own vision tower through GenAI's documented
    ChatHistory API.  Keep this separate from the hand-rendered RP text prompt:
    the Gemma-4 PA adapter can return an empty result when visual embeddings are
    paired with an already-rendered prompt on its continuous-batching path.
    """
    if not _has_vision:
        raise AttachmentError(_vision_reason)
    history = g.ChatHistory()
    history.append({"role": "user", "content": str(prompt)})
    config = g.GenerationConfig()
    config.max_new_tokens = 256
    config.repetition_penalty = REP_PEN
    with GEN_LOCK:
        result = pipe.generate(history, images=[_decode_image(source)],
                               generation_config=config)
    text = result.texts[0] if getattr(result, "texts", None) else ""
    print("[ov] native vision completed images=1 output_chars=%d" % len(text), flush=True)
    return strip_thinking(text)

FMT = os.environ.get("OV_FMT", "gemma")   # gemma = <|turn> format, qwen = <|im_start|> format

def build_prompt_qwen(messages, thinking=None):
    # Standard Qwen format: <|im_start|>role ... <|im_end|>. No-think via pre-closed <think> block.
    thinking = THINK if thinking is None else bool(thinking)
    p = ""
    for m in messages:
        r = m.get("role", "user")
        c = m.get("content", "")
        if isinstance(c, list):
            c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
        c = str(c).strip()
        if r not in ("system", "user", "assistant"):
            r = "user"
        p += "<|im_start|>" + r + "\n" + c + "<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    if not thinking:
        p += "<think>\n\n</think>\n\n"
    return p

def build_prompt_mistral(messages):
    # Mistral Nemo: [INST] user [/INST] assistant</s>. System folded into the first user turn.
    sys_txt, turns = "", []
    for m in messages:
        role = m.get("role", "user"); content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
        content = str(content).strip()
        if role == "system":
            sys_txt += content + chr(10) + chr(10); continue
        turns.append(("assistant" if role == "assistant" else "user", content))
    out = "<s>"; first = True
    for r, c in turns:
        if r == "user":
            u = (sys_txt + c) if (first and sys_txt) else c
            first = False
            out += "[INST] " + u.strip() + " [/INST]"
        else:
            out += " " + c + "</s>"
    return out


def build_prompt_muse(messages):
    # Muse Glimmer uses recipient channels.  The production chat path is a
    # direct user answer at the model's lowest supported reasoning strength;
    # unlike Gemma/Qwen, Muse has no true thinking-off switch.
    system_parts, turns = [], []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        content = str(content).strip()
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        turns.append(("assistant" if role == "assistant" else "user", content))

    system_text = "\n\n".join(system_parts) or "You are an AI assistant."
    prompt = (
        "<|begin_of_text|><|start|>system<|message|>"
        + system_text
        + "\n\nReasoning strength: low."
        + "\n\n# Valid recipients: \"self\", \"user\".<|eot|>"
    )
    for role, content in turns:
        if role == "assistant":
            prompt += "<|start|>assistant to=user<|message|>" + content + "<|eot|>"
        else:
            prompt += "<|start|>user<|message|>" + content + "<|eot|>"
    return prompt + "<|start|>assistant to=user<|message|>"


def build_prompt(messages, thinking=None):
    thinking = THINK if thinking is None else bool(thinking)
    if FMT == "mistral":
        return build_prompt_mistral(messages)
    if FMT == "qwen":
        return build_prompt_qwen(messages, thinking=thinking)
    if FMT == "muse":
        return build_prompt_muse(messages)
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
    if sys_txt.strip() or thinking:
        p += "<|turn>system\n"
        if thinking: p += "<|think|>\n"
        p += sys_txt.strip() + "<turn|>\n"
    for r, c in turns:
        p += "<|turn>%s\n%s<turn|>\n" % (r, c)
    p += "<|turn>model\n"
    if not thinking:
        p += "<|channel>thought\n<channel|>"   # empty pre-closed thought channel = skip reasoning
    return p

# Backstops for models that ignore a pre-closed thought channel.  An unfinished
# thought is never safe to expose: it is rejected rather than heuristically
# converted into an answer.
_GEMMA_THOUGHT = re.compile(
    r"^\s*<\|channel>\s*thought\s*\n.*?<channel\|>\s*",
    re.DOTALL | re.IGNORECASE,
)
_XML_THOUGHT = re.compile(
    r"^\s*<(think|thought|reasoning)>.*?</\1>\s*",
    re.DOTALL | re.IGNORECASE,
)
_THOUGHT_LINE = re.compile(r"^[\s:>*_|-]*thought[\s:>*_-]*\n", re.IGNORECASE)
_REASONING_MARKER = re.compile(
    r"<\|channel>\s*thought|<channel\|>|<\|think\|>|"
    r"</?(?:think|thought|reasoning)>|<\|start\|>assistant\s+to=self|"
    r"assistant\s+to=self<\|message\|>",
    re.IGNORECASE,
)


class ReasoningBoundaryError(RuntimeError):
    pass


def strip_thinking(t):
    t = (t or "").strip()
    t = _GEMMA_THOUGHT.sub("", t, count=1).strip()
    t = _XML_THOUGHT.sub("", t, count=1).strip()
    t = _THOUGHT_LINE.sub("", t, count=1).strip()
    return t


def clean_visible_answer(text):
    text = strip_thinking(text)
    for marker in (
        "<turn|>", "<eos>", "<pad>", "<bos>",
        "<|eot|>", "<|end_of_text|>", "<|begin_of_text|>",
    ):
        text = text.replace(marker, "")
    text = text.strip()
    if _REASONING_MARKER.search(text):
        raise ReasoningBoundaryError("reasoning control marker remained in visible output")
    return text


def extract_reasoned_answer(full):
    """Return only a completed answer, or None when thought never closed."""
    full = full or ""
    if FMT == "gemma":
        close = full.find("<channel|>")
        if close < 0:
            return None
        return clean_visible_answer(full[close + len("<channel|>"):])
    if FMT == "qwen":
        match = re.search(r"</think>", full, re.IGNORECASE)
        if not match:
            return None
        return clean_visible_answer(full[match.end():])
    # No validated hidden-reasoning wire format exists for this formatter.
    return None


def make_cfg(req, thinking=None):
    thinking = THINK if thinking is None else bool(thinking)
    c = g.GenerationConfig()
    mt = int(req.get("max_tokens") or req.get("max_new_tokens") or req.get("max_completion_tokens") or 512)
    if thinking:
        mt += THINK_HEADROOM   # client max_tokens remains the visible-answer allowance
    mt = max(1, min(mt, _MAX_NEW_TOKENS))
    c.max_new_tokens = mt
    t = req.get("temperature")
    if t is None:
        t = TEMPERATURE_DEFAULT
    if float(t) > 0:
        c.do_sample = True; c.temperature = float(t)
        top_p = req.get("top_p")
        top_k = req.get("top_k")
        min_p = req.get("min_p")
        try: c.top_p = float(TOP_P_DEFAULT if top_p is None else top_p)
        except Exception: pass
        try: c.top_k = int(TOP_K_DEFAULT if top_k is None else top_k)
        except Exception: pass
        try: c.min_p = float(MIN_P_DEFAULT if min_p is None else min_p)
        except Exception: pass
    try:
        requested_rep = float(req.get("repetition_penalty") or req.get("repeat_penalty") or REP_PEN)
        c.repetition_penalty = min(requested_rep, REP_PEN_MAX) if REP_PEN_MAX > 0 else requested_rep
    except Exception: pass
    for k in ("frequency_penalty", "presence_penalty"):   # client-tunable from CHIM connector
        if req.get(k) is not None:
            try: setattr(c, k, float(req[k]))
            except Exception: pass
    try: c.apply_chat_template = False
    except Exception: pass
    return c


def _pipe_generate(prompt, cfg, images, streamer=None):
    kwargs = {"generation_config": cfg}
    if images:
        kwargs["images"] = images
    if streamer is not None:
        kwargs["streamer"] = streamer
    return pipe.generate(prompt, **kwargs)


def _result_text(result):
    for attr in ("texts", "m_generation_ids"):
        try:
            value = getattr(result, attr)
            if value:
                return value[0]
        except Exception:
            pass
    return str(result)


def generate(messages, req):
    cfg = make_cfg(req, thinking=THINK)
    reserve_tokens = int(getattr(cfg, "max_new_tokens", 512) or 512)
    messages = cap_context(messages, reserve_tokens=reserve_tokens, thinking=THINK)
    messages, images = prepare_vision_messages(messages)
    prompt = build_prompt(messages, thinking=THINK)
    if THINK:
        # Keep special tokens so the thought-close can be proven. If it never
        # appears, discard the entire trace and retry once through the native
        # no-think prompt. Nothing from an unfinished trace can reach a client.
        col = _Collector()
        with GEN_LOCK:
            _pipe_generate(prompt, cfg, images, streamer=col)
            full = TOK.decode(col.toks, skip_special_tokens=False) if col.toks else ""
            answer = extract_reasoned_answer(full)
            if answer:
                return answer
            print(
                "[ov] reasoning boundary missing/empty after %d tokens; "
                "discarding trace and retrying no-think" % len(col.toks),
                flush=True,
            )
            fallback_cfg = make_cfg(req, thinking=False)
            fallback_prompt = build_prompt(messages, thinking=False)
            fallback = _pipe_generate(fallback_prompt, fallback_cfg, images)
        answer = clean_visible_answer(_result_text(fallback))
        if not answer:
            raise ReasoningBoundaryError("no visible answer after no-think retry")
        return answer
    with GEN_LOCK:
        result = _pipe_generate(prompt, cfg, images)
    answer = clean_visible_answer(_result_text(result))
    if not answer:
        raise ReasoningBoundaryError("model returned an empty visible answer")
    return answer

import hmac
API_KEY = os.environ.get("OV_API_KEY", "")

class H(BaseHTTPRequestHandler):
    def _authorized(self):
        if API_KEY and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + API_KEY):
            self._json({"error": "unauthorized"}, 401)
            return False
        return True

    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"}); return
        if not self._authorized(): return
        if self.path.rstrip("/") == "/v1/models":
            self._json({"object": "list", "data": [{"id": MODEL_NAME, "object": "model",
                        "owned_by": "ov", "native_vision": _has_vision,
                        "native_audio": _has_audio, "vision_reason": _vision_reason}]})
        else:
            self._json({"status": "ok", "model": MODEL_NAME, "device": DEVICE,
                        "native_vision": _has_vision, "native_audio": _has_audio,
                        "vision_reason": _vision_reason})
    def do_POST(self):
        if not self._authorized(): return
        self.connection.settimeout(30)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2097152 or self.headers.get("Transfer-Encoding"):
                raise ValueError()
        except ValueError:
            self._json({"error": "Content-Length must be 1..2097152 bytes"}, 413); return
        if self.path.rstrip("/") == "/av":
            self._json({"error": "portable package is text-only"}, 409); return
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/v1/completions"):
            self._json({"error": "not found"}, 404); return
        try:
            ln = int(self.headers.get("Content-Length", "0") or 0)
            req = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            self._json({"error": "bad json"}, 400); return
        if not isinstance(req, dict):
            self._json({"error": "request must be an object"}, 400); return
        if req.get("model") not in (None, MODEL_NAME):
            self._json({"error": "requested model is not loaded"}, 404); return
        if any(req.get(k) for k in ("tools", "functions", "response_format", "grammar")):
            self._json({"error": "tools and constrained output are not supported"}, 400); return
        msgs = req.get("messages")
        if msgs is None and "prompt" in req: msgs = [{"role": "user", "content": req["prompt"]}]
        if not isinstance(msgs, list) or not msgs or any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), (str, list)) for m in msgs):
            self._json({"error": "messages must contain system/user/assistant text turns"}, 400); return
        if any(isinstance(m['content'], list) and any(not isinstance(p, dict) or p.get('type') not in ('text', 'input_text') or not isinstance(p.get('text'), str) for p in m['content']) for m in msgs):
            self._json({"error": "portable package accepts text only; image/audio inputs are unsupported"}, 400); return
        try:
            text = generate(msgs, req)
        except (AttachmentError, ValueError, TypeError) as e:
            self._json({"error": str(e)}, 400); return
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
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

HOST = os.environ.get("OV_HOST", "127.0.0.1")
print("[ov] HTTP up on %s:%d" % (HOST, PORT), flush=True)
ThreadingHTTPServer((HOST, PORT), H).serve_forever()
