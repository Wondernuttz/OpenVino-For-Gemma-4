#!/usr/bin/env python3
"""
av_pipeline.py: vision + audio for Gemma-4 12B unified on OpenVINO (Intel Arc), the pipeline
OpenVINO GenAI does not have. GenAI feeds this model's vision tower 16x16 patches where the
unified architecture wants 48x48 merged patches, and never attempts audio. This script drives
the exported IRs directly instead:

  image: transformers Gemma4UnifiedImageProcessor (16px patchify + 3x3 merge -> [N,6912])
         -> vision IR -> soft tokens [N,3840]
  audio: raw 16kHz mono chunked into 640-sample frames (40ms/token) -> RMSNorm (no scale)
         -> single Linear 640->3840 (audio_projection.npy; the unified model has NO audio
         tower by design, this 5MB projection is the entire audio encoder)
  both spliced into the text embedding stream at their placeholder token positions, then a
  manual stateful generate loop drives the language model IR.

Requirements: openvino nightly (2026.3-dev), torch+torchvision (CPU is fine, only used for
preprocessing), and the gemma4_unified processor module from transformers main (vendor
image_processing_gemma4_unified.py into transformers/models/gemma4_unified/ with an empty
__init__.py if your transformers predates it).

Self-tests: --gate 0 text | 1 red square | 2 image (--image PATH) | 3 needle | 4 audio
(--audio PATH, 16kHz mono WAV) | 5 benchmark
"""
import argparse, sys, time, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", type=str, default=".", help="dir with the OpenVINO IRs + audio_projection.npy")
ap.add_argument("--gate", type=int, default=0)
ap.add_argument("--image", type=str, default="")
ap.add_argument("--audio", type=str, default="")
ap.add_argument("--prompt", type=str, default="")
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--device", type=str, default="GPU")
args = ap.parse_args()
MODEL_DIR = args.model_dir

import openvino as ov
core = ov.Core()

print("[1] loading tokenizer + config ...", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
cfg = json.load(open(MODEL_DIR + "/config.json"))
IMAGE_TOKEN_ID = cfg.get("image_token_id")
BOI_ID = cfg.get("boi_token_id")
EOI_ID = cfg.get("eoi_token_id")
AUDIO_TOKEN_ID = cfg.get("audio_token_id")
BOA_ID = cfg.get("boa_token_id")
EOA_ID = cfg.get("eoa_token_id") or cfg.get("eoa_token_index")
EOS_IDS = cfg.get("eos_token_id")
EOS_IDS = EOS_IDS if isinstance(EOS_IDS, list) else [EOS_IDS]
print("    image_token=%s boi=%s eoi=%s eos=%s" % (IMAGE_TOKEN_ID, BOI_ID, EOI_ID, EOS_IDS), flush=True)

print("[2] compiling IRs on %s ..." % args.device, flush=True)
DQ0 = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}   # mandatory for the 12B
emb_c = core.compile_model(MODEL_DIR + "/openvino_text_embeddings_model.xml", args.device, DQ0)
lm_m = core.read_model(MODEL_DIR + "/openvino_language_model.xml")
lm_c = core.compile_model(lm_m, args.device, DQ0)
print("    LM inputs :", [ (p.get_friendly_name(), list(p.get_partial_shape())) for p in lm_m.get_parameters() ], flush=True)
print("    LM outputs:", [ o.get_any_name() for o in lm_m.outputs ][:3], flush=True)
vis_c = None
if args.gate in (1, 2, 5):
    vis_c = core.compile_model(MODEL_DIR + "/openvino_vision_embeddings_model.xml", args.device, DQ0)

EMB_IN = emb_c.input(0)
print("    embed model input:", emb_c.input(0).get_names(), flush=True)

def embed_ids(ids):
    a = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    return emb_c({ EMB_IN: a })[emb_c.output(0)]

def vision_soft_tokens(img):
    """img: PIL.Image -> (soft_tokens [1,N,3840])"""
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor
    print("[3] preprocessing image ...", flush=True)
    ip = Gemma4UnifiedImageProcessor.from_pretrained(MODEL_DIR)
    out = ip(images=[img], return_tensors="pt")
    n = int(out["num_soft_tokens_per_image"][0])
    pv = out["pixel_values"][:, :n].numpy().astype(np.float32)
    ipos = out["image_position_ids"][:, :n].numpy().astype(np.int64)
    print("    %d soft tokens (trimmed from %d); pv %s, pos %s" % (n, out["pixel_values"].shape[1], pv.shape, ipos.shape), flush=True)
    feed = {}
    for port in vis_c.inputs:
        names = " ".join(port.get_names())
        feed[port] = ipos if "position" in names else pv
    print("    vision feed:", {" ".join(p.get_names()): v.shape for p, v in feed.items()}, flush=True)
    r = vis_c(feed)
    st = r[vis_c.output(0)]
    print("    vision soft tokens:", st.shape, flush=True)
    return st

def audio_soft_tokens(wav_path):
    """wav 16kHz mono -> (soft_tokens [1,N,3840]) via RMSNorm(frame) @ W.T (no audio tower by design)"""
    import wave
    print("[3] preprocessing audio ...", flush=True)
    w = wave.open(wav_path, "rb")
    assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16kHz mono, got %d Hz %d ch" % (w.getframerate(), w.getnchannels())
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    spf = 640                                   # audio_samples_per_token, 40ms at 16kHz
    n = (len(pcm) + spf - 1) // spf
    pcm = np.pad(pcm, (0, n * spf - len(pcm)))
    frames = pcm.reshape(n, spf)
    W = np.load(MODEL_DIR + "/audio_projection.npy")   # [3840, 640] f32 from bf16
    normed = frames * np.power((frames ** 2).mean(-1, keepdims=True) + 1e-6, -0.5)
    st = (normed @ W.T)[None].astype(np.float32)
    print("    %.1fs audio -> %d frames -> soft tokens %s" % (len(pcm) / 16000.0, n, st.shape), flush=True)
    return st

def build_sequence(user_text, soft, modality="image"):
    """Returns (embeds [1,S,3840], token_type_ids [1,S]) with soft tokens spliced if given."""
    pre = "<bos><|turn>user\n"
    post = user_text + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    if soft is None:
        ids = pre_ids + post_ids
        emb = embed_ids(ids)
        tt = np.zeros((1, emb.shape[1]), dtype=np.int64)
        return emb, tt
    if modality == "audio":
        bo, tok_id, eo = BOA_ID, AUDIO_TOKEN_ID, EOA_ID
    else:
        bo, tok_id, eo = BOI_ID, IMAGE_TOKEN_ID, EOI_ID
    n = soft.shape[1]
    mid_ids = ([bo] if bo else []) + [tok_id] * n + ([eo] if eo else [])
    ids = pre_ids + mid_ids + post_ids
    emb = embed_ids(ids)
    tt = np.zeros((1, emb.shape[1]), dtype=np.int64)
    s = len(pre_ids) + (1 if bo else 0)
    emb[0, s:s + n, :] = soft[0]
    tt[0, s:s + n] = 1
    print("    spliced %d %s soft tokens at positions %d..%d (seq %d)" % (n, modality, s, s + n - 1, emb.shape[1]), flush=True)
    return emb, tt

LM_PORT = {}
for _p in lm_c.inputs:
    _n = " ".join(_p.get_names()) + " " + _p.get_node().get_friendly_name()
    for key in ("inputs_embeds", "attention_mask", "position_ids", "token_type_ids", "beam_idx"):
        if key in _n:
            LM_PORT[key] = _p
assert len(LM_PORT) == 5, "unmapped LM inputs: " + str([p.get_names() for p in lm_c.inputs])

def lm_feed(emb, mask_len, pos, tt):
    return {
        LM_PORT["inputs_embeds"]: emb.astype(np.float32),
        LM_PORT["attention_mask"]: np.ones((1, mask_len), dtype=np.int64),
        LM_PORT["position_ids"]: pos,
        LM_PORT["token_type_ids"]: tt,
        LM_PORT["beam_idx"]: np.zeros((1,), dtype=np.int32),
    }

def generate(emb, tt, max_new):
    req = lm_c.create_infer_request()
    req.reset_state()
    S = emb.shape[1]
    t0 = time.time()
    res = req.infer(lm_feed(emb, S, np.arange(S, dtype=np.int64).reshape(1, -1), tt))
    logits = res[lm_c.output(0)]
    ttft = time.time() - t0
    nxt = int(np.argmax(logits[0, -1]))
    out_ids = [nxt]
    pos = S
    t1 = time.time()
    while len(out_ids) < max_new and out_ids[-1] not in EOS_IDS:
        e = embed_ids([out_ids[-1]])
        res = req.infer(lm_feed(e, pos + 1, np.array([[pos]], dtype=np.int64), np.zeros((1, 1), dtype=np.int64)))
        out_ids.append(int(np.argmax(res[lm_c.output(0)][0, -1])))
        pos += 1
    dt = time.time() - t1
    txt = tok.decode([i for i in out_ids if i not in EOS_IDS], skip_special_tokens=True)
    print("    TTFT %.2fs | %d tokens in %.2fs (%.1f tok/s)" % (ttft, len(out_ids), dt, (len(out_ids) - 1) / max(dt, 0.01)), flush=True)
    return txt

# ---------------- gates ----------------
if args.gate == 0:
    print("[GATE 0] text-only through the manual loop", flush=True)
    emb, tt = build_sequence(args.prompt or "Say hello in one short sentence.", None)
    print("OUTPUT: %r" % generate(emb, tt, args.max_new), flush=True)

elif args.gate in (1, 2):
    from PIL import Image
    if args.gate == 1:
        img = Image.new("RGB", (224, 224), (220, 30, 30))
        q = args.prompt or "What color is this image? Answer with one word."
    elif args.image:
        img = Image.open(args.image).convert("RGB")
        q = args.prompt or "Describe this image in two sentences."
    else:
        from PIL import ImageDraw
        img = Image.new("RGB", (480, 480), (110, 170, 235))          # sky
        d = ImageDraw.Draw(img)
        d.rectangle([0, 330, 480, 480], fill=(70, 150, 60))           # grass
        d.ellipse([340, 40, 440, 140], fill=(250, 220, 60))           # sun
        q = args.prompt or "Describe this image in two sentences."
    soft = vision_soft_tokens(img)
    emb, tt = build_sequence(q, soft)
    print("OUTPUT: %r" % generate(emb, tt, args.max_new), flush=True)

elif args.gate == 3:
    print("[GATE 3] speed + text regression", flush=True)
    emb, tt = build_sequence("The Stormcloak courier password is AMBERFROST-92. What is the password? Just the password.", None)
    print("OUTPUT: %r" % generate(emb, tt, 24), flush=True)

elif args.gate == 4:
    print("[GATE 4] audio: transcribe TTS speech", flush=True)
    soft = audio_soft_tokens(args.audio)
    q = args.prompt or "What does the speaker say? Transcribe the speech exactly."
    emb, tt = build_sequence(q, soft, modality="audio")
    print("OUTPUT: %r" % generate(emb, tt, args.max_new), flush=True)

elif args.gate == 5:
    print("[GATE 5] PP/TPS bench through the AV pipeline", flush=True)
    filler = "The dragon circled the ruined tower while the wind carried ash across the valley. "
    fill_ids = tok(filler, add_special_tokens=False)["input_ids"]
    for target in (512, 2048, 6144):
        reps = max(1, (target - 24) // len(fill_ids))
        text = ("run%d " % target) + filler * reps + "Summarize in one word."
        t0 = time.time()
        emb, tt = build_sequence(text, None)
        t_embed = time.time() - t0
        req = lm_c.create_infer_request(); req.reset_state()
        S = emb.shape[1]
        t0 = time.time()
        req.infer(lm_feed(emb, S, np.arange(S, dtype=np.int64).reshape(1, -1), tt))
        dt = time.time() - t0
        print("    text prefill %5d tok: %6.2fs infer (embed %.2fs) = %6.0f tok/s" % (S, dt, t_embed, S / dt), flush=True)
    # decode speed: 128 tokens after a 512-token prefill
    emb, tt = build_sequence("bench-decode " + filler * 6 + "Tell me a short story about a fox.", None)
    txt = generate(emb, tt, 128)
    # multimodal TTFT: image and audio end-to-end
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (960, 540), (110, 170, 235)); d = ImageDraw.Draw(img)
    d.rectangle([0, 380, 960, 540], fill=(70, 150, 60)); d.ellipse([700, 40, 830, 170], fill=(250, 220, 60))
    t0 = time.time(); soft = vision_soft_tokens(img); t_vis = time.time() - t0
    emb, tt = build_sequence("Describe briefly.", soft)
    generate(emb, tt, 32)
    print("    image path: preprocess+vision IR %.2fs, then TTFT/decode above" % t_vis, flush=True)
    if args.audio:
        t0 = time.time(); soft = audio_soft_tokens(args.audio); t_aud = time.time() - t0
        emb, tt = build_sequence("Transcribe.", soft, modality="audio")
        generate(emb, tt, 48)
        print("    audio path: preprocess+project %.2fs, then TTFT/decode above" % t_aud, flush=True)

print("GATE %d DONE" % args.gate, flush=True)
