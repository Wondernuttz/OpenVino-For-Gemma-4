# =============================================================================
# DiffusionGemma-26B-A4B-it-abliterated  ->  OpenVINO int4  (Phase 1 Colab export)
# Block-diffusion 26B-A4B MoE, world-first OpenVINO port. Driven by a numpy
# block-diffusion sampler on the box (Intel Arc).
#
# Produces TWO stateless OpenVINO IRs + tokenizer/config/manifest:
#   IR-E  cache-builder  (causal encoder, one pass per committed block)
#   IR-D  denoiser       (bidirectional decoder + lm_head + fp32 softcap, traced at
#                         DiffusionGemmaForBlockDiffusion.forward with input_ids=None)
#
# We DO NOT reimplement the exotic math (scaling=1.0, v_proj=None global shared-KV,
# p-RoPE, QK-norm, embed_scale, self-conditioning, final softcap). We trace the REAL
# model.forward, which already contains all of it. Our job is (a) make the MoE expert
# loop traceable, (b) split the graph correctly, and (c) hand the numpy sampler an
# interface it can drive with explicit attention masks + KV tensors.
#
# This script incorporates the 2026-07-07 adversarial review (3 blockers + 7 majors)
# and the AUTHORITATIVE DESIGN DECISIONS D1..D8 (see DIFFUSION_GEMMA_OV_SPEC.md).
# Verified against modeling_diffusion_gemma.py / modular_diffusion_gemma.py /
# generation_diffusion_gemma.py / convert_diffusion_gemma_weights.py.
#
# KEY VERIFIED FACTS (override the stale library defaults):
#   hidden_size 2816, num_attention_heads 16, num_hidden_layers 30, vocab 262144
#   SLIDING layers: head_dim 256, num_key_value_heads 8  (NOT 4; config default is
#                   stale) -> kv tensor [B, 8, S, 256]
#   GLOBAL  layers: global_head_dim 512, num_global_key_value_heads 2 -> [B, 2, S, 512]
#   layer_types: full_attention at 0-indexed 5,11,17,23,29 (last forced full); 5:1
#   dense MLP width (also SC MLP width) intermediate_size=2112 ; moe_intermediate=704
#   num_experts 128, top_k 8 ; final_logit_softcapping 30.0 (fp32) ; rms_eps 1e-6
#   embed_scale = sqrt(2816) ~= 53.07 (baked in-graph) ; embed tied to lm_head
#   Decoder: is_causal=False everywhere. Reference windows sliding layers by PHYSICALLY
#     capping the sliding KV cache to sliding_window(1024); we reproduce that exactly by
#     keeping a UNIFORM full-length cache and feeding IR-D an explicit windowed sliding
#     mask (masked keys contribute nothing == evicted keys). See D1 below.
#   SC feedback (verified generation L1070-1071): next-step self_conditioning_logits =
#     processed_logits.to(embed_tokens.weight.dtype == bf16). The fp32 softmax happens
#     INSIDE the decoder on that bf16 input (L1279). So bf16 SC input is CORRECT parity.
#   Committed token = argmax(processed_logits) (temperature-invariant). Entropy /
#     multinomial / SC-feedback all use the SAME processed = raw_softcapped / temperature.
# =============================================================================


# ===== CELL 0 : deps =========================================================
# diffusion_gemma DOES ship in released transformers (>=5.11; the checkpoint was saved with
# 5.11.0, latest release is 5.13.0). The old note here said "main-only" -- that was stale,
# true only back when the newest release was 4.57.0. Use the STABLE release: it is a fast
# wheel install (git+main is a slow source build that can be broken on any given day) and it
# matches the checkpoint era, so the weight keys + submodule names the export cells poke at
# line up. If a later export cell trips on an internal attribute name, pin ==5.11.0.
#
# NOTE: diffusion_gemma is NOT wired into the transformers Auto* mapping even in the release
# (verified at v5.13.0), so AutoConfig/AutoModel/AutoTokenizer.from_pretrained ALWAYS raise
# KeyError 'diffusion_gemma'. That is expected -- CELL 1 loads the concrete classes directly
# and registers the config by hand to work around it. Do not "fix" it by upgrading harder.
import subprocess, sys
def pip(*a): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *a])
pip("--upgrade", "pip")
pip("--upgrade", "transformers>=5.11,<6", "accelerate", "sentencepiece", "safetensors")
pip("openvino==2025.3.0", "nncf==2.18.0")
pip("huggingface_hub")

# !!! COLAB GOTCHA -- the #1 cause of "KeyError: 'diffusion_gemma'" in CELL 1 !!!
# Colab preloads an OLDER transformers before this cell runs, and the freshly installed
# version is NOT picked up until the kernel restarts. After this cell finishes:
#     Runtime > Restart session   (Ctrl+M .)   then run from CELL 1 -- do NOT re-run CELL 0.
print("\n" + "=" * 72 +
      "\nDEPS INSTALLED.  NOW DO:  Runtime > Restart session,  then run CELL 1."
      "\n(Do NOT re-run this cell after restarting; the install already persisted to disk.)\n"
      + "=" * 72)


# ===== CELL 1 : imports, config, load model + tokenizer ======================
import os, json, math, gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache

torch.set_grad_enabled(False)
# EXPORT RUNS ON CPU (like the 26B MoE notebook: "the GPU sits idle during conversion").
# ov.convert_model is a CPU/RAM operation; putting the ~26B model on the GPU adds VRAM pressure
# and makes convert_model trace a CUDA graph, which crashed the kernel. On CPU the scratch swap
# can actually absorb the ~175GB convert peak. Trace is a single 16-token forward -> slow but fine.
DEVICE = "cpu"

# Target the EGA (Expert-Granular Abliteration) HERETIC build, NOT DuoNeural's.
# DuoNeural's abliteration skipped the batched MoE expert tensor and still refuses
# noticeably (see its own discussion #1). edwixx's EGA reaches into the
# experts.down_proj [128, 2816, 704] batched parameter: 13/100 refusals from 100/100,
# KL 0.49. Architecturally identical (diffusion_gemma, 2816 / 30L / 16-8kv /
# 128e-top8 / canvas 256), so this script is unchanged apart from this repo string.
# A/B RUN 2026-07-10: exporting the ORIGINAL google weights to test whether the 'own'/junk
# token attractor comes from the HERETIC abliteration (EGA on experts.down_proj, KL 0.49).
# NOTE: the google repo is GATED -- your HF token must have accepted the Gemma license.
# To re-export the HERETIC instead, flip the two commented lines.
SRC_REPO = "google/diffusiongemma-26B-A4B-it"
DST_REPO = "Wondernutts/diffusiongemma-26B-A4B-it-openvino-int4"
# SRC_REPO = "edwixx/diffusiongemma-26B-A4B-it-HERETIC-Uncensored"
# DST_REPO = "Wondernutts/diffusiongemma-26B-A4B-it-HERETIC-openvino-int4"
# Pin ALL heavy artifacts to the big scratch disk when present (like the 26B MoE notebook).
# /content is small: the fp32 IR-D intermediate alone is tens of GB and will fill it. If a
# run recycles, scratch is wiped too, but the HF/Drive uploads in CELL 8 are the durable copy.
SCRATCH  = "/mnt/local-scratch" if os.path.isdir("/mnt/local-scratch") else "/content"
OUT_DIR  = os.path.join(SCRATCH, "dg_ov")
os.makedirs(OUT_DIR, exist_ok=True)
# Route the HF download cache to scratch too, so re-pulling the ~50GB checkpoint can't fill
# the boot disk. (Harmless if the model is already cached elsewhere this session.)
os.environ.setdefault("HF_HOME", os.path.join(SCRATCH, "hf"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(SCRATCH, "hf", "hub"))
print("scratch base:", SCRATCH, "| OUT_DIR:", OUT_DIR)

# HF auth. Token lifted from your COLAB_31B.txt. This is a SECRET: keep this
# notebook private and do not share it with the token still inline.
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
from huggingface_hub import login as _hf_login
_hf_login(token=HF_TOKEN)  # authenticate early

# Persist the finished artifacts to Google Drive so they survive the Colab
# session (the /content workspace is wiped when the runtime recycles). The
# mount is interactive in Colab (a consent popup); it no-ops off Colab.
DRIVE_DIR = "/content/drive/MyDrive/DiffusionGemma_OV"
try:
    from google.colab import drive as _gdrive
    _gdrive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
    _SAVE_TO_DRIVE = True
    print("Google Drive mounted, artifacts will be copied to", DRIVE_DIR)
except Exception as _e:
    _SAVE_TO_DRIVE = False
    print("Google Drive not mounted (not on Colab?):", _e)

# diffusion_gemma ships in the transformers CODEBASE (models/diffusion_gemma/) but is NOT
# registered in the AutoConfig mapping (confirmed on main 2026-07), so
# AutoConfig.from_pretrained raises KeyError 'diffusion_gemma'. Import the concrete classes
# by full module path and register them, bypassing the auto loader entirely.
# If THIS import fails, the kernel is still running the OLD transformers: do
# Runtime > Restart session, then Run all (CELL 0 installed the main build to disk).
try:
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
except Exception as _e:
    raise SystemExit(
        "Cannot import transformers.models.diffusion_gemma (%r). "
        "The kernel is on an OLD transformers: Runtime > Restart session, then Run all." % _e)
_ModelCls = DiffusionGemmaForBlockDiffusion
try:
    AutoConfig.register("diffusion_gemma", DiffusionGemmaConfig)
except Exception:
    pass  # already registered this session

config = DiffusionGemmaConfig.from_pretrained(SRC_REPO)
try:
    tokenizer = AutoTokenizer.from_pretrained(SRC_REPO)
except Exception:
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(SRC_REPO)

# eager attention: guarantees the explicit-additive-mask path (eager_attention_forward
# adds the 4D mask and upcasts softmax to fp32, modeling L269/L272), which is what we
# want traced for the fp32 parity goal (sdpa/flash would fuse and may narrow to fp16).
model = _ModelCls.from_pretrained(
    SRC_REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
)  # NB: transformers 5.x renamed torch_dtype -> dtype (torch_dtype now warns/errors)
model.eval().to(DEVICE)

# ---- constants from the live config, asserted against verified ground truth ----
tc = getattr(config, "text_config", config)
def cfg(name, default): return getattr(tc, name, getattr(config, name, default))

HIDDEN          = int(cfg("hidden_size", 2816))
N_LAYERS        = int(cfg("num_hidden_layers", 30))
N_HEADS         = int(cfg("num_attention_heads", 16))
SLIDING_KV      = int(cfg("num_key_value_heads", 8))          # verified: 8 (NOT 4)
SLIDING_HD      = int(cfg("head_dim", 256))
GLOBAL_KV       = int(cfg("num_global_key_value_heads", 2))
GLOBAL_HD       = int(cfg("global_head_dim", 512))
VOCAB           = int(cfg("vocab_size", 262144))
CANVAS          = int(getattr(config, "canvas_length", getattr(tc, "canvas_length", 256)))
INTERMEDIATE    = int(cfg("intermediate_size", 2112))
MOE_INTER       = int(cfg("moe_intermediate_size", 704))
N_EXPERTS       = int(cfg("num_experts", 128))
TOP_K           = int(cfg("top_k_experts", 8))
SOFTCAP         = float(cfg("final_logit_softcapping", 30.0))
RMS_EPS         = float(cfg("rms_norm_eps", 1e-6))
SLIDING_WINDOW  = int(cfg("sliding_window", 1024))
EMBED_SCALE     = float(HIDDEN ** 0.5)                        # ~53.07, baked in-graph
BLOCK_LENGTH    = CANVAS                                      # one committed block == canvas

# sampler defaults (no generation_config.json ships; taken from code)
MAX_STEPS   = 48
ENTROPY_BOUND = 0.1
T_MIN, T_MAX  = 0.4, 0.8
STABILITY_THRESHOLD  = 1
CONFIDENCE_THRESHOLD = 0.005

# layer_types: full at (i+1)%6==0, last forced full (already satisfied at 29)
LAYER_TYPES = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
               for i in range(N_LAYERS)]
LAYER_TYPES[-1] = "full_attention"

def kv_hd(lt):   # (num_kv_heads, head_dim) for a layer type
    return (GLOBAL_KV, GLOBAL_HD) if lt == "full_attention" else (SLIDING_KV, SLIDING_HD)

# token ids for the sampler (from tokenizer, fallback to config)
def _tok_id(tok_attr, cfg_attr):
    v = getattr(tokenizer, tok_attr, None)
    if v is None:
        v = getattr(tc, cfg_attr, getattr(config, cfg_attr, None))
    return v
TOKEN_IDS = {
    "eos": _tok_id("eos_token_id", "eos_token_id"),
    "pad": _tok_id("pad_token_id", "pad_token_id"),
    "bos": _tok_id("bos_token_id", "bos_token_id"),
}

print("layer_types:", LAYER_TYPES)
print("per-layer kv (heads,dim):", [kv_hd(lt) for lt in LAYER_TYPES])
print("token_ids:", TOKEN_IDS)
assert N_LAYERS == 30 and VOCAB == 262144 and HIDDEN == 2816
assert SLIDING_KV == 8, f"sliding kv heads must be 8, got {SLIDING_KV}"


# ===== CELL 2 : monkeypatch MoE experts -> traceable CAPACITY-DISPATCH form ==================
# WHY: reference DiffusionGemmaTextExperts.forward (modeling L572-596) uses a data-dependent
# python loop + nonzero/where/index_add_ => cannot trace. The v1 export used a DENSE all-128-
# experts reformulation: numerically exact but 16x the active FLOPs -- measured ~1.2-1.5s per
# denoise step on Arc (the MoE bmms are ~95% of step time). v2 (this cell) dispatches each
# token to ONLY its top-8 experts via a GShard-style capacity table:
#
#   y_t = sum_k w_{t,k} * f_{e(t,k)}(x_t)          (reference; renorm folded into w, L551+L554)
#
#   flat assignments a = (t,k) -> expert e_a. Within each expert, assignments take numbered
#   slots (cumsum order). A [E, C, H] table gathers each slot's token embedding, the 3 expert
#   matmuls run batched at [E, C, *] instead of [E, T, *] (C << T), and each assignment gathers
#   back its expert-slot row weighted by w. EXACT same math per kept assignment; an assignment
#   is dropped ONLY if its expert exceeds C tokens -- C is chosen from MEASURED routing on a
#   real noise canvas with 1.5x slack, and the parity test below asserts ZERO drops at T=256.
#   Bonus: non-selected experts never see the token at all, so the v1 0*inf NaN hazard is gone
#   by construction (empty slots compute f_e(0)=0: no biases anywhere in the expert MLP).
#
# OV-frontend constraints honored: no einsum/chunk/index_put_/scatter_/one_hot. Only
# eq-broadcast, cumsum (int32: bf16 cumsum loses integer exactness past 256), matmul, bmm,
# index_select, clamp, reshape/expand -- all convert.

def traceable_experts_forward(self, hidden_states, arg_a, arg_b):
    # reference call is experts(x, top_k_index, top_k_weights); be order-robust by dtype
    if arg_a.dtype in (torch.long, torch.int64, torch.int32):
        top_k_index, top_k_weights = arg_a, arg_b
    else:
        top_k_weights, top_k_index = arg_a, arg_b
    orig_dtype = hidden_states.dtype
    x = hidden_states                                            # [T, H]
    E = int(getattr(self, "num_experts", self.gate_up_proj.shape[0]))
    C = int(self.capacity)                                       # static slots per expert
    K = int(top_k_index.shape[-1])
    flat_e = top_k_index.reshape(-1)                             # [T*K] assignment -> expert
    e_ids = torch.arange(E, device=x.device).view(1, E)
    oh = (flat_e.view(-1, 1) == e_ids).to(torch.int32)           # [T*K, E] assignment one-hot
    pos = torch.cumsum(oh, dim=0) - oh                           # prior same-expert assignments
    slot = (pos * oh).sum(-1)                                    # [T*K] slot within its expert
    keep = (slot < C).to(torch.int32)                            # overflow assignments dropped
    slot_c = torch.clamp(slot, max=C - 1)
    c_ids = torch.arange(C, device=x.device).view(1, C)
    soh = (slot_c.view(-1, 1) == c_ids).to(torch.int32)          # [T*K, C] slot one-hot
    disp = (oh.unsqueeze(-1) * soh.unsqueeze(1)) * keep.view(-1, 1, 1)   # [T*K, E, C]
    disp_flat = disp.reshape(-1, E * C)                          # [T*K, E*C]
    # row ids via cumsum(ones)-1, NOT arange(shape[0]): trace would bake T*K as a constant
    rows = (torch.cumsum(torch.ones_like(flat_e, dtype=torch.int32), 0) - 1).view(-1, 1)
    tok_table = (disp_flat * rows).sum(0)                        # [E*C] source assignment per slot
    used = disp_flat.sum(0)                                      # [E*C] slot occupied 0/1
    x_rep = x.unsqueeze(1).expand(-1, K, -1).reshape(-1, x.shape[-1])    # [T*K, H]
    xd = x_rep.index_select(0, tok_table.to(torch.long))         # [E*C, H] gather (no big matmul)
    xd = (xd * used.to(xd.dtype).view(-1, 1)).view(E, C, -1)     # zero the empty slots
    gu = torch.matmul(xd, self.gate_up_proj.transpose(1, 2))     # [E, C, 2I]
    half = gu.shape[-1] // 2
    inter = self.act_fn(gu[..., :half]) * gu[..., half:]         # gelu_pytorch_tanh on gate
    out = torch.matmul(inter, self.down_proj.transpose(1, 2))    # [E, C, H]
    es = flat_e.to(torch.int32) * C + slot_c.to(torch.int32)     # [T*K] expert-slot row per assignment
    y_k = out.reshape(E * C, -1).index_select(0, es.to(torch.long))      # [T*K, H] gather back
    w = top_k_weights.reshape(-1).to(y_k.dtype) * keep.to(y_k.dtype)     # dropped -> weight 0
    y = (y_k * w.view(-1, 1)).view(-1, K, y_k.shape[-1]).sum(1)  # [T, K, H] -> [T, H]
    return y.to(orig_dtype)

# ---- dtype-agnostic reference + dense forms, parameterised by explicit weights ----
# (used ONLY by the equivalence unit test; run entirely in fp32, see D3)
def _experts_ref_loop(gate_up, down, act_fn, x, idx, w):
    y = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            e = int(idx[t, k]); xt = x[t]
            gu = xt @ gate_up[e].T
            g, u = gu.chunk(2, dim=-1)
            f = (act_fn(g) * u) @ down[e].T
            y[t] += w[t, k].to(f.dtype) * f
    return y

def _experts_dense(gate_up, down, act_fn, x, idx, w):
    E = gate_up.shape[0]; T = x.shape[0]
    gu = torch.einsum("th,eoh->eto", x, gate_up)
    gate, up = gu.chunk(2, dim=-1)
    inter = act_fn(gate) * up
    out = torch.einsum("eti,ehi->eth", inter, down)
    full_w = torch.zeros(T, E, dtype=x.dtype, device=x.device)
    full_w.scatter_(1, idx, w.to(x.dtype))
    return torch.einsum("eth,te->th", out, full_w)

# ---- D4: discover the experts class(es) ROBUSTLY (do not hardcode decoder layer 0) ----
# Every encoder AND decoder layer runs a dense MLP AND an MoE branch, so `.experts` exists
# per layer (verified modular L471-472 encoder, L546-547 decoder), but we do not assume it:
# collect every module exposing gate_up_proj + down_proj + act_fn, assert at least one lives
# under the encoder and one under the decoder, and patch EVERY distinct class found.
expert_modules = {}
for name, mod in model.named_modules():
    if (hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj") and hasattr(mod, "act_fn")
            and getattr(mod, "gate_up_proj", None) is not None
            and getattr(mod.gate_up_proj, "ndim", 0) == 3):
        expert_modules[name] = mod
assert len(expert_modules) > 0, "no MoE experts modules found (gate_up_proj+down_proj+act_fn)"
enc_expert_names = [n for n in expert_modules if "encoder" in n]
dec_expert_names = [n for n in expert_modules if "decoder" in n]
assert len(enc_expert_names) > 0, "no experts found under the encoder stack"
assert len(dec_expert_names) > 0, "no experts found under the decoder stack"
expert_classes = {type(m) for m in expert_modules.values()}
print(f"[experts] {len(expert_modules)} modules, classes={[c.__name__ for c in expert_classes]}, "
      f"enc={len(enc_expert_names)} dec={len(dec_expert_names)}")

# ---- capture REAL experts input (hidden states + router idx/weights) for the test ----
# Try a real forward; fall back to a norm-shaped synthetic if it errors (never halt).
real_x = real_idx = real_w = None
try:
    _cap = {}
    _tgt = expert_modules[dec_expert_names[0]]
    def _cap_hook(m, i, o):
        _cap["x"], _cap["idx"], _cap["w"] = i[0].detach(), i[1].detach(), i[2].detach()
    _h = _tgt.register_forward_hook(_cap_hook)
    with torch.no_grad():
        _p = 8
        _zeros = lambda kv: torch.zeros(1, 1, CANVAS, kv, dtype=torch.float32, device=DEVICE)
        _md = {"full_attention": _zeros(_p + CANVAS), "sliding_attention": _zeros(_p + CANVAS)}
        model(
            input_ids=torch.randint(0, VOCAB, (1, _p), device=DEVICE),
            decoder_input_ids=torch.randint(0, VOCAB, (1, CANVAS), device=DEVICE),
            decoder_attention_mask=_md,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
    _h.remove()
    real_x, real_idx, real_w = _cap["x"].float(), _cap["idx"], _cap["w"].float()
    print("[experts] captured REAL experts input:", tuple(real_x.shape))
except Exception as e:
    try: _h.remove()
    except Exception: pass
    print("[experts] real-capture failed, using norm-shaped synthetic:", repr(e))
    _raw = torch.randn(64, HIDDEN, device=DEVICE)
    _ln = model.model.decoder.layers[0].pre_feedforward_layernorm_2
    real_x = _ln(_raw.to(next(_ln.parameters()).dtype)).float()
    real_idx = torch.stack([torch.randperm(N_EXPERTS, device=DEVICE)[:TOP_K] for _ in range(64)])
    _w = torch.rand(64, TOP_K, device=DEVICE)
    real_w = _w / _w.sum(-1, keepdim=True)

# ---- EXPERT CAPACITY from MEASURED routing (drives the dispatch table size) ----
# real_idx above is router output on an actual uniform-noise canvas (the worst-case skew the
# model sees at denoise step 0). C = 1.5x the max observed per-expert load, 16-aligned,
# floor 48 (covers the synthetic-fallback case where measured skew is artificially flat).
_loads = torch.bincount(real_idx.reshape(-1), minlength=N_EXPERTS)
_maxload = int(_loads.max())
EXPERT_CAPACITY = max(48, min(int(-(-(_maxload * 1.5) // 16) * 16), 256))
print(f"[experts] measured max expert load {_maxload} (mean {float(_loads.float().mean()):.1f}) "
      f"over {real_idx.shape[0]} tokens -> capacity C={EXPERT_CAPACITY} "
      f"(compute cut vs dense: {256 / EXPERT_CAPACITY:.1f}x at T=256)")

# ---- D3: MoE equivalence unit test, run in ONE dtype (fp32 copies of the weights) ----
# torch matmul/einsum do NOT promote fp32<->bf16, so we upcast the bf16 expert weights to
# fp32 and run x/idx/w in fp32 so the ~1e-3 assertion stays meaningful (and does not crash).
with torch.no_grad():
    _probe = expert_modules[dec_expert_names[0]]
    gu32 = _probe.gate_up_proj.detach().float()
    dn32 = _probe.down_proj.detach().float()
    act  = _probe.act_fn
    n = min(32, real_x.shape[0])                      # cap the O(T*K) python loop
    xf, idxf, wf = real_x[:n].float(), real_idx[:n], real_w[:n].float()
    ref = _experts_ref_loop(gu32, dn32, act, xf, idxf, wf)
    got = _experts_dense(gu32, dn32, act, xf, idxf, wf)
    err = (ref - got).abs().max().item()
    print(f"[experts] dense-vs-loop max abs err on REAL hidden states (fp32): {err:.3e}")
    assert err < 1e-3, "dense experts reformulation diverged from the top-k loop"

# ---- REPLACE each experts MODULE with a plain wrapper (NOT a cls.forward patch!) ----
# DiffusionGemmaTextExperts is decorated @use_experts_implementation (modeling L559). That
# decorator dispatches the layer's self.experts(...) call (L659/L734) to a BATCHED kernel using
# einsum/chunk/index_add, and it routes AROUND any `cls.forward = ...` monkeypatch -- so patching
# forward does nothing and the trace still shows aten::einsum/chunk/index_put_. Swapping the whole
# module for a plain nn.Module the decorator never wrapped makes self.experts(...) land on OUR
# traceable forward. We reuse the ORIGINAL Parameters (no weight copy).
class TraceableExperts(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.num_experts  = int(getattr(orig, "num_experts", orig.gate_up_proj.shape[0]))
        self.gate_up_proj = orig.gate_up_proj      # same Parameter objects (shared, not copied)
        self.down_proj    = orig.down_proj
        self.act_fn       = orig.act_fn
        self.capacity     = EXPERT_CAPACITY        # static dispatch slots per expert
    forward = traceable_experts_forward            # capacity dispatch (OV-convertible ops only)

_n_repl = 0
for name in list(expert_modules):
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, TraceableExperts(expert_modules[name]))
    _n_repl += 1
print(f"[experts] replaced {_n_repl} experts modules with TraceableExperts (dispatcher bypassed)")

# ---- verify the REPLACED module (capacity dispatch) against the dense reference at FULL T ----
# Full captured T (256 canvas tokens, real noise-canvas routing) so capacity overflow -- the ONLY
# way this path can diverge from exact -- is exercised at the real shape, not a toy slice.
with torch.no_grad():
    _m = model.get_submodule(dec_expert_names[0])
    assert isinstance(_m, TraceableExperts), "experts module was not replaced"
    # zero-drop assertion: every expert's measured load must fit its capacity
    _loads_t = torch.bincount(real_idx.reshape(-1), minlength=N_EXPERTS)
    _drops = int((_loads_t - EXPERT_CAPACITY).clamp(min=0).sum())
    assert _drops == 0, f"capacity {EXPERT_CAPACITY} would drop {_drops} assignments -- raise slack"
    xb = real_x.to(_m.gate_up_proj.dtype)
    got_call  = _m(xb, real_idx, real_w.to(_m.gate_up_proj.dtype))    # __call__ -> capacity forward
    # same-dtype comparison (bf16 vs bf16): fp32 correctness of the dense form is proven by the
    # dense-vs-loop test above; here we need "same math, same precision" -> relative error.
    got_dense = _experts_dense(_m.gate_up_proj, _m.down_proj, _m.act_fn,
                               xb, real_idx, real_w.to(xb.dtype))
    _rel = (got_call.float() - got_dense.float()).abs().max() / (got_dense.float().abs().max() + 1e-6)
    assert _rel < 5e-2, f"capacity experts path mismatch (rel err {_rel:.4f})"
    print(f"[experts] capacity-dispatch module verified vs dense at T={real_x.shape[0]} "
          f"(rel err {_rel:.4f}, drops {_drops})")

# ---- flatten 0-dim (scalar) buffers/params to shape [1] so the OV frontend can const-fold them ----
# OpenVINO's torch_tensor_to_ov_const chokes on ndim==0 tensors ("too many indices for an array:
# 1 (ndim = 0)"). embed_scale = torch.tensor(sqrt(hidden)) (modeling L756) and the clipped-linear
# +-inf clamp buffers (L187-190) are 0-dim. All are used only in broadcast elementwise ops, so
# reshaping a scalar to [1] is numerically identical (it broadcasts the same way).
_n_flat = 0
for _mod in model.modules():
    for _bn, _b in list(_mod._buffers.items()):
        if _b is not None and _b.ndim == 0:
            _mod._buffers[_bn] = _b.reshape(1); _n_flat += 1
    for _pn, _p in list(_mod._parameters.items()):
        if _p is not None and _p.ndim == 0:
            _mod._parameters[_pn] = nn.Parameter(_p.reshape(1), requires_grad=_p.requires_grad); _n_flat += 1
print(f"[trace-prep] flattened {_n_flat} scalar (0-dim) buffer(s)/param(s) to shape [1]")


# ===== CELL 3 : shared helpers (cache build, masks, dynamic shapes) ==========
import openvino as ov
from openvino import Dimension, PartialShape

NEG = torch.finfo(torch.float32).min

# transformers 5.x DynamicLayer.lazy_initialization sets keys/values to a RANK-1 empty tensor
# (torch.tensor([])). Eager cat() silently ignores a 1-D empty, but the OV tracer records a real
# aten::cat([empty_rank1, new_rank4], dim=-2) and rejects it ("Axis -2 out of tensor rank range
# [-1,0]"). Patch lazy init to a rank-4 zero-LENGTH tensor so every concat axis stays valid.
from transformers.cache_utils import DynamicLayer
def _rank4_lazy_initialization(self, key_states, value_states):
    self.dtype, self.device = key_states.dtype, key_states.device
    kb, kh, _, kd = key_states.shape
    vb, vh, _, vd = value_states.shape
    self.keys   = key_states.new_zeros((kb, kh, 0, kd))
    self.values = value_states.new_zeros((vb, vh, 0, vd))
    self.is_initialized = True
DynamicLayer.lazy_initialization = _rank4_lazy_initialization

def build_cache(kv_flat):
    """kv_flat: [k0,v0,k1,v1,...] -> a DynamicCache preseeded per layer by directly setting each
       layer's rank-4 keys/values (NOT via .update(), which would cat onto the empty-init tensor
       and put an invalid-axis concat into the trace). After this, the model's own update()/
       append_to_cache() only ever concatenates rank-4 prefix + rank-4 new -> traces cleanly."""
    layers = []
    for i in range(N_LAYERS):
        k, v = kv_flat[2 * i], kv_flat[2 * i + 1]
        L = DynamicLayer()
        L.dtype, L.device = k.dtype, k.device
        L.keys, L.values = k, v
        L.is_initialized = True
        layers.append(L)
    c = DynamicCache()
    c.layers = layers
    return c

def read_cache(cache):
    """Per-layer [(k,v),...] readout, robust across transformers cache APIs.
       transformers 5.x removed to_legacy_cache(); the current layout is
       cache.layers[i].keys/.values. Fall back to the older parallel-list and
       legacy-tuple APIs so this works on any pinned build (fallback note #5)."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [(layers[i].keys, layers[i].values) for i in range(len(layers))]
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return [(cache.key_cache[i], cache.value_cache[i])
                for i in range(len(cache.key_cache))]
    if hasattr(cache, "to_legacy_cache"):
        return list(cache.to_legacy_cache())
    raise AttributeError("cannot read KV out of cache type %r" % type(cache))

def make_kv(seq, batch=1, dtype=torch.bfloat16):
    """UNIFORM per-layer-type KV, all layers at length `seq` (the D1 runtime contract)."""
    kv = []
    for lt in LAYER_TYPES:
        h, d = kv_hd(lt)
        kv.append(torch.zeros(batch, h, seq, d, dtype=dtype, device=DEVICE))  # key
        kv.append(torch.zeros(batch, h, seq, d, dtype=dtype, device=DEVICE))  # value
    return kv

def make_kv_hetero(p_slid, p_full, batch=1, dtype=torch.bfloat16):
    """Per-layer-TYPE divergent KV lengths. Used ONLY for the IR-E trace example so the
       per-layer new-token slice (D8) is captured as a dynamic axis, not baked from layer 0."""
    kv = []
    for lt in LAYER_TYPES:
        h, d = kv_hd(lt)
        p = p_full if lt == "full_attention" else p_slid
        kv.append(torch.zeros(batch, h, p, d, dtype=dtype, device=DEVICE))
        kv.append(torch.zeros(batch, h, p, d, dtype=dtype, device=DEVICE))
    return kv

# ---- IR-E (encoder, causal) masks: additive [1,1,Lq,Lk], 0 where attend else NEG ----
def additive_causal(q_pos, k_pos, sliding=False, dtype=torch.float32):
    """Causal (k_pos <= q_pos); sliding also windows per-query to SLIDING_WINDOW.
       Over a uniform full-length cache this reproduces the encoder's capped-sliding behavior."""
    q = q_pos.view(-1, 1); k = k_pos.view(1, -1)
    allow = k <= q
    if sliding:
        allow = allow & ((q - k) < SLIDING_WINDOW)
    m = torch.where(allow, torch.zeros((), dtype=dtype), torch.full((), NEG, dtype=dtype))
    return m.view(1, 1, q_pos.numel(), k_pos.numel()).to(DEVICE)

# ---- IR-D (decoder, bidirectional) masks over [prefix + canvas], additive ----
# D1: full = all-zeros (every canvas query attends every prefix+canvas key). sliding = zeros
# EXCEPT NEG on prefix key columns older than sliding_window relative to the block start
# (== the keys the reference sliding cache would have evicted); canvas-to-canvas fully visible.
def dec_full_mask(prefix_len, canvas=CANVAS, batch=1, dtype=torch.float32):
    return torch.zeros(batch, 1, canvas, prefix_len + canvas, dtype=dtype, device=DEVICE)

def dec_sliding_mask(prefix_len, cache_len=None, canvas=CANVAS, window=SLIDING_WINDOW,
                     batch=1, dtype=torch.float32):
    if cache_len is None:
        cache_len = prefix_len          # block starts at absolute position cache_len
    m = torch.zeros(batch, 1, canvas, prefix_len + canvas, dtype=dtype, device=DEVICE)
    if prefix_len > 0:
        col = torch.zeros(prefix_len + canvas, dtype=torch.bool, device=DEVICE)
        j = torch.arange(prefix_len, device=DEVICE)          # prefix key j -> absolute pos j
        col[:prefix_len] = j < (cache_len - window)          # older than window -> evicted -> NEG
        m[:, :, :, col] = NEG
    return m

def set_dynamic(ovm, dyn_axes, names):
    for inp, axes, nm in zip(ovm.inputs, dyn_axes, names):
        ps = inp.get_partial_shape()
        shape = [Dimension(-1) if i in axes else ps[i] for i in range(len(ps))]
        inp.get_node().set_partial_shape(PartialShape(shape))
        inp.get_tensor().set_names({nm})
    ovm.validate_nodes_and_infer_types()

def name_outputs(ovm, names):
    for out, nm in zip(ovm.outputs, names):
        out.get_tensor().set_names({nm})

# ---- D1 post-build assertion: masks are all-attend for full and windowed for sliding ----
# (NO causal triangle: every canvas query row is identical, i.e. bidirectional; sliding
# masks ONLY old prefix columns). Run on a case with prefix > window so NEG actually appears.
_Pbig = SLIDING_WINDOW + 300
_fm = dec_full_mask(_Pbig); _sm = dec_sliding_mask(_Pbig)
assert (_fm == 0).all(), "full mask must be all-attend (all zeros)"
assert (_sm[..., -CANVAS:] == 0).all(), "sliding canvas-to-canvas must be fully visible"
assert (_sm[..., : _Pbig][..., : _Pbig - SLIDING_WINDOW] == NEG).all(), "old prefix cols must be NEG"
assert (_sm[..., : _Pbig][..., _Pbig - SLIDING_WINDOW :] == 0).all(), "recent prefix cols must attend"
assert (_sm[:, :, 0, :] == _sm[:, :, -1, :]).all(), "rows must be identical (bidirectional, no causal triangle)"
print("[masks] D1 structural check passed: full=all-attend, sliding=windowed, no causal triangle")


# ===== CELL 4 : UNIFIED single-backbone IR (encoder + decoder roles in ONE graph) ===========
# DiffusionGemma ties ALL transformer weights between encoder and decoder (_tied_weights_keys,
# modeling L1481-1491); the ONLY structural fork is the decoder's self-conditioning at the input
# (decoder L1286 vs encoder L940). So we export ONE graph that stores the shared ~26B weights ONCE
# and switches role via inputs -- instead of two IRs that each bake a full copy (2x VRAM waste).
#
#   inputs : current_ids, self_conditioning_logits, self_conditioning_mask, apply_sc,
#            position_ids, full_mask, sliding_mask, prefix_key_i/prefix_value_i (per layer)
#   outputs: new_key_i/new_value_i (per layer, the CURRENT tokens' K/V) + logits
#
#   ENCODER role: prompt/committed tokens, CAUSAL masks, apply_sc=0 -> use new K/V, ignore logits
#   DECODER role: canvas(256),          BIDIR  masks, apply_sc=1 + sc -> use logits, ignore new K/V
#
# A torch-level VERIFICATION GATE (below) proves both roles match the real encoder/decoder in
# seconds, BEFORE the expensive trace/quantize -- so a unification bug never costs a full export.

# ---- SWAP: converting a ~26B model to OV peaks around ~175GB RAM (> the 167GB box), which
#      OOM-crashes the kernel mid-convert (the 26B MoE run hit this exact wall). Add swap on the
#      big scratch disk so the peak can spill. Idempotent -- skips if swap is already on. ----
import subprocess as _sp, os as _os
_SWAP = "/mnt/local-scratch/swapfile" if _os.path.isdir("/mnt/local-scratch") else "/content/swapfile"
if "swapfile" not in _sp.run("swapon --show", shell=True, capture_output=True, text=True).stdout:
    _sp.run(f"swapoff {_SWAP} 2>/dev/null; rm -f {_SWAP}", shell=True)
    _sp.run(f"fallocate -l 96G {_SWAP} && chmod 600 {_SWAP} && mkswap {_SWAP} && swapon {_SWAP}",
            shell=True, check=True)
    print("[swap] enabled 96G at", _SWAP)
_sp.run("free -g | head -2; swapon --show", shell=True)

# ---- PREFLIGHT: both fixes must be LIVE in this kernel before the expensive convert. If either
#      assert fires, re-run the named cell (cells must be EXECUTED this session, not just pasted). ----
from transformers.cache_utils import DynamicLayer as _DL
_e0 = model.get_submodule([n for n, _ in model.named_modules() if n.endswith("experts")][0])
assert type(_e0).__name__ == "TraceableExperts", "EXPERTS NOT REPLACED -> re-run CELL 2"
assert "rank4" in getattr(_DL.lazy_initialization, "__name__", ""), "CACHE NOT PATCHED -> re-run CELL 3"
assert not any(b is not None and b.ndim == 0 for m in model.modules() for b in m._buffers.values()), \
    "SCALAR (0-dim) BUFFERS PRESENT -> re-run CELL 2 (the flatten-scalars step)"
print("[preflight] experts=TraceableExperts, cache lazy-init=rank4, no 0-dim buffers; safe to convert")

from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaSelfConditioning, DiffusionGemmaDecoderTextAttention,
)

# role state shared with the two monkeypatches during a traced forward
_ROLE = {"apply_sc": None, "kv": None}

# monkeypatch 1: gate self-conditioning by a traced apply_sc flag. The decoder path always does
# inputs_embeds = self_conditioning(embed, soft); apply_sc=0 returns the PLAIN embeddings (== the
# encoder, modeling L940), apply_sc=1 returns the SC output (L1286). torch.where is a select (not
# control flow) -> one OpenVINO Select, both roles in ONE graph.
_orig_sc_forward = DiffusionGemmaSelfConditioning.forward
def _sc_gated(self, inputs_embeds, self_conditioning_signal):
    sc_out = _orig_sc_forward(self, inputs_embeds, self_conditioning_signal)
    flag = _ROLE["apply_sc"]                                                # [B,1,1] float {0.,1.}, or None
    if flag is None:                        # NOT inside a BackboneExport forward (e.g. the verify
        return sc_out                       # gate's REAL model call) -> original decoder behavior
    return torch.where(flag > 0.5, sc_out, inputs_embeds)
DiffusionGemmaSelfConditioning.forward = _sc_gated

# monkeypatch 2: stash each layer's NEW (current-token) K/V so the wrapper can output it.
# append_to_cache receives exactly the current tokens' K/V (post k_norm + rope), which is what the
# ENCODER role needs to write to the cache. No-op stash when not inside a BackboneExport forward.
_orig_append = DiffusionGemmaDecoderTextAttention.append_to_cache
def _append_stash(self, past_key_values, key_states, value_states):
    if _ROLE["kv"] is not None:
        _ROLE["kv"][self.layer_idx] = (key_states, value_states)
    return _orig_append(self, past_key_values, key_states, value_states)
DiffusionGemmaDecoderTextAttention.append_to_cache = _append_stash


class BackboneExport(nn.Module):
    """One shared-backbone graph. Runs the decoder path for both roles; apply_sc picks the role."""
    def __init__(self, full_model):
        super().__init__()
        self.full = full_model
    def forward(self, current_ids, self_conditioning_logits, self_conditioning_mask, apply_sc,
                position_ids, full_mask, sliding_mask, *prefix_kv):
        _ROLE["apply_sc"] = apply_sc.reshape(-1, 1, 1)             # broadcast over [B, L, H]
        _ROLE["kv"] = [None] * N_LAYERS
        cache = build_cache(list(prefix_kv))
        out = self.full(
            input_ids=None,
            decoder_input_ids=current_ids,
            decoder_attention_mask={"full_attention": full_mask, "sliding_attention": sliding_mask},
            decoder_position_ids=position_ids,
            past_key_values=cache,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
        )
        new = []
        for i in range(N_LAYERS):
            k, v = _ROLE["kv"][i]
            new.append(k.contiguous()); new.append(v.contiguous())     # current tokens' K/V
        return (*new, out.logits)                                       # per-layer k,v, then logits

bb = BackboneExport(model).eval()

# ---- VERIFICATION GATE (torch-level, seconds): unified must match the REAL encoder & decoder ----
with torch.no_grad():
    PRE, LC = 6, CANVAS
    _prefix = make_kv(PRE)
    _pos_c  = torch.arange(PRE, PRE + LC, device=DEVICE).view(1, -1)
    _fullb  = dec_full_mask(PRE); _slidb = dec_sliding_mask(PRE)
    _canvas = torch.randint(0, VOCAB, (1, LC), device=DEVICE)
    _sc     = torch.randn(1, LC, VOCAB, dtype=torch.bfloat16, device=DEVICE)
    _sc0    = torch.zeros(1, dtype=torch.bfloat16, device=DEVICE)
    _one    = torch.ones(1, dtype=torch.float32, device=DEVICE)
    _zero   = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    # (A) DECODER role: unified logits must equal the real ForBlockDiffusion decoder logits.
    ref_logits = model(
        input_ids=None, decoder_input_ids=_canvas,
        decoder_attention_mask={"full_attention": _fullb, "sliding_attention": _slidb},
        decoder_position_ids=_pos_c, past_key_values=build_cache(list(_prefix)),
        self_conditioning_logits=_sc, self_conditioning_mask=_sc0,
    ).logits
    uni_logits = bb(_canvas, _sc, _sc0, _one, _pos_c, _fullb, _slidb, *_prefix)[-1]
    d_dec = (ref_logits.float() - uni_logits.float()).abs().max().item()
    print(f"[verify] decoder-role logits vs real decoder: max|delta|={d_dec:.3e}")
    assert d_dec < 1e-2, "unified decoder role does NOT match the real decoder"

    # (B) ENCODER role: unified new-K/V must equal the real encoder's new-K/V for the same tokens.
    ENC = 8
    _pe   = torch.arange(0, ENC, device=DEVICE).view(1, -1)
    _ids  = torch.randint(0, VOCAB, (1, ENC), device=DEVICE)
    _fce  = additive_causal(_pe.flatten(), torch.arange(0, ENC, device=DEVICE), sliding=False)
    _sce  = additive_causal(_pe.flatten(), torch.arange(0, ENC, device=DEVICE), sliding=True)
    _emptypre = make_kv(0)
    _enc_cache = build_cache(list(_emptypre))
    model.model.encoder(input_ids=_ids, attention_mask={"full_attention": _fce, "sliding_attention": _sce},
                        position_ids=_pe, past_key_values=_enc_cache, use_cache=True)
    ref_kv = read_cache(_enc_cache)
    _scE = torch.zeros(1, ENC, VOCAB, dtype=torch.bfloat16, device=DEVICE)
    uni_e = bb(_ids, _scE, _sc0, _zero, _pe, _fce, _sce, *_emptypre)
    d_kv = 0.0
    for i in range(N_LAYERS):
        d_kv = max(d_kv, (ref_kv[i][0].float() - uni_e[2 * i].float()).abs().max().item())
        d_kv = max(d_kv, (ref_kv[i][1].float() - uni_e[2 * i + 1].float()).abs().max().item())
    print(f"[verify] encoder-role new-K/V vs real encoder: max|delta|={d_kv:.3e}")
    assert d_kv < 1e-2, "unified encoder role does NOT match the real encoder"
    print("[verify] UNIFICATION CORRECT -- both roles match the reference. Safe to trace.")

# ---- trace the ONE graph (decoder-role example shapes; both roles share it via dynamic L) ----
_P = 24
_ex = (
    torch.randint(0, VOCAB, (1, CANVAS), device=DEVICE),                    # current_ids
    torch.randn(1, CANVAS, VOCAB, dtype=torch.bfloat16, device=DEVICE),     # self_conditioning_logits
    torch.zeros(1, dtype=torch.bfloat16, device=DEVICE),                    # self_conditioning_mask
    torch.ones(1, dtype=torch.float32, device=DEVICE),                      # apply_sc (1 = decoder)
    torch.arange(_P, _P + CANVAS, device=DEVICE).view(1, -1),               # position_ids
    dec_full_mask(_P), dec_sliding_mask(_P),                                # masks (bidirectional ex.)
    *make_kv(_P),                                                           # prefix k/v
)
print("Converting unified backbone IR ...")
ov_bb = ov.convert_model(bb, example_input=_ex)

bb_in = ["current_ids", "self_conditioning_logits", "self_conditioning_mask", "apply_sc",
         "position_ids", "full_mask", "sliding_mask"]
for i in range(N_LAYERS):
    bb_in += [f"prefix_key_{i}", f"prefix_value_{i}"]
# dyn axes: ids[B,L]->0,1 ; sc_logits[B,L,V]->0,1 ; sc_mask[B]->0 ; apply_sc[B]->0 ;
#           position_ids[B,L]->0,1 ; masks[B,1,L,S]->0,2,3 ; prefix kv[B,h,P,d]->0,2
bb_dyn = [[0, 1], [0, 1], [0], [0], [0, 1], [0, 2, 3], [0, 2]] + [[0, 2]] * (2 * N_LAYERS)
set_dynamic(ov_bb, bb_dyn, bb_in)
bb_out = []
for i in range(N_LAYERS):
    bb_out += [f"new_key_{i}", f"new_value_{i}"]
bb_out += ["logits"]
name_outputs(ov_bb, bb_out)
# fp32 tail preserved (lm_head + softcap + SC softmax); CELL 6 int4's the bulk with the router /
# per_expert_scale / self_conditioning / lm_head / embed ignored_scope, all in this ONE graph.
ov.save_model(ov_bb, os.path.join(OUT_DIR, "ir_bb_fp32.xml"), compress_to_fp16=False)
del ov_bb; gc.collect()
print("unified backbone IR converted -> ir_bb_fp32.xml (ONE graph, shared weights stored once)")

# free the torch model before quantization (NNCF works on the OV graph)
del bb, model
gc.collect(); torch.cuda.empty_cache()


# ===== CELL 6 : int4 quantization (MoE-aware ignored_scope) ==================
# INT4_SYM group 64, INT8_SYM backup for the rest. Router matmuls + per-expert/router scales
# + self-conditioning + tied embed/lm_head are kept OUT of int4 (router precision loss garbles
# top-k selection; keeping embed/lm_head/softcap/SC uncompressed preserves the fp32 tail from D2).
# Patterns match PyTorch-frontend friendly_names; widen on the box if a router/scale MatMul slips.
#
# AWQ + scale_estimation give the best quality but need an nncf.Dataset of real activations.
# We ship a data-free INT4 pass by default; enable AWQ by passing dataset=<nncf.Dataset(...)>.
import nncf

import re

# Weights that must stay OUT of int4. The two IRs have DIFFERENT node sets: IR-E is the ENCODER
# (no lm_head, no self_conditioning; layer_scalar gets const-folded away), IR-D is the DECODER
# (has them). A fixed IgnoredScope trips NNCF's strict "every pattern must match a node" check on
# IR-E. So per graph we keep only the patterns that actually match a node, and assert the router
# pattern matched -- quantizing the MoE router is the one thing that breaks the model (your 26B
# MoE playbook), so we refuse to proceed rather than silently int4 it.
MOE_IGNORE_PATTERNS = [
    r".*router.*",              # router.proj MatMul + scale  (CRITICAL: int4 here garbles top-k)
    r".*per_expert_scale.*",    # per-expert combine scales
    r".*self_conditioning.*",   # SC pre/post norm + gate/up/down MLP (decoder only; keep fp32)
    r".*layer_scalar.*",        # residual-stream scalar (often const-folded -> may be absent)
    r".*embed_tokens.*",        # tied embedding (SC matmul side)
    r".*lm_head.*",             # tied lm_head (decoder only; keep the fp32 tail)
]

def _ignored_for(m):
    names = [op.get_friendly_name() for op in m.get_ops()]
    kept = [p for p in MOE_IGNORE_PATTERNS if any(re.search(p, nm) for nm in names)]
    assert any("router" in p for p in kept), (
        "router ignore pattern matched NO node -> refusing to quantize (would garble the MoE). "
        "Print [op.get_friendly_name() for op in m.get_ops()] and widen the router pattern.")
    print("  [ignore] patterns kept for this graph:", kept)
    return nncf.IgnoredScope(patterns=kept)

def compress_int4(xml_in, xml_out, dataset=None, keep_fp32=False):
    m = ov.Core().read_model(xml_in)
    m = nncf.compress_weights(
        m,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=64,
        ratio=1.0,
        backup_mode=nncf.BackupMode.INT8_SYM,
        ignored_scope=_ignored_for(m),
        awq=(dataset is not None),
        scale_estimation=(dataset is not None),
        dataset=dataset,
    )
    # IR-D: compress_to_fp16=False so the ignored (lm_head / SC / softcap) constants and every
    # activation op stay fp32. IR-E: fp16 is fine (KV cache builder, not the tie-sensitive tail).
    ov.save_model(m, xml_out, compress_to_fp16=(not keep_fp32))
    del m; gc.collect()
    print("wrote", xml_out, "(fp32 tail)" if keep_fp32 else "(fp16)")

# ONE unified IR. keep_fp32=True: the tie-sensitive tail (lm_head + softcap + SC softmax) stays
# fp32; the bulk (attention + dense MLP + MoE experts) goes int4; router / per_expert_scale /
# self_conditioning / embed_tokens / lm_head are held out by _ignored_for (all in this graph now).
compress_int4(os.path.join(OUT_DIR, "ir_bb_fp32.xml"), os.path.join(OUT_DIR, "ir_bb_int4.xml"),
              keep_fp32=True)

# drop the uncompressed intermediate to shrink the upload
for f in ["ir_bb_fp32.xml", "ir_bb_fp32.bin"]:
    p = os.path.join(OUT_DIR, f)
    if os.path.exists(p): os.remove(p)


# ===== CELL 7 : config, tokenizer, manifest for the numpy sampler ===========
config.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

per_layer_kv = []
for i, lt in enumerate(LAYER_TYPES):
    h, d = kv_hd(lt)
    per_layer_kv.append({"layer": i, "type": lt, "kv_heads": h, "head_dim": d,
                         "shape": ["B", h, "S", d]})

manifest = {
    "model": SRC_REPO,
    "arch": "encoder-decoder tied-weight block-diffusion MoE; ONE unified stateless OV IR "
            "(shared backbone stored once; role selected by apply_sc + mask)",
    "irs": {
        "ir_bb_int4.xml": {
            "role": "unified shared-backbone graph. ENCODER pass (apply_sc=0, CAUSAL masks): use "
                    "new_key/new_value, ignore logits. DECODER/denoiser pass (apply_sc=1 + sc, BIDIR "
                    "masks): use logits, ignore new_key/new_value. Encoder runs 1x/block; decoder <=48x.",
            "inputs_in_order": bb_in,
            "input_notes": {
                "current_ids": ["B", "L (int64)", "ENCODER: prompt/committed tokens. DECODER: canvas (256)"],
                "self_conditioning_logits": ["B", "L", VOCAB,
                    "DECODER: previous step processed logits (temp-scaled + softcapped) cast to bf16 "
                    "(step 1: any dummy, paired with self_conditioning_mask=0). ENCODER: any (B,L,V) dummy "
                    "(gated off by apply_sc=0)."],
                "self_conditioning_mask": ["B", "DECODER step-1: 0.0 (zeros path). DECODER live: 1.0. ENCODER: ignored."],
                "apply_sc": ["B", "ROLE FLAG. 0.0 => ENCODER (plain embeddings, self-conditioning skipped). "
                                  "1.0 => DECODER (self-conditioning applied). Also tells you which output to read."],
                "position_ids": ["B", "L", "ENCODER: arange over the new tokens. DECODER: arange(cache_len, cache_len+256)"],
                "full_mask": ["B", 1, "L", "cache_len+L; additive fp32. ENCODER: causal. DECODER: all-zeros (bidirectional)"],
                "sliding_mask": ["B", 1, "L", "cache_len+L; additive fp32. ENCODER: causal AND (q-k)<1024. "
                    "DECODER: zeros except -inf on prefix cols j < (cache_len-1024)"],
                "prefix_key_i/prefix_value_i": "per-layer accumulated cache (prompt+committed blocks); prefix axis dynamic. "
                    "Feed a NON-EMPTY prefix on GPU (Arc NEO aborts on 0-length SVM buffers -> use a 1-token masked dummy on prefill).",
            },
            "outputs_in_order": bb_out,
            "output_notes": "new_key_i/new_value_i = the CURRENT tokens' K/V (ENCODER: append to your running per-layer "
                            "cache; DECODER: ignore). logits[B,L,V] fp32 ALREADY softcapped tanh(l/30)*30 (DECODER: these "
                            "are RAW temp=1 logits, divide by the step temperature FIRST, see numerics_D6; ENCODER: ignore).",
            "host_interface": "the bf16 KV / sc ports must be re-typed to fp32 at the host via PrePostProcessor on the box "
                              "(Arc GPU aborts on bf16 host SVM staging); bf16 convert then lives inside the graph.",
            "mask_dtype": "float32 (both masks). Do not feed fp16/bf16 masks.",
        },
    },
    "layer_types": LAYER_TYPES,
    "per_layer_kv": per_layer_kv,
    "per_layer_type_kv_shapes": {
        "sliding_attention": {"kv_heads": SLIDING_KV, "head_dim": SLIDING_HD, "shape": ["B", SLIDING_KV, "S", SLIDING_HD]},
        "full_attention":    {"kv_heads": GLOBAL_KV,  "head_dim": GLOBAL_HD,  "shape": ["B", GLOBAL_KV, "S", GLOBAL_HD]},
    },
    "token_ids": TOKEN_IDS,
    "dims": {
        "hidden_size": HIDDEN, "num_attention_heads": N_HEADS, "num_hidden_layers": N_LAYERS,
        "vocab_size": VOCAB, "canvas_length": CANVAS, "block_length": BLOCK_LENGTH,
        "sliding": {"head_dim": SLIDING_HD, "num_key_value_heads": SLIDING_KV,
                    "num_key_value_groups": N_HEADS // SLIDING_KV, "rope_theta": 1e4, "rope_type": "default"},
        "global": {"head_dim": GLOBAL_HD, "num_global_key_value_heads": GLOBAL_KV,
                   "num_key_value_groups": N_HEADS // GLOBAL_KV, "rope_theta": 1e6,
                   "rope_type": "proportional", "partial_rotary_factor": 0.25,
                   "rope_head_dim_key": "global_head_dim"},
        "sliding_window": SLIDING_WINDOW,
        "intermediate_size": INTERMEDIATE, "moe_intermediate_size": MOE_INTER,
        "num_experts": N_EXPERTS, "top_k_experts": TOP_K, "rms_norm_eps": RMS_EPS,
    },
    "constants_for_sampler": {
        "final_logit_softcapping": SOFTCAP,               # applied in-graph, fp32
        "embed_scale": EMBED_SCALE,                        # sqrt(hidden), baked in-graph
        "rms_norm_eps": RMS_EPS,
        "attention_scaling": 1.0,                          # NOT 1/sqrt(d); QK-norm replaces it
        "self_conditioning": {
            "sc_dtype": "bf16 (embed_tokens.weight dtype); reference feeds processed_logits.to(bf16), "
                        "then the SC softmax runs in fp32 INSIDE the decoder (verified L1279)",
            "pre_norm": "HAS a learnable scale; post_norm is with_scale=False (no weight)",
            "wiring": "post_norm(inputs_embeds + SC_MLP(pre_norm(soft_emb))); the whole embedding sum is "
                      "renormalized every step (post_norm has no scale)",
            "step1": "self_conditioning_mask=0.0 reproduces the zeros path exactly (SC_MLP(0)=0, pre_norm(0)=0)",
            "feedback": "next-step sc_logits = processed_logits.to(bf16) (temperature-scaled + softcapped)",
            "ov_precision_caveat": "IR-D keeps this softmax fp32; on the box confirm the SC softmax node "
                                   "did not get narrowed by any downstream re-quantization",
        },
        "numerics_D6": (
            "processed_logits = (IR-D raw logits) / temperature. token entropy (acceptance AND stopping), "
            "the multinomial draw, and the SC feedback all operate on the SAME processed_logits. Divide by "
            "temperature FIRST. Only the COMMITTED token = argmax(processed_logits) is temperature-invariant; "
            "entropy and sampling are NOT."),
        "position_threading": "decoder_position_ids = arange(cache_len, cache_len+256); next block's encoder "
                              "position_ids = the previous decoder_position_ids; cur_len += 256 per committed block",
        "committed_output": "argmax(processed_logits) per position (with finished-row freezing); NOT the "
                            "multinomial denoiser_canvas nor the renoised current_canvas",
        "noise": "uniform randint(0, vocab); NO mask token; rejected positions get FRESH randint every step",
        "sliding_window_semantics": (
            "The reference windows sliding layers by physically capping the sliding KV cache to 1024. We keep a "
            "UNIFORM full-length cache and instead feed IR-D a windowed sliding_mask (masked old keys contribute "
            "nothing == evicted). Build sliding_mask each step: -inf on prefix columns j < (cache_len - 1024)."),
    },
    "sampler_defaults": {
        "max_denoising_steps": MAX_STEPS, "entropy_bound": ENTROPY_BOUND, "t_min": T_MIN, "t_max": T_MAX,
        "stability_threshold": STABILITY_THRESHOLD, "confidence_threshold": CONFIDENCE_THRESHOLD,
        "canvas_length": CANVAS, "block_length": BLOCK_LENGTH,
        "temperature": "t_min + (t_max-t_min)*(cur_step/max_denoising_steps); cur_step counts DOWN 48..1 "
                       "(first step=t_max=0.8, last=~0.4083; never reaches t_min). Verified generation L315.",
        "entropy_units": "nats (natural log over full vocab). Categorical(logits=processed).entropy() = "
                         "-sum(p*clamp(log_softmax(processed), min=finfo.min)); use log-softmax, not naive p*log(p)",
        "acceptance": ("STICKY entropy-bound locking (validated on-device 2026-07-10; the reference "
                       "per-step accept-then-renoise re-randomizes previously accepted positions, never "
                       "converges, and commits noise-context junk): maintain a locked mask; per step, "
                       "sort UNLOCKED positions by entropy asc, lock those with cumsum(H)-H <= entropy_bound "
                       "(min 1/step) to their categorical sample PERMANENTLY; renoise only never-locked "
                       "positions; no locking during the first 4 warm steps (self-conditioning must form "
                       "before pure-noise confidence can commit); unlock a locked position if the model "
                       "later confidently disagrees (argmax != committed AND H < 0.05-0.1); at the horizon, "
                       "argmax-fill any never-locked stragglers"),
        "stopping": "stop when all positions locked (typ. 22-43 steps at entropy_bound 1.0) or at "
                    "max_denoising_steps; entropy_bound trades speed vs quality: 0.1 = careful, 1.0 = fast",
    },
    "quantization": {"mode": "INT4_SYM", "group_size": 64, "backup": "INT8_SYM",
                     "ir_precision": "one unified IR, fp32 tail (compress_to_fp16=False): lm_head + softcap + "
                                     "SC softmax stay fp32; bulk attention/MLP/MoE goes int4",
                     "ignored": ["router", "per_expert_scale", "self_conditioning",
                                 "layer_scalar", "embed_tokens", "lm_head"]},
    "fp32_tail_note": (
        "The unified IR is exported fp32 so the lm_head, the final softcap tanh(l/30)*30, and the 262144-wide "
        "self-conditioning softmax execute in fp32. The sampler's entropy / acceptance / argmax / SC feedback "
        "all depend on fp32 logits over a 262144 vocab; fp16 there flips argmax on near-ties and shifts the "
        "entropy-budget acceptance set."),
}
with open(os.path.join(OUT_DIR, "sampler_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("manifest written")


# ===== CELL 8 : upload to private HF repo ===================================
from huggingface_hub import HfApi, login
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
api.create_repo(DST_REPO, private=True, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=OUT_DIR, repo_id=DST_REPO, repo_type="model")
print("uploaded to", DST_REPO)

# Also copy the full artifact folder to Google Drive (survives the session).
if _SAVE_TO_DRIVE:
    import shutil
    _dst = os.path.join(DRIVE_DIR, os.path.basename(OUT_DIR.rstrip("/")))
    shutil.rmtree(_dst, ignore_errors=True)
    shutil.copytree(OUT_DIR, _dst)
    print("copied artifacts to Google Drive:", _dst)
else:
    print("Drive copy skipped (not mounted); artifacts are on HF only")


# =============================================================================
# FALLBACKS if a trace fails on the box:
#
# 1) MASK-DICT API: both the encoder (modeling L1135) and decoder (L1301) accept a
#    {full_attention, sliding_attention} dict of 4D masks and use it directly. If a pinned
#    build changed this, monkeypatch create_diffusion_decoder_attention_mask /
#    create_masks_for_generate to return the two 4D masks stashed on the wrapper.
#
# 2) input_ids=None routing: if ov.convert_model cannot trace ForBlockDiffusion with
#    input_ids=None, wrap DiffusionGemmaModel.forward with a trace-time MODE flag (encoder-only
#    vs decoder-only) so the flag is a python constant and no dynamic control flow enters the graph.
#
# 3) experts patch not taking through __call__ (e.g. a @use_experts_implementation dispatcher):
#    the CELL 2 assert `_m.forward.__func__ is traceable_experts_forward` catches this. If it
#    fires, also override the module's __call__/dispatch method to route to traceable_experts_forward.
#
# 4) router/scale MatMul still in int4 (garbled output): print
#    [n.get_friendly_name() for n in ov.Core().read_model(xml).get_ops()] and widen MOE_IGNORE.
#
# 5) DynamicCache.from_legacy_cache / to_legacy_cache renamed: build_cache already falls back to
#    cache.update(k,v,i); for readout swap to cache.layers[i].keys / .values.
#
# 6) sliding-window eviction boundary (< vs <=, measured from block start vs last canvas pos):
#    settle on the A100 by comparing IR-D windowed-mask logits to the torch reference run with a
#    real DynamicCache past 1024 tokens; adjust the (cache_len - window) boundary in dec_sliding_mask
#    and the manifest to match. This is the one interface detail that needs a live A100 check.
# =============================================================================


# ===== UTILITY : disk / memory reclaim (optional; safe to run between convert attempts) =====
# Frees disk pressure from FAILED runs so the ~100GB fp32 IR intermediate has room. Keeps the
# loaded `model` alive (only frees GPU fragmentation). Leaves the HF checkpoint cache ALONE by
# default -- wiping it forces a full ~50GB re-download in CELL 1. Frees DISK, not the convert-time
# RAM spike (for that, add a swapfile on scratch like the 26B MoE notebook does).
import os, gc, shutil, subprocess, torch

WIPE_OUT_DIR  = True    # delete IR .xml/.bin fragments from FAILED converts (safe; rebuilt on retry)
WIPE_TMP      = True    # /tmp + pip cache + misc caches (safe)
WIPE_HF_CACHE = False   # DANGER: deletes the checkpoint -> CELL 1 re-downloads ~50GB

def _du(p):
    try:
        t, u, f = shutil.disk_usage(p); return "%-22s %5.0f GB used / %5.0f GB free" % (p, u/1e9, f/1e9)
    except Exception as e: return "%-22s n/a (%s)" % (p, e)

_mounts = [p for p in ["/content", "/mnt/local-scratch", "/root", "/tmp"] if os.path.exists(p)]
print("BEFORE:"); [print("  " + _du(p)) for p in _mounts]
for p in sorted(set(filter(None, [os.environ.get("HF_HUB_CACHE"), "/root/.cache/huggingface/hub"]))):
    if os.path.isdir(p):
        print("  HF cache:", subprocess.run("du -sh " + p, shell=True, capture_output=True, text=True).stdout.strip())

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()

if WIPE_OUT_DIR and "OUT_DIR" in globals():
    _n = 0
    for fn in os.listdir(OUT_DIR):
        if fn.endswith((".xml", ".bin")): os.remove(os.path.join(OUT_DIR, fn)); _n += 1
    print("cleared %d stale IR fragment(s) in %s" % (_n, OUT_DIR))

if WIPE_TMP:
    subprocess.run("rm -rf /tmp/* ~/.cache/pip /root/.cache/matplotlib 2>/dev/null; pip cache purge 2>/dev/null", shell=True)
    print("cleared /tmp + pip cache")

if WIPE_HF_CACHE:
    for p in set(filter(None, [os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"])):
        if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True); print("WIPED HF cache:", p)
    print(">>> checkpoint cache gone -- CELL 1 will RE-DOWNLOAD ~50GB")

gc.collect()
print("AFTER:"); [print("  " + _du(p)) for p in _mounts]
# =============================================================================
