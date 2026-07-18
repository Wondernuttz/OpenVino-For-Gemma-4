# DiffusionGemma 26B-A4B on Intel Arc: running notes

Field notes for running the OpenVINO port of DiffusionGemma 26B-A4B (block-diffusion
MoE) on Intel Arc. This is, to our knowledge, the first OpenVINO port of a diffusion
LLM (first text generated on an Arc B70, 2026-07-10). Intel's llm-scaler-vllm added
its own diffusiongemma support for Arc Pro in July 2026 via a different runtime.

Everything here was measured on one box: 2x Arc B70, driver 26.14.37833, OpenVINO 2026.2,
30GB host RAM. Export details live in `DIFFUSION_GEMMA_OV_SPEC.md` and
`colab/COLAB_DiffusionGemma.py`.

## Artifacts

| HF repo | What it is |
|---|---|
| `Wondernutts/diffusiongemma-26B-A4B-it-openvino-int4-c32` | The build to use. Expert capacity C=32, SC8 baked, RoPE LUT, 14.8GB. |
| `Wondernutts/diffusiongemma-26B-A4B-it-openvino-int4` | Original C=48 export, kept as the full-capacity reference. |
| `Wondernutts/diffusiongemma-26B-A4B-it-openvino-int4-c24` | C=24 curve point. Measurably worse quality, do not deploy. |

Capacity curve (24-step blocks, 12 draws each, same prompt, same card):

| Build | Step time | Bad draws | Reply length floor |
|---|---|---|---|
| C=48, DQ on | 154ms | 5/12 | 15 tok |
| C=32, DQ off | 134ms | 0/12 | 47 tok |
| C=24, DQ off | 125ms | 2/12 | 7 tok |

C=32 with dynamic quantization disabled wins on both speed and quality. The C=48
numbers were taken through the damaged DQ path (see bug 4); C=48 with DQ disabled
is untested.

## Quick start

```bash
# one-shot generation
DG_DQ0=1 python serving/dg_sampler.py /path/to/model GPU.1 "Why is the sky blue?"

# OpenAI-compatible server (imports the sampler)
OV_MODEL=/path/to/model OV_DEVICE=GPU.1 OV_PORT=8092 DG_DQ0=1 python serving/ovserver_dg.py
```

Validated serving recipe (every knob is an env var, defaults are sane):

```
DG_DQ0=1          mandatory on Arc, see bug 4 (also sharpens logits)
DG_EBOUND=0.1     entropy lock bound; higher trades quality for length
DG_STEPS=24       denoise cap per 256-token block
DG_DUMMY=48       dummy cache prefix; never raise it (bug 7)
DG_WARM=4         no-lock warmup steps
DG_REVH=0.05      confident-revision unlock
DG_ADJPEN=3.0     adjacent-duplicate penalty (diffusion-specific artifact)
DG_MAX_PROMPT=660 keeps cache+canvas inside the 1024 sliding window
```

Server-side: reject prompts under ~2500 chars (thin prompts are draw-unstable) and
cap at 2 blocks for chat. Block time ~4s, chat replies ~5s end to end on a B70.

## Arc driver bug catalog (the cost of admission)

All of these are worked around in `dg_sampler.py`. `CL_OUT_OF_RESOURCES` on Arc
means a kernel page fault, not VRAM exhaustion.

1. **First-inference allocation fault.** Cold-starting most shapes page-faults.
   Fix: warm shapes in a gentle ascending ladder after compile (the sampler does
   this); never jump, reorder, or extrapolate past the top rung.
2. **Causal seq-len 9..32 is broken** (except exactly 8 and 16), even warm.
   Fix: pad every encode to >=48 tokens with masked pads, slice the KV after.
3. **bf16 gemm writes garbage logits rows >=225 at M=256**, and the GPU plugin
   silently downcasts an fp32 lm_head to f16. Fix: the sampler cuts the lm_head
   out of the backbone at load and runs it as a separate static fp32 model.
4. **The s8xs4 dynamic-quantization matmul faults** on the C=24/C=32 expert shapes
   at every group size (32/64/128/default). Fix: `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`
   (`DG_DQ0=1`). Note: the OpenVINO compile cache does not key on this property;
   wipe `.ovcache` after changing compile config or you get a stale blob.
5. **bf16 thin dynamic eltwise chains fail layout selection** at compile ("No layout
   format available"). The export keeps the gate-renorm chain in f32 for this reason.
6. **TopK with sort="none" crashes; sorted works.** And never fuse a TopK behind a
   gemm in one graph: the planner emits a pathological kernel. Separate models.
7. **Fully-masked attention regions produce NaN.** Keep the dummy cache prefix at
   48; masked regions must never reach kernel block size.
8. Microbench doctrine: never trust back-to-back kernel benches for in-loop cost
   (results move 2-3x in both directions vs inside a real denoise loop), and never
   A/B against a baseline from another session.

## What is inside the sampler

- **Sticky entropy-bound locking**: positions lock when entropy drops below the
  bound and stay locked (the reference re-randomizes accepted positions and never
  converges); warmup steps, confident-revision unlock, frequency penalty with
  EOS/PAD exempt.
- **Adjacent-duplicate penalty** (`DG_ADJPEN`): parallel denoising lets neighboring
  positions lock the same token ("the the", "(I I"). A candidate matching an
  already-locked neighbor gets docked. AR-style repetition penalty does not target
  this failure and hurts legitimate repeats.
- **SC8**: self-conditioning fed from the top-8 candidates only instead of the full
  262k softmax against the embedding table (concept cross-validated against
  Echo9Zulu's Arcaine engine). Baked into the c32 IR; the sampler applies it as
  load-time graph surgery on older exports.
- **Two-stage exact top-64** over the 262k vocab: segment-max reduce, TopK over
  segment maxes, gather, TopK again. Same result as a monolithic TopK at a
  fraction of the in-loop cost.
- **Device-resident denoise loop**: prefix KV uploaded once per block, scaled
  logits stay on device as the next step's self-conditioning input, unused KV
  outputs bound to remote tensors. Host reads 128KB per step instead of ~950MB.
- **Chunked prompt encode**: capacity-dispatch exports verify drop-free routing at
  T=256 only, so longer encodes split into causal chunks against the growing KV
  (exact math, and measurably better encodes on long prompts).
- **Adaptive stop**: trim-boundary stall detection; mostly mercy-kills weak draws
  early rather than shortening good ones.
- **Canvas seeding** (`DG_SEED_TEXT`): lock a reply prefix on the canvas before
  denoising, a diffusion-native alternative to weight-level uncensoring. Off by
  default; it was compensating for DQ damage and is unnecessary with DQ0.

## Known limits

- Sliding-window masks are not implemented: total context (dummy + prompt + canvas)
  must stay under 1024, hence `DG_MAX_PROMPT=660`. This is the top open item.
- The per-step host work that remains (sampling math and lock bookkeeping on the
  [256,64] top-k slice, ~15ms/step) is numpy. Expressing it as a small OV graph,
  likely CPU-placed, would let the compiler fuse it and remove the per-step python
  overhead; folding it into the device chain would remove the per-step host sync
  entirely (suggested by Echo9Zulu). The fat host paths (full-vocab softmax, top-k,
  self-conditioning) already moved in-graph in earlier passes.
- Long-form generation past block 1 is weak (lock rate collapses; the trimmer
  stops cleanly instead).
- The checkpoint has a vision tower; this export traces the text path only.
- Structured output converges fastest (JSON template fill in 12-15 steps, ~2s);
  free prose runs the full step cap. Convergence tracks layout ambiguity, not
  semantic difficulty.
