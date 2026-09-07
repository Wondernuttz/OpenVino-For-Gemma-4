# Gemma-4 on Intel Arc with OpenVINO INT4 + bug catalog & working toolkit

Runtime patches, conversion tools, and measured results for **Gemma-4 on Intel Arc**.
Context and quality validation are specific to each model and runtime profile; see the reports below.

## 26B at a glance — Wondernuttz's custom OpenVINO fork

**Latest update: corrected wide-query cached prefill.** On the B70/Linux Heretic
reference, **24,576-token uncached PP rose from 4,841 to 5,653 tok/s (+16.8%)**;
the cleaned deployment build confirmed **5,667 tok/s**. Short decode stayed about
112 tok/s. The 26B bots have the update, with their existing16K context unchanged.
Read the [configuration, repeated U4 checks, and remaining baseline caveat](WIDEQ_24K_20260906.md).
This adds to the earlier binary-lookup gains below; no model reconversion is needed.

- **Tested hardware and OS:** one **Intel Arc Pro B70, 32 GB VRAM, running Linux**.
- **Custom runtime:** [Wondernuttz's OpenVINO fork](https://github.com/Wondernuttz/openvino/tree/arc-xe2-gemma4-pa-2026.4), branch `arc-xe2-gemma4-pa-2026.4`, with matching OpenVINO GenAI. This combines custom Gemma/Arc optimizations with credited Intel upstream fixes—not an unmodified stock wheel.
- **Model:** Gemma 4 **26B-A4B Heretic**, OpenVINO INT4. Related fine-tunes have separate quality gates; the headline speeds are not automatically their measured speeds.
- **Performance:** **7,435 tokens/s full prefill at 6,622 tokens**, with prefix reuse OFF. Compiled kernels are warmed; model loading and cold compilation are not included.
- **VRAM:** approximately **26.5 GiB sampled peak at 24K input**, including an 8 GiB KV-cache allocation, in the reference test. Repository/download size is not total runtime memory. A 12 GB GPU minimum has not been established.
- **Context:** 24K release checks plus separate 32K reference probes. **131K RoPE LUT coverage does not mean validated 131K usable context.** Reserve output space within total-token limits. **StyleTune V2 remains held on its older 16K profile.**
- **Getting started:** the optimized path needs the patched runtime and documented scheduler settings; see the [reference setup](https://github.com/Wondernuttz/openvino/blob/7b27ac8eb88faec682d4c96dfba749b93df32285/WONDERNUTTZ_GEMMA4_PREFILL_20260906.md#reference-text-only-pipeline-configuration). Valid existing exports do not need recompression. These IR weights load through OpenVINO GenAI, not directly through Transformers.
- **Not covered by these benchmarks:** Windows, CPU speed, other Arc cards, native vision/audio, or dense 31B performance.

## Earlier September benchmark: 7,435 tok/s uncached prefill on one Arc Pro B70

**September 6, 2026 — Gemma 4 26B-A4B Heretic, INT4, one 32 GB B70 on Linux, using Wondernuttz's custom OpenVINO fork.**
The new grouped-MoE binary lookup improves long-prompt processing without changing
model weights, expert routing, or quantization settings.

| Input tokens | Original lookup PP tok/s | Patched lookup PP tok/s | Improvement |
|---:|---:|---:|---:|
| 4,096 | 6,681.7 | **7,224.4** | +8.1% |
| 6,622 | 6,028.1 | **7,435.1** | +23.3% |
| 15,872 | 2,920.9 | **6,866.4** | +135.1% / 2.35x |
| 24,576 | 2,734.8 | **4,767.0** | +74.3% |

**Full prompt processing, not prefix-cache hits.** Prefix reuse was disabled.
These are warmed-kernel measurements, not cold-start/model-load timings; PP is input
tokens divided by time to first token. Both configurations use the same binary,
DQ128, U4 KV cache, 8 GiB cache allocation, batch16384, and one sequence. The first
three rows use A/B/B/A order; the 24K row uses one matched process pair.

At 24K, time to first token fell from **~8.99 to ~5.15 seconds**. Decode remained
approximately **93 tok/s at 24K** and **111 tok/s on short prompts**; no meaningful
decode gain is claimed. A separate clean Release gate reproduced **4,771 PP tok/s
at 24K (+76% against its matched control)**.

**[Full benchmark, methodology, configuration, and patch explanation](https://github.com/Wondernuttz/openvino/blob/arc-xe2-gemma4-pa-2026.4/WONDERNUTTZ_GEMMA4_PREFILL_20260906.md)**

**[Release validation, per-checkpoint quality gates, and known limitations](https://github.com/Wondernuttz/openvino/blob/arc-xe2-gemma4-pa-2026.4/WONDERNUTTZ_GEMMA4_RELEASE_GATE_20260906.md)**

[Patched OpenVINO source branch](https://github.com/Wondernuttz/openvino/tree/arc-xe2-gemma4-pa-2026.4)
— this optimization requires a matching patched runtime; setting the switch on a
stock wheel does not add it. Existing valid model exports do not need recompression.

**Scope:** measured on the 26B-A4B Heretic reference model, not a universal per-tune
speed or intelligence claim. Bounded regression/coherence checks passed, but they
are not an exhaustive quality evaluation. **StyleTune V2 remains held on its older
16K profile** following a long-context RP discrepancy. Dense 31B does not use this
MoE optimization. Known test/teardown GPU faults are documented in the linked
reports; this is not an all-OOMs-fixed or multi-user stability claim.

## Models and getting started

**New to OpenVINO? Start with [QUICKSTART.md](QUICKSTART.md).**

**Docker / B50 / 12B:** [portable server, explicit profiles, and Docker instructions](serving/README.md).
The default container uses pinned upstream OpenVINO for 12B text; the custom-fork
image requires matching runtime wheels. The 26B/B70 memory profile is not a B50 profile.

**Other models and benchmark reports:** [Qwen, Magnum, INTBIT, INTERNARY, and the complete model index](OTHER_MODELS.md).

**Working models, pre-patched, download and run:**
- [gemma-4-26B-A4B heretic int4-ov](https://huggingface.co/Wondernutts/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) (the fast MoE)
- [gemma-4-31B heretic int4-ov](https://huggingface.co/Wondernutts/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) (the smarter, slower dense)
- [gemma-4-12B heretic int4-ov](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) (the "impossible" one; fits 12-16GB cards; needs GenAI nightly; VISION and AUDIO work via the bundled av_pipeline.py)

## Earlier benchmark history (pre-September update)

The following measurements describe older profiles, not the September lookup
results above. Context limits and validation must not be mixed across profiles.

| Single Arc Pro B70 (32 GB) | Decode | Prefill | Verified context |
|---|---|---|---|
| 26B-A4B MoE, 2026.4 fork | 112.2 tok/s short; 94.9 after 6,622 | 5,827 tok/s at 6,622; 6,500 tok/s at 4K | 16K on CB; separate stock single-stream path passed 32K |
| 31B dense | ~27 tok/s (~19 @6K) | pp512 1,662; 16K in 43 s | 8K thinking / 16K no-think |
| 12B dense | ~55 tok/s (~26 @6K) | pp512 2,301; 16K in 14 s | 40K no-think / 16K thinking; vision and audio |

The July 26B result used the
[`arc-xe2-gemma4-pa-2026.4`](https://github.com/Wondernuttz/openvino/tree/arc-xe2-gemma4-pa-2026.4)
fork at commit `2c82358676`. The final change allows Gemma's five 512-head global-attention
layers to select OpenVINO's existing Xe2 micro-SDPA/XMX route. Global paged-attention device
time fell from 280.920 ms to 67.546 ms. Full-model 6,622-token prefill rose from the previous
accepted 4,448 tok/s to a 5,827 tok/s sustained mean, with the same output hash and the same
4/4 coherence result.

The complete settings, benchmark history, context curve, rejected runs and build instructions
are in [BENCHMARK_HISTORY_GEMMA4_26B.md](BENCHMARK_HISTORY_GEMMA4_26B.md). The older OpenVINO
2026.2 `VLMPipeline` numbers below remain useful for compatibility and 32K testing, but they are
not the runtime behind the September headline result. The optimized fork path has been tested on Linux
only.

That older profile was verified on the local Gemma-4 26B StyleTune V2 build at 5,827 PP and
112.2 decode, with the retrieval gate passing 4/4. See
[STYLETUNE26_XMX512_VALIDATION.md](STYLETUNE26_XMX512_VALIDATION.md). This is not
approval of the September 24K rollout for StyleTune; see the hold above.

Earlier compatibility-path measurements on a single Arc Pro B70 (OpenVINO 2026.2): **~99 tok/s decode (1.9x the best published
same-card SYCL figure), ~2,900 tok/s prefill at matched pp512 vs SYCL 1,129 (about 2.5x), needle
retrieval verified at 8/16/32K with thinking OFF and ON** (thinking used to collapse at 2-4K
before the rope patch). As of writing there are no other public OpenVINO Gemma-4-on-Arc
datapoints; the best published same-card baseline (llama.cpp SYCL, PMZFX) is 1,129 tok/s
prompt-processing (pp512) and 52.6 tok/s decode. CORRECTION HISTORY, for transparency: earlier
prefill figures published here and on the cards were re-measured with cache-defeating unique
prompts after two methodology bugs were found. The Qwen-family figures had been inflated by KV
prefix-cache reuse in the benchmark (warm run cached the prompt; the giveaway was time-to-first-
token falling as prompt length grew). The original 26B figure came from a hand-timed request with
a guessed overhead subtraction. The 31B and 12B originals were re-validated by the clean
re-measurement (within a few percent) and stand. All numbers above are from the corrected,
cache-defeated method, TTFT-based and therefore slightly conservative.

---

## The bug catalog

Every issue below was hit in the field between 2026-06 and 2026-07. The original catalog covers
OpenVINO 2026.2 and 2026.3 nightlies. The 26B continuous-batching update uses the matching
OpenVINO 2026.4 fork and GenAI commits listed in the benchmark history.

### 1. Those "zeroed" RoPE frequencies are NOT export corruption, don't "fix" them
The exported global-RoPE `inv_freq` constant has **192 of 256 values equal to zero** and it looks
exactly like converter breakage. It isn't. Gemma-4's global attention uses **proportional
(partial) RoPE** (`rope_type: "proportional"`, `partial_rotary_factor: 0.25` in
`text_config.rope_parameters`): only the first quarter of the frequency pairs are rotated, the
rest are position-agnostic *by design* (zero freq → cos=1/sin=0 → identity). See
`transformers/modeling_rope_utils.py::_compute_proportional_rope_parameters`, it literally
concatenates zeros.
Spent weeks "repairing" that spectrum with the standard geometric formula
([`patches/ov_rope_const_fix.py`](patches/ov_rope_const_fix.py), kept for the historical record,
**do not use it**). The MoE tolerated the spurious rotation; the dense 31B visibly degraded at
16K from it. The *actual* long-context killer was #2 all along. Lesson: check
`rope_parameters` in the source config before declaring an export broken.

### 2. Intel GPU plugin executes RoPE in fp16 is hard wall at ~16-20K even with correct constants
The graph computes `sin/cos(position x inv_freq)` in f32, but the GPU plugin downcasts execution
to fp16. At position 20,000 the rotation angle is ~20,000 radians; fp16 resolution at that
magnitude is ±16. The angles are garbage before sin/cos ever run.
Dead ends we proved so you don't have to: whole-model f32 execution OOMs a 32GB card;
hand-setting `precise`/`disable_fp16_compression` rt_info on an exported model is **silently
stripped** by `ov.save_model` (2026.2 *and* 2026.3 nightly); there is no Python API to mark
precision-sensitive ops post-export.
**Fix:** [`patches/ov_rope_lut.py`](patches/ov_rope_lut.py): replace the runtime angle math with
precomputed f32 sin/cos lookup tables + `Gather(position_ids)`, built from the graph's own
inv_freq constants **with the p-RoPE zeros preserved** (see #1). Table values live in [-1, 1],
which fp16 represents fine; a Gather has no arithmetic to corrupt. Verified: 26B-A4B coherent at
32K. Bonus: slightly *faster* than the subgraph it replaces.
Usage: `python ov_rope_lut.py /path/to/original-int4-export /path/to/output-dir`

### 3. Quantizing the MoE router breaks GPU loading
 Doesn't INT4-quantizing the router kills the GPU plugin's MoE fusion; the model exports fine and then
**won't load**. `optimum-cli` has no flag for this; you must use the Python API with
`ignored_scope={"patterns": [".*router.*"]}` (matches Intel's own published config). Should have caught 
this at first, but I didn't, so don't make that mistake I did or you'll find your redoing this on
colab, and the MOE needs almost everything the high-ram runtime has to offer....and conversation is NOT 
on 80GB VRAM. Doesn't take too long though. 
See [`colab/COLAB_26B_MoE.py`](colab/COLAB_26B_MoE.py) CELL 3.


### 4. `awq=True` is silently ignored, you get plain INT4 garbage
`OVWeightQuantizationConfig(awq=True, ...)` does nothing. You must pass
`quant_method="awq"`. Plain data-free INT4 without AWQ produced incoherent output on the MoE;
AWQ (Intel's recipe, group size 64) is load-bearing for quality.

### 5. Continuous batching requires the corrected 2026.4 stack
On OpenVINO 2026.2, batched inference repeats `thought///`-style junk unless
`DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, and long prompts can hit the old GenAI input-buffer bug.
The July 23 OpenVINO 2026.4 and matching GenAI stack contain the Gemma-4 PagedAttention graph,
token layout and sliding-window fixes. That path is coherent at DQ128 and is the base for the
current benchmark. Do not mix a 2026.4 GPU plugin with a 2026.2 runtime. Also,
`KV_CACHE_PRECISION=f32` crashes PagedAttention with a `BY_CHANNEL` block-size error and is not a
precision workaround.

### 6. Gemma-4 12B (`gemma4_unified`): "unsupported", but it runs. Four stacked fixes.

**Current DQ setting depends on the pipeline:**

- **Text-only `VLMPipeline`: `DYNAMIC_QUANTIZATION_GROUP_SIZE=128`.** The later matched sweep passed all four retrieval checks at both 6,620 and 30,000 input tokens, with byte-identical answers to DQGS 0. This supersedes the old blanket "DQ0 always" recommendation.
- **Python `av_pipeline.py` and native vision/audio: keep `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`.** Those paths were not included in the text-only DQ128 sweep.

See the [matched DQGS measurements](OTHER_MODELS/GEMMA_QWEN_DQ128.md#gemma-12b-dense)
and the [12B model card's text-only example](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov#how-to-run).
The 26B grouped-MoE lookup optimization does not apply to dense 12B; do not copy the
26B scheduler/cache profile onto it. In OpenArc, the corresponding loader is `vlm`
(`VLMPipeline`), even for text-only requests. Confirm the installed runtime versions
and model-load error before concluding that a custom Docker build is required.

The following is the original July compatibility history, with the DQ correction above:

CORRECTION (2026-07-04): we previously wrote this model off, and so does the ecosystem
([optimum-intel#1764](https://github.com/huggingface/optimum-intel/issues/1764) says it cannot
be exported; GenAI rejects it with "Unsupported VLM model type"). All four of these are
required, and together they work:
1. Spoof `model_type` to `gemma4` in config.json. The rejection is a literal string compare,
   and the 12B text graph is the 26B graph minus one input, so the gemma4 pipeline drives it.
2. Run the GenAI NIGHTLY (2026.3-dev). The 2026.2 stateful decode corrupts this graph (first
   token fine, then garbage); nightly decodes clean. Continuous batching is broken on both.
3. The original workaround was DQGS 0 after an earlier default garbled long context.
   The later validated setting is **DQGS 128 for text-only `VLMPipeline`**;
   keep **DQGS 0 for vision/audio** as described above.
4. The rope LUT patch (#2 above); the fp16 wall hits the 12B earlier, around 8K.
Result: needle retrieval verified at 40K no-think and 16K with thinking, ~55 tok/s on a B70,
7.5 GB. Published:
[the 12B repo](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov).

UPDATE (2026-07-06): no longer text-only. VISION AND AUDIO BOTH WORK, which as far as we can
find makes this the first gemma4_unified anywhere with sight and hearing on OpenVINO. GenAI's
blocker was preprocessing, not the model: the unified architecture patchifies at 16x16 and then
merges 3x3 neighbors into 6912-wide model patches, while the gemma4 pipeline feeds unmerged
768-wide patches. And the audio side has NO encoder tower at all by design: raw 16 kHz waveform
is chunked into 640-sample frames (40 ms per token), RMSNorm-ed (no learned scale), and lifted
into LM space by ONE 5 MB linear projection that was sitting in the checkpoint all along. The
12B repo now ships `av_pipeline.py` (correct preprocessing via transformers-main
Gemma4UnifiedImageProcessor + manual stateful generate loop over the exported IRs) plus
`audio_projection.npy`. Verified: accurate detailed description of real 2752x1536 cover art,
verbatim transcription of a 6 s TTS clip, correct understanding of an 18 s real microphone
recording. Decode 50-54 tok/s (~95% of the C++ text pipeline), image preprocess+vision IR
~0.13 s, TTFT ~0.35 s after an image.

### 7. Native C++ vision AND audio: the gemma4-unified-audio patch

UPDATE (2026-07-06, same day as the Python pipeline): both modalities also run NATIVELY in
OpenVINO GenAI's C++ VLMPipeline via a 259-line patch in this repo
([gemma4-unified-audio.patch](gemma4-unified-audio.patch)). As far as we can find this is
the first audio input support OpenVINO GenAI has had for any model. Model files and the
Python-pipeline alternative live in the
[12B HF repo](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov).

Vision costs zero source changes on current GenAI main (their PR #4001, merged 2026-07-03,
added the unified 48x48 patch merge), but has three usage rules, and getting any wrong
looks like a broken model instead of a broken call:

1. `model_type` must be `gemma4_unified` (the real name, NOT the gemma4 spoof this toolkit
   recommends for older runtimes; the spoofed path crashes with images: MatMul shape error
   then CL_OUT_OF_RESOURCES).
2. `DYNAMIC_QUANTIZATION_GROUP_SIZE: 0` for this native vision/audio path.
   The separate text-only `VLMPipeline` path is validated at DQGS 128 (section 6).
3. The prompt must contain `<|image|>` where the image belongs. Without it GenAI prepends
   the image block BEFORE `<bos>` and the model half-works: shapes recognized, colors and
   bindings scrambled (a solid red square answers "Green"). We verified GenAI's
   preprocessing is numerically identical to the HF processor, so that failure mode is
   pure token order.

Audio is the actual patch. The unified architecture has no audio tower: raw 16 kHz mono is
chunked into 640-sample frames (40 ms per soft token), RMSNorm-ed without scale, and lifted
into LM space by a single Linear(640, 3840). The patch teaches the gemma4/gemma4_unified
path to accept f32 waveform tensors through the existing images API (shape
[nsamples, 1, 1]), run them through an `openvino_audio_embeddings_model.xml` compiled next
to the vision model (build it with [make_audio_ir.py](make_audio_ir.py) from the
`audio_projection.npy` in the 12B repo), and splice at `<|audio|>` tags with correct
attention marking.

Verified on a B70: word-for-word transcription of a 6 s TTS clip and an 18 s real
microphone recording, identical output to the Python pipeline; vision regression clean.

Historical DQGS 0 comparison: all three paths, same card, cache-clean single runs
(TTFT includes tokenization). The later text-only DQ128 sweep is linked in section 6:

| Metric | Text-only (stock GenAI) | Native C++ AV (this patch) | Python (av_pipeline.py) |
|---|---|---|---|
| Prefill 512 | 2,301 tok/s (0.22 s) | 0.63 s TTFT | 0.40 s TTFT |
| Prefill 2K | 3,170 tok/s (0.65 s) | 0.93 s (~2,200 tok/s) | 1.18 s |
| Prefill 6K | 2,120 tok/s (2.9 s) | 2.09 s (~2,940 tok/s) | 3.0-5.4 s |
| Image request e2e (2752x1536) | n/a | 0.73 s to first token | ~0.44 s |
| 18 s audio request e2e | n/a | 0.62 s to first token | ~0.39 s |
| Decode, short | ~55 tok/s | ~49 tok/s | 50-54 tok/s |
| Decode at 6K | ~26 tok/s | 16.8 (12.1 with image in context) | 25.9 tok/s |

Multimodality itself is nearly free (encoder-free architecture: an image is 264 context
tokens, 18 s of audio is 458; the Python path decodes 25.9 tok/s at 6K vs 26 text-only).
The native path wins first-token latency at depth but currently decodes slower at deep
context (per-step overhead in GenAI's unified branch, cause not yet isolated); the Python
path holds decode speed everywhere but pays 2x on deep prefill. Today the Python pipeline
is the best all-rounder; the native patch buys serving-grade C++ machinery and the fastest
first token on long prompts.

Apply: `git apply gemma4-unified-audio.patch` on openvino.genai main, build as usual, run
`make_audio_ir.py <model_dir>` once, and mind the three rules above.

### 7. Gemma-4's own repetition collapse (it is not an OpenVINO bug)
Documented upstream ([google-deepmind/gemma#622](https://github.com/google-deepmind/gemma/issues/622)):
token-repetition collapse during long generation on **every** backend, firing most reliably under
grammar-constrained (structured JSON) output. Mitigations: repetition_penalty ~1.2, never use
`response_format`/JSON grammar mode, serve no-thinkif it becomes an issue under Google fixes.
For now, both models are getting the best speeds I've seen and maintaining coherent outputs on everything a single
B70 can fit, I have not tested two cards.

### 8. GenAI registers Gemma-4 as VLM-only
Use `VLMPipeline`, not `LLMPipeline`, even for text-only. Needs transformers 5.5.x at export time.

### 9. You can't strip the reasoning channel from the text output
The `<channel|>` boundary token is eaten by the detokenizer, `VLMDecodedResults` exposes no token
IDs, and `r.parsed` is empty. If you enable thinking and need to strip it server-side: pass a
custom `StreamerBase` that collects token IDs, then re-decode with `skip_special_tokens=False`
and split on `<channel|>`. Implemented in [`serving/ovserver_moe.py`](serving/ovserver_moe.py).

### 10. GPU device enumeration will burn you
On a 2-dGPU + iGPU box: OpenVINO's `GPU.0/1/2` order ≠ `xpu-smi` device order ≠
`ZE_AFFINITY_MASK` order. Always confirm with
`core.get_property("GPU.N", "DEVICE_PCI_INFO")` before loading, or you'll land a test model on
your production card. The serving scripts here take an `OV_EXPECT_BUS` guard that aborts on
mismatch.

### 11. Host-RAM OOM during large-model GPU compile
Compiling the 31B (18GB INT4) onto the GPU peaks well above 30GB host RAM and the allocations
are fast enough to outrun the OOM killer's mercy, it took down sshd and unrelated services.
[`tests/guarded_runner.sh`](tests/guarded_runner.sh) wraps risky loads with a watchdog that kills
the test process (not your box) below a MemAvailable floor.

### 12. Prompt format: it's `<|turn>`, not `<start_of_turn>`
These QAT-heretic builds use `<|turn>role\n...<turn|>` with a `<|channel>thought` reasoning
channel (check `chat_template.jinja`). Classic Gemma format *tolerates* but leaks a stray
`thought` prefix and runs ~20% slower. Thinking control is binary: pre-close the channel
(`<|channel>thought\n<channel|>`) for fast no-think, or put `<|think|>` in the system turn.

---

## Repo layout

| Path | What it is |
|---|---|
| `patches/ov_rope_const_fix.py` | **Historical record only do not use** (our #1 misdiagnosis, kept as a warning!) |
| `patches/ov_rope_lut.py` | The sin/cos LUT graph patch for full 32K coherence (#2), p-RoPE-aware; run `python ov_rope_lut.py SRC_DIR DST_DIR` |
| `patches/dg_rope_lut.py` | The same LUT patch adapted for the DiffusionGemma unified IR |
| `patches/dg_bake_sc8.py` | Bakes the SC8 top-8 self-conditioning rewrite into the diffusion IR and prunes a 392MB trace-materialized duplicate of the embedding table |
| `colab/COLAB_26B_MoE.py` | Full Colab conversion for the MoE: AWQ INT4, router excluded, verify-before-upload (#3, #4) |
| `colab/COLAB_31B_dense.py` | Colab conversion for the dense 31B |
| `colab/COLAB_DiffusionGemma.py` | **DiffusionGemma 26B-A4B block-diffusion export, the first diffusion LLM to run on OpenVINO/Arc.** Unified single-backbone IR (tied encoder/decoder weights deduped, `apply_sc` role switch), capacity-dispatch top-8 MoE with `SPEED_CAPACITY` knob (C=32 is the validated ship point) and f32 gate-renorm, fp32 lm_head/softcap tail, INT4 AWQ bulk. Needs an A100 80GB runtime; base model is Apache-2.0. Running it on Arc requires `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, see `DIFFUSION_NOTES.md` |
| `serving/dg_sampler.py` | The block-diffusion sampler for Arc: warmup ladder, lm_head tail split, sticky entropy-bound locking, adjacent-duplicate penalty, SC8, two-stage top-64, device-resident denoise loop, chunked encode, adaptive stop |
| `serving/ovserver_dg.py` | OpenAI-compatible server wrapping the sampler (thought-leak filter, prompt gates, block cap) |
| `DIFFUSION_NOTES.md` | Diffusion running notes: validated recipe, capacity curve, the Arc driver bug catalog, sampler internals |
| `DIFFUSION_GEMMA_OV_SPEC.md` | The architecture spec the export was built from (attention layout, SC wiring, sampler contract) |
| `colab/COLAB_12B_heretic_EXPERIMENTAL.py` | Experimental Colab conversion for the 12B heretic (dense recipe variant) |
| `serving/ovserver_moe.py` | OpenAI-compatible `/v1/chat/completions` server on VLMPipeline (no-think/think, rep_pen, ctx cap, bus guard) |
| `serving/start-ov-*.sh` | Launcher examples (device pinning, env config) |
| `tests/coherence_sweep.py` | The 8/16/24/32K long-context coherence test |
| `tests/bench31b_ab.py` | Prefill/decode benchmark with GenAI perf metrics (DQ on/off A/B) |
| `tests/guarded_runner.sh` | Memory-watchdog wrapper for risky loads (#11) |

Set `YOUR_HF_TOKEN_HERE` placeholders before using the Colab scripts.

## Status / known-open

- 26B-A4B MoE: **fully working**, published, needle-retrieval verified to 32K (thinking on and off).
- DiffusionGemma 26B-A4B: **working end to end on Arc** (C=32 build, ~134ms/step, ~4s per 256-token block, chat replies ~5s). Sampler + server + notes in this repo; model repos linked from `DIFFUSION_NOTES.md`. Open: sliding-window masks (context capped under 1024), long-form past block 1, vision path unexported.
- 31B dense: **published.** Needle retrieval passes at 8K with thinking and 16K without; the wall is the card's 32 GB (18.6 GB weights + KV), not the rope. Numbers in the table up top.
- 12B dense: **published.** The four-fix recipe is issue #6; GenAI nightly required. Vision + audio work via av_pipeline.py in the model repo AND natively in C++ via gemma4-unified-audio.patch (section 7), the first audio input OpenVINO GenAI has had.
- OpenVINO 2026.3 nightly does **not** fix #2 on its own (retested 2026-07-03).
- Past 32K the practical wall on my 30GB-host-RAM box is host memory during prefill (both
  single-shot and chunked/continuous-batching paths); the model itself is rated to 262K.
  I don't have the hardware for that and I don't need it. I'm using the MOE for RP in a Skyrim Chim AI Mod
  Project I've been working on. B70 isn't fast enough for the 31B realtime chat but almost nothing is.   
  That will be up to Intel to test past where I got, if they can get around to it soon.
