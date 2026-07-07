# Gemma-4 on Intel Arc with OpenVINO INT4 + bug catalog & working toolkit

Everything we hit (and fixed) getting **Gemma-4 26B-A4B MoE** and **31B dense** heretics running
fast and *coherent to 32K context* on Intel Arc B-series GPUs with OpenVINO GenAI.

**New to OpenVINO? Start with [QUICKSTART.md](QUICKSTART.md).**

**Working models, pre-patched, download and run:**
- [gemma-4-26B-A4B heretic int4-ov](https://huggingface.co/Wondernutts/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) (the fast MoE)
- [gemma-4-31B heretic int4-ov](https://huggingface.co/Wondernutts/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) (the smarter, slower dense)
- [gemma-4-12B heretic int4-ov](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) (the "impossible" one; fits 12-16GB cards; needs GenAI nightly; VISION and AUDIO work via the bundled av_pipeline.py)

| Single Arc Pro B70 (32 GB) | Decode | Prefill (cache-defeated) | Verified context (needle retrieval) |
|---|---|---|---|
| 26B-A4B MoE | ~99 tok/s | pp512 2,879 (2.5x SYCL 1,129); 16K in 14 s; 32K in 61 s | 32K, thinking ON and OFF |
| 31B dense | ~27 tok/s (~19 @6K) | pp512 1,662 (2.8x SYCL 601); 16K in 43 s; 32K exceeds VRAM | 8K thinking / 16K no-think (VRAM-capped, not rope) |
| 12B dense | ~55 tok/s (~26 @6K) | pp512 2,301; 16K in 14 s (no published same-card baseline) | 40K no-think / 16K thinking (GenAI nightly + DQ=0 required); vision + audio via custom pipeline |

Measured on a single Arc Pro B70 (OpenVINO 2026.2): **~99 tok/s decode (1.9x the best published
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

Every issue below was hit in the field between 2026-06 and 2026-07. Versions: OpenVINO / GenAI
2026.2 (2026.3 nightly retested where noted), optimum-intel git-main, transformers 5.5.0.

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

### 5. ContinuousBatchingPipeline garbles Gemma-4 INT4 without one property
Batched inference repeats `thought///`-style junk unless you pass
`{"DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}`. Single-stream `VLMPipeline` is coherent with defaults.
Also: `KV_CACHE_PRECISION=f32` **crashes** PagedAttention ("Incorrect block size ... BY_CHANNEL"), don't use it as a precision workaround.

### 6. Gemma-4 12B (`gemma4_unified`): "unsupported", but it runs. Four stacked fixes.
CORRECTION (2026-07-04): we previously wrote this model off, and so does the ecosystem
([optimum-intel#1764](https://github.com/huggingface/optimum-intel/issues/1764) says it cannot
be exported; GenAI rejects it with "Unsupported VLM model type"). All four of these are
required, and together they work:
1. Spoof `model_type` to `gemma4` in config.json. The rejection is a literal string compare,
   and the 12B text graph is the 26B graph minus one input, so the gemma4 pipeline drives it.
2. Run the GenAI NIGHTLY (2026.3-dev). The 2026.2 stateful decode corrupts this graph (first
   token fine, then garbage); nightly decodes clean. Continuous batching is broken on both.
3. `DYNAMIC_QUANTIZATION_GROUP_SIZE: 0` always. DQ-default garbles the 12B from 4K context.
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
2. `DYNAMIC_QUANTIZATION_GROUP_SIZE: 0`, as everywhere with the 12B.
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

Speeds, all three paths, same card, cache-clean single runs (TTFT includes tokenization):

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
| `colab/COLAB_26B_MoE.py` | Full Colab conversion for the MoE: AWQ INT4, router excluded, verify-before-upload (#3, #4) |
| `colab/COLAB_31B_dense.py` | Colab conversion for the dense 31B |
| `serving/ovserver_moe.py` | OpenAI-compatible `/v1/chat/completions` server on VLMPipeline (no-think/think, rep_pen, ctx cap, bus guard) |
| `serving/start-ov-*.sh` | Launcher examples (device pinning, env config) |
| `tests/coherence_sweep.py` | The 8/16/24/32K long-context coherence test |
| `tests/bench31b_ab.py` | Prefill/decode benchmark with GenAI perf metrics (DQ on/off A/B) |
| `tests/guarded_runner.sh` | Memory-watchdog wrapper for risky loads (#11) |

Set `YOUR_HF_TOKEN_HERE` placeholders before using the Colab scripts.

## Status / known-open

- 26B-A4B MoE: **fully working**, published, needle-retrieval verified to 32K (thinking on and off).
- 31B dense: **published.** Needle retrieval passes at 8K with thinking and 16K without; the wall is the card's 32 GB (18.6 GB weights + KV), not the rope. Numbers in the table up top.
- 12B dense: **published.** The four-fix recipe is issue #6; GenAI nightly required. Vision + audio work via av_pipeline.py in the model repo AND natively in C++ via gemma4-unified-audio.patch (section 7), the first audio input OpenVINO GenAI has had.
- OpenVINO 2026.3 nightly does **not** fix #2 on its own (retested 2026-07-03).
- Past 32K the practical wall on my 30GB-host-RAM box is host memory during prefill (both
  single-shot and chunked/continuous-batching paths); the model itself is rated to 262K.
  I don't have the hardware for that and I don't need it. I'm using the MOE for RP in a Skyrim Chim AI Mod
  Project I've been working on. B70 isn't fast enough for the 31B realtime chat but almost nothing is.   
  That will be up to Intel to test past where I got, if they can get around to it soon.
