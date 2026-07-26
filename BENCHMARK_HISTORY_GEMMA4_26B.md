# Gemma-4 26B-A4B INT4 on one Arc Pro B70

Last updated: 2026-07-26

This is the benchmark and fix history for the Wondernuttz OpenVINO INT4 build of
Gemma-4 26B-A4B. Every accepted number below came from one Intel Arc Pro B70
with 32 GB VRAM. Prompt-processing tests used one request at a time and unique
prompt contents so prefix-cache hits could not inflate the result.

## Current result

Tested stack:

- OpenVINO fork: `Wondernuttz/openvino`
- Branch: `arc-xe2-gemma4-pa-2026.4`
- Tested code commit: `2c82358676`
- OpenVINO base: `2c3fe6fa4f195e0b49b31bde46c72867fc58a6dd`
- OpenVINO GenAI: `79bc246970146922a385b6c0342f185b45478f4b`
- GPU plugin SHA-256: `5c2085a1f7a8edc86d782d8da805ffd98e47c5035dffb6722b8e0e0945a1959c`

Runtime settings:

```text
DYNAMIC_QUANTIZATION_GROUP_SIZE=128
max_num_batched_tokens=8192
cache_size=8 GB
enable_prefix_caching=true
MOE_USE_GROUPED_GEMM_PREFILL=1
MOE_MICRO_GEMM_N_HINT=128
```

The prompts differed between runs. Prefix caching was enabled for deployment
parity, but no accepted PP number is a cached-prefix result.

| Test | Result |
| --- | ---: |
| 6,622-token cold PP | 5,125.7 tok/s |
| 6,622-token sustained PP, run 1 | 5,821.9 tok/s |
| 6,622-token sustained PP, run 2 | 5,832.5 tok/s |
| 6,622-token sustained PP, mean | **5,827.2 tok/s** |
| 6,622-token coherence PP | 5,779.3 tok/s |
| Short-context decode | **112.2 tok/s** |
| Decode after 6,622 tokens | 94.9 tok/s |

The coherence run retrieved three facts placed far apart in the prompt and
continued the requested style. It passed 4/4 checks. Output SHA-256:
`b123146233af2aac9e725826d9011513ee4c3dc9ec4634fd892e6937f06afb58`.

## Gain history

| Stage | 6,622 PP tok/s | Decode tok/s | Coherence | Change |
| --- | ---: | ---: | --- | --- |
| July 6 nightly CB, DQ0 | 971.6 | 93.2 | failed | PagedAttention graph still incomplete |
| July 23 stock CB, DQ128 | 1,048.4 | 95.6 | passed | upstream Gemma-4 PA and token layout fixes |
| Fork, one scheduler pass | 4,143.5 | 94.6 | passed | `max_num_batched_tokens=8192` |
| Fork, grouped MoE and N128 | 4,448.5 | 95.1 | passed | grouped expert prefill and tile selection |
| Fork, 512-head Xe2 micro-SDPA | **5,827.2** | **112.2 short** | passed | global attention moved onto the XMX route |

The final PP result is 31.0% above the previous accepted fork result and 5.56x
the coherent July 23 stock CB result. These are full-model gains, not isolated
kernel TOPS.

## Why the last patch matters

Gemma-4 26B has five global-attention layers with a head size of 512. OpenVINO
already contained Xe2 micro-SDPA tuning records for that size, but the route
eligibility check rejected every head size above 256. Commit `2c82358676`
raises that limit to 512.

| Profiled device time | Old route | 512-head XMX route | Change |
| --- | ---: | ---: | ---: |
| Five global PA layers | 280.920 ms | 67.546 ms | 4.16x faster |
| 25 local PA layers | 55.629 ms | 54.039 ms | no material change |
| Total paged attention | 336.549 ms | 121.585 ms | 63.9% lower |
| Full profiled GPU inference | 963.435 ms | 752.369 ms | 21.9% lower |

The model weights, tokenizer and generation settings did not change.

## Context curve

Release plugin, DQ128, second request at each shape:

| Input tokens | PP tok/s |
| ---: | ---: |
| 966 | 5,158.9 |
| 2,048 | 6,359.5 |
| 4,096 | **6,500.2** |
| 6,622 | 5,869.5 |

The older pre-512-head build passed the same coherence gate at 16K with a
three-run mean of 2,424.0 PP tok/s and 78.8 decode tok/s. It required
`max_num_batched_tokens=16384` so the prompt stayed in one scheduler pass.
The 512-head patch has not yet been rerun at 16K.

Do not use an 8,192 scheduler limit to split a 16K prefill. That experiment
triggered an Xe GPU fault. Continuous batching at 30K is not validated either;
one shape exceeded the GPU maximum allocation size and the smaller-chunk retry
faulted. The model's older single-stream RoPE-LUT path remains separately
verified at 32K, but that is not the same runtime path as this benchmark.

## Public comparison

The public B70 benchmark repository from PMZFX reports the same Gemma-4
26B-A4B class at 1,129 pp512 and 52.6 tg128 with llama.cpp SYCL. Our nearest
short prefill point is 5,158.9 tok/s at 966 tokens, so it is not a shape-matched
PP comparison. Decode is directly useful: 112.2 tok/s is 2.13x the public B70
result.

Public single-RTX-3090 results for this model are about 129 to 131 tok/s with
llama.cpp and n-gram speculative decoding. The current B70 result is 112.2
tok/s without speculative decoding, which puts it in RTX 3090 decode territory.
An RTX 5090 result reports about 228 tok/s without DFlash and 578 tok/s with
DFlash. The B70 is not at 5090 decode performance.

Sources:

- https://github.com/PMZFX/intel-arc-pro-b70-benchmarks
- https://github.com/tfriedel/qwen3.6-rtx3090-lab/blob/main/GEMMA_FINDINGS.md
- https://www.reddit.com/r/LocalLLaMA/comments/1t796qe/gemma_4_26b_hits_600_toks_on_one_rtx_5090/
- https://www.intel.com/content/www/us/en/ark/products/series/242616/intel-arc-pro-b-series-graphics.html

Intel rates the B70 at 367 peak dense INT8 TOPS and 608 GB/s. Those hardware
figures are not model throughput and should not be compared directly with
tokens per second.

## Reproduce

The repository ships the benchmark and coherence gate as
`tests/bench_gemma4_a4w4.py`. A representative 6.6K run is:

```bash
python tests/bench_gemma4_a4w4.py \
  --dir /path/to/gemma4-26b-heretic-ov \
  --device GPU.0 \
  --dqgs 128 \
  --cb-cache-gb 8 \
  --prefix-cache \
  --max-batched-tokens 8192 \
  --moe-grouped-prefill on \
  --targets 6622 \
  --runs 3 \
  --decode-tokens 128 \
  --coherence-target 6622 \
  --coherence-tokens 384 \
  --no-coherence-warm
```

Confirm the selected `GPU.N` PCI address before running on a multi-GPU host.
Run on a cleared card, keep other workloads off that device and report cold
and sustained requests separately.

## Build

```bash
git clone --branch arc-xe2-gemma4-pa-2026.4 \
  https://github.com/Wondernuttz/openvino.git openvino-fork

cmake -S openvino-fork -B openvino-fork/build-gemma4-pa \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_INTEL_GPU=ON \
  -DENABLE_INTEL_CPU=OFF \
  -DENABLE_INTEL_NPU=OFF \
  -DENABLE_PYTHON=OFF \
  -DENABLE_PYTHON_API=OFF \
  -DENABLE_TESTS=OFF \
  -DENABLE_SAMPLES=OFF \
  -DENABLE_ONEDNN_FOR_GPU=ON

cmake --build openvino-fork/build-gemma4-pa \
  --target openvino_intel_gpu_plugin -j4
```

Use this plugin only with the matching OpenVINO 2026.4 ABI
(`libopenvino.so.2640`). A plugin built from this branch is not compatible with
an OpenVINO 2026.2 runtime. The optimized branch has been tested on Linux only.
The model's basic stock `VLMPipeline` path may run on Windows, but that path is
still untested here and does not carry these benchmark claims.
