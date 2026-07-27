# INTBIT OpenVINO Benchmarks

Date: 2026-07-19

## Test System

- One Intel Arc Pro B70, 32 GB, PCI ID `8086:e223`
- 2,800 MHz reported core clock
- Ubuntu 24.04.4 LTS
- Linux 7.0.0, `xe` kernel driver
- Intel compute runtime `26.22.38646.6`
- OpenVINO `2026.2.0`, base commit
  `52ddc07385712456dd9f8c5ecf05d7e49c6da329`
- INTBIT implementation commit
  `be1b2f3f50947aa9a287833ee0851b2cfcca18b5`
- OpenVINO GenAI `2026.2.0.0`
- Patched Intel GPU plugin SHA-256
  `3557b0d03d523effe7e871a0a9e05e34e00797714b74499cce6693c5363b09fe`

## Measurement Rules

- The measured card was cleared to about 51 MiB before model load.
- No second model was resident on the card.
- Prefix caching was disabled.
- Kernel and graph compilation were excluded from timed measurements.
- Every prompt shape was warmed before measurement.
- Generation was greedy with EOS ignored and a fixed output-token count.
- Prompt throughput is input tokens divided by GenAI TTFT.
- Decode throughput is OpenVINO GenAI's generated-token throughput.
- The 966-token release point uses three same-pipeline repetitions.

Run the included harness only on a cleared card:

```bash
python runtime/benchmark.py \
  --model ./Bonsai-27B-INTBIT-OpenVINO-Arc \
  --device GPU.0 \
  --contexts 128,512,966,2048,4096,6144 \
  --decode-tokens 64
```

## Release Point: 966 Input Tokens

| Run | Prompt tok/s | TTFT ms | Decode tok/s | TPOT ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 551.504 | 1,751.575 | 33.980 | 29.429 |
| 2 | 542.409 | 1,780.942 | 33.932 | 29.470 |
| 3 | 545.018 | 1,772.418 | 34.029 | 29.387 |

The honest published range is therefore:

- Prompt processing: `542.4-551.5 tok/s`
- Decode: `33.93-34.03 tok/s`
- TPOT: `29.39-29.47 ms`

## Context Sweep

| Input tokens | Prompt tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 128 | 841.820 | 33.924 |
| 512 | 1,248.653 | 33.893 |
| 966 | 543.193-551.284 | 33.750-33.967 |
| 2,048 | 1,327.646 | 33.349 |
| 4,096 | 1,323.928 | 32.959 |
| 6,144 | 1,305.706 | 32.762 |

The context sweep and the dedicated three-run 966 release point were separate
runs, which is why their 966 ranges differ slightly.

## Previous OpenVINO Graph

At the same 966-token shape, the previous W1A16 graph measured:

- Prompt processing: `392.4-401.4 tok/s`
- Decode: `32.88-32.96 tok/s`

The projection-group INTBIT graph is about 37 percent faster in prompt processing
and about 3 percent faster in decode at equal shape.

This is an internal same-model OpenVINO comparison. It is not a comparison with
Prism ML's llama.cpp, CUDA, or Metal results.

## Operator Benchmark

The native packed W1A8 kernel uses one-bit resident weights, A8 activations, and
Xe2 `dpas.s2.s8`. Three clean-card runs across each primary shape measured:

| Shape class | Dense-equivalent TOPS |
| --- | ---: |
| Wide N | 187.9-188.7 |
| Big K | 182.0-183.7 |
| Square | 198.2-198.7 |

The full-size quantized oracle passed. Dense-equivalent TOPS counts the equivalent
dense multiply-accumulate work. It is not physical FP TFLOPS and excludes the
rest of the model.

## Optimization Status

These are the current measured results, not a hardware-ceiling claim. The
966-token shape remains an outlier compared with the 2,048 through 6,144-token
results. More shape-specific kernels and dispatch policies still need to be
tested. Results will be added only after clean-card end-to-end validation.

## Correctness Gates

- Four-projection micrograph outputs were bit-identical at rows 1, 33, and 960.
- Conservative and canonical graph contracts produced the same fixed greedy
  128-token output.
- Token-sequence SHA-256:
  `44ab59c0d8bba7a6b0ccac3f975117f8471b4e34c170c11fc552125c65a94b50`.
- A 6,748-token retrieval test recovered every hidden fact and the requested
  style derivation.

## Exclusions

The exact 1,024-token shape showed an abnormal lifecycle split. One request
measured 1,300.734 prompt tok/s while subsequent same-pipeline requests measured
790.113-790.339 prompt tok/s. Both values are excluded from release claims.

An earlier 966-token result measured while an 18 GB model was resident on the
same 32 GB card was discarded. It was a memory-contention result, not a valid
INTBIT measurement.

## Serving Stability

The original HTTP server entered one shared OpenVINO pipeline from a newly
created OS thread for every request. Identical requests degraded on every call.
Routing all generations through one persistent worker thread removed the
staircase. Six identical HTTP requests then held at 1.48-1.50 seconds after the
first request. The included server follows that contract.

## Model-Quality Gate

The graph passed fixed-token equivalence and long-context retrieval. A separate
18-case Skyrim persona workload returned all 18 responses without visible
reasoning, AI identity leakage, foreign fragments, or repetition loops. Under
that gate's fixed response caps it still produced five likely truncations and
failed exact recall in two cases. Those results are a release limitation, not a
kernel numerical mismatch.
