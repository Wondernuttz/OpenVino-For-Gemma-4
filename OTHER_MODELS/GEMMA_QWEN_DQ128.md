# Gemma and Qwen INT4 DQGS sweep

Date: 2026-07-26
Device: one Intel Arc Pro B70, `GPU.2`
GPU.1 remained live on the Discord Qwen 35B MoE model.

## Result

`DYNAMIC_QUANTIZATION_GROUP_SIZE=128` is deployed for Gemma 12B, Gemma 31B,
and dense Qwen 27B. It is rejected for Qwen 35B MoE. Gemma 26B MoE benefits in
the plain pipeline but is not deployed because its continuous-batching path
fails before scoring.

This sweep uses the existing OpenVINO Intel GPU fully-connected path. It is a
runtime configuration win, not a new custom kernel.

## Gemma 12B dense

OpenVINO GenAI 2026.3 nightly, plain `VLMPipeline`.

| Input | DQGS 0 | DQGS 128 | Gain |
|---:|---:|---:|---:|
| 512 | 2,117.597 tok/s | 2,808.161 tok/s | +32.6% |
| 2,048 | 2,956.760 tok/s | 3,834.672 tok/s | +29.7% |
| 6,144 | 2,406.055 tok/s | 2,956.049 tok/s | +22.9% |
| 6,620 coherence | 2,310.170 tok/s | 2,754.378 tok/s | +19.2% |
| 30,000 coherence | 805.238 tok/s | 851.101 tok/s | +5.7% |

Short decode remained 56.3 versus 56.1 tok/s. The 6,620 and 30,000-token
coherence runs passed all four checks and produced byte-identical answers.
DQGS 128 is valid through the server's 30K cap. The separate vision and audio
paths were not part of this sweep and remain on DQGS 0.

## Gemma 31B dense

OpenVINO GenAI 2026.2, plain `VLMPipeline`.

| Input | DQGS 0 | DQGS 128 | Gain |
|---:|---:|---:|---:|
| 512 | 1,079.422 tok/s | 1,573.667 tok/s | +45.8% |
| 2,048 | 1,264.324 tok/s | 1,655.586 tok/s | +31.0% |
| 6,144 | 986.617 tok/s | 1,191.225 tok/s | +20.7% |
| 6,620 coherence | 945.374 tok/s | 1,123.329 tok/s | +18.8% |

Decode remained 27.5 tok/s short and 19.45 tok/s on the coherence run. Both
settings passed all four checks and produced the exact same 110-token answer.
An exact 16K DQGS 0 run crossed the card's VRAM cliff and did not finish within
ten minutes. It was stopped and discarded. The 16K server cap remains a
capacity ceiling, not a performance claim.

## Qwen 27B dense

OpenVINO GenAI 2026.2.

| Input | DQGS 0 | DQGS 128 | Gain |
|---:|---:|---:|---:|
| 512 | 834.084 tok/s | 1,381.764 tok/s | +65.7% |
| 2,048 | 1,399.606 tok/s | 1,897.784 tok/s | +35.6% |
| 6,144 | 1,554.934 tok/s | 2,018.113 tok/s | +29.8% |
| 6,741 coherence | 1,494.039 tok/s | 1,977.841 tok/s | +32.4% |

Short decode remained about 32 tok/s. Both coherence answers passed all four
checks. The only output difference was the invented date in Entry 121.

Plain DQGS 128 passed the 24K coherence gate at 1,561.948 tok/s PP and 24.232
tok/s decode. Prefix-cached serving has a separate fixed-KV limit:

| Scheduler | Result |
|---|---|
| Prefix cache 8 GB, 12K | PASS, all checks |
| Prefix cache 8 GB, 13K | Truncated after item 2 |
| Prefix cache 8 GB, 14K and above | Zero generated tokens |
| Prefix cache 12 GB, 16K | PASS, 1,327.549 PP tok/s, 29.090 decode tok/s |
| Prefix cache 12 GB, 24K | Zero generated tokens |

Deployment is therefore DQGS 128, 12 GB prefix cache, and a 16K served cap.
The plain non-scheduler pipeline remains valid through 24K.

The production `ovhub` load path was then exercised on the empty buddy slot.
Qwen 27B loaded on GPU.2 in 20.4 seconds and logged:

```text
DYNAMIC_QUANTIZATION_GROUP_SIZE=128
prefix caching ON (cache 12GB, max batch 256, CB backend)
max_ctx=16000 tok
```

Both `/health` and `/v1/models` returned HTTP 200. The server chat test was
asked to return `QWEN27 READY` exactly and did so.

## Rejected or blocked

Qwen 35B MoE at 2K dropped from 3,332.286 to 1,945.458 PP tok/s with DQGS 128.
Decode also dropped from 104.993 to 98.310 tok/s. It remains on DQGS 0.

Gemma 26B MoE gained 47.4% at 512, 27.6% at 2K, 3.5% at 6K, and 5.9% on
the 6,620-token coherence prompt. The answer was byte-identical. Its
continuous-batching path failed at both DQGS settings with the OpenVINO GenAI
GPU USM buffer-copy size error, so no production change was made.

## Deployment

Server registry SHA256:
`917f3fe20f5a41c5dfaf19726f59e3b1e15c2bad85a28d5c1e366dffe5855a7e`

`ovhub` caches the registry until restart. To avoid dropping the live Discord
process, the three winners were also written to `runtime_overrides.json`.
`ovserver_moe.py` now reads `OV_DQGS` through the same override path already
used for prefix-cache size and context limits. This makes the settings active
on the next model load without restarting GPU.1.

Hugging Face README commits:

| Model | Commit |
|---|---|
| Gemma 12B | `24a57ba782db8b159752914fe97a24d757817967` |
| Gemma 31B | `0044be63c02d7db9f85051a6f4d21c009d18c12a` |
| Qwen 27B | `7c352592e8e8b67c6ec4f82a315603a42b0c9923` |

All three remote README files were downloaded after publication and matched
the local files byte for byte.

## Next kernel work

1. Fix the OpenVINO GenAI continuous-batching GPU buffer-copy failure before
   deploying the Gemma 26B DQGS gain.
2. Keep Qwen 35B on its existing grouped-MoE winners. DQGS is the wrong path
   for that graph.
3. Tune the dense fully-connected selector around the DQGS 128 baseline.
4. Implement native A4W4 inside the fused Gemma MoE primitive while preserving
   gather, SwiGLU, down projection, scatter, and reduction.
