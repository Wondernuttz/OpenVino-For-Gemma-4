# Full Model Results on Arc Pro B70

Date: 2026-07-25

Device: GPU.2, clean benchmark card unless stated otherwise.

## Production Finding

The OpenVINO fork defaults `MOE_USE_GROUPED_GEMM_PREFILL` to enabled. On these
Xe2 MoE models that prevents the faster micro-GEMM prefill path from running.
Setting `MOE_USE_GROUPED_GEMM_PREFILL=0` produced the only deployable full-model
gain from this session. No model files or weights changed.

Gemma 4 26B, stock INT4 model, prefix cache disabled, fresh process per shape:

| Test | Grouped control | Micro-GEMM | Change |
|---|---:|---:|---:|
| 512-token prompt processing | 1,443.245 tok/s | 1,564.566 tok/s | +8.4% |
| 2,048-token prompt processing | about 1,619 tok/s | 3,203.470 tok/s | about +98% |
| 6,144-token prompt processing | about 2,247 tok/s | 3,653.655 tok/s | +62.6% |
| Decode | about 99 tok/s on the HF card | 103.794 tok/s | about +4.8% |

The existing HF figures near 2,900 tok/s at 512 and 4,700 tok/s at 2K were not
reproduced by this fresh-process harness and should not be used as current
controls. The old 6K figure near 3,200 tok/s was exceeded by 14 percent.

Qwen3.5 35B, buddy model:

| Test | Grouped control | Micro-GEMM | Change |
|---|---:|---:|---:|
| 2K PP, prefix cache disabled | 2,645.983 tok/s | 3,275.818 tok/s | +23.8% |
| 2K cold PP, prefix cache 6 | 1,288.383 tok/s | 1,840.674 tok/s | +42.9% |
| Decode, prefix cache 6 | not rerun | 106.081 tok/s | n/a |

The buddy service now carries `MOE_USE_GROUPED_GEMM_PREFILL=0` and
`OV_PREFIX_CACHE=6`. The Gemma registry entry carries the same prefill setting
and `OV_PREFIX_CACHE=8` for its next launch.

Follow-up on the same date changed the Xe2 microkernel selection hint from N32
to N256. The exact Qwen prefix-cache 6 result increased again from a three-run
mean of 1,849.531 to 2,016.139 tok/s. See
`MICRO_GEMM_N_HINT_RESULTS.md` for the complete race and coherence gate.

The cached path's remaining gap came from the scheduler default of only 256
batched prefill tokens. The deployed 4,096-token ceiling and 9 GB cache reached
3,425.052 tok/s at 2K and passed a 31,000-token coherence run at 2,951.568 PP
tok/s. See `PREFIX_CACHE_SCHEDULER_RESULTS.md`.

## Coherence

Gemma received 6,620 input tokens and returned 3,341.120 PP tok/s. All four
deterministic Skyrim checks passed, including the exact requested ending.

Qwen received 6,739 input tokens with prefix cache 6. It returned 1,695.884 PP
tok/s and 103.681 decode tok/s. All four deterministic checks passed.

These are bounded coherence checks, not a general model-quality evaluation.

## Experimental A4W4 Integration

The new A4W4 path preserves the original asymmetric U4 weights. It uses signed
activation codes `q-8`, group-64 scales, and the correction
`(8-zp)*sum(Aq)`. The production output type is FP16.

Verified operator results:

- Random numerical oracle passed.
- Source U4 decode matched OpenVINO CPU on 256 samples.
- Projection cosine was about 0.999968 against the independent quantized
  reference.
- Fused quantize, GEMM, and FP32 output reached about 39.12 TOPS at M32,
  61.78 TOPS at M128, and 63.86 TOPS at M384.
- GEMM alone reached 105.68 TOPS at M384.
- A real layer-0, M512 gate/up subgraph fell from about 5.256 ms to 2.803 ms,
  a 46.7 percent reduction. Cosine against the original FP16 activation path
  was 0.994257 and 0.994243 for the two projections.

The full custom model reached only about 488 PP tok/s at 512 tokens, versus
1,443 tok/s for the stock grouped path and 1,565 tok/s for micro-GEMM. Replacing
the gate/up MatMuls prevents OpenVINO from forming `MOE3GemmFusedCompressed`.
Routing, gather, gate/up, activation, down projection, scatter, and reduction
then become separate work. The faster local operator loses badly at model
level.

The custom graph is experimental and must not be deployed. It remains private
at:

`/home/wondernutts/models/heretics/gemma-4-26B-A4B-heretic-int4-a4w4-gateup-ov`

The plugin experiment is committed as `4585e5f2` on branch
`gemma-a4w4-moe-2026.2` in:

`/home/wondernutts/openvino-gemma-a4w4`

## Next Kernel Target

The next implementation belongs inside the existing fused MoE primitive, not
in another graph surgery. Patch the gate/up generators in:

`src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp`

The relevant stages are `MoE3GemmSwigluMLPGateUp` and the prefill
`MoE3GemmMicroGenerator` gate/up paths. A useful implementation must retain the
current gather, SwiGLU, down projection, scatter, and reduction fusion while
adding A4 activation packing and native low-low DPAS.

## Known Harness Defect

Repeated same-shape requests can fail with `CL_OUT_OF_RESOURCES` in both stock
and custom pipelines. Shape-warm measurements are invalid until that separate
pipeline or driver issue is fixed. Each context result above came from a fresh
process.
