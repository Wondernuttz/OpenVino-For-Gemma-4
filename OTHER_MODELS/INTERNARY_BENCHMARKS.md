# INTERNARY Benchmarks

Measured 2026-07-20 on one cleared Intel Arc Pro B70 with 32 GB.

## Software

- Ubuntu 24.04.4 LTS
- Linux kernel 7.0.0, `xe` driver
- Intel compute runtime `26.22.38646.6`
- OpenVINO `2026.2.0`
- OpenVINO GenAI `2026.2.0.0`
- OpenVINO commit `6339785dcd5a4356a01e8647afdac5a994363af7`
- Plugin SHA-256 `fbf51b2fd61df7d3e6229b787ef97963164ae45da7e9509fe629fde2c1829cc3`

## Method

The benchmark card was cleared to 0.15 percent memory before model load. Prefix
caching was disabled. Dynamic quantization group size was zero. Each shape was
warmed before measurement. Generation used fixed token counts with EOS ignored
so early stopping could not inflate decode throughput.

Prompt throughput is input tokens divided by mean time to first token. Decode
throughput comes from OpenVINO GenAI performance metrics. The 966-token shape was
run three times. Other published shapes were run once after warmup.

## End-to-End OpenVINO GenAI

| Input tokens | Repeat | Prompt tok/s | Decode tok/s | TTFT ms | TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 927.002 | 22.572 | 138.080 | 44.304 |
| 512 | 1 | 1,376.576 | 22.539 | 371.937 | 44.368 |
| 966 | 1 | 564.525 | 22.550 | 1,711.174 | 44.346 |
| 966 | 2 | 559.140 | 22.512 | 1,727.652 | 44.420 |
| 966 | 3 | 555.869 | 22.452 | 1,737.819 | 44.539 |
| 2,048 | 1 | 1,476.592 | 22.288 | 1,386.977 | 44.867 |
| 4,096 | 1 | 1,471.294 | 22.167 | 2,783.945 | 45.113 |
| 6,144 | 1 | 1,450.742 | 22.030 | 4,235.075 | 45.392 |

The 966-token shape cliff is real and unresolved. It is not hidden by reporting
only the faster long-context shapes.

## Long Context

The 6,748-token retrieval case used a 384-token generation cap:

| Input tokens | Prompt tok/s | Decode tok/s | TTFT ms | TPOT ms |
| ---: | ---: | ---: | ---: | ---: |
| 6,748 | 1,169.041 | 26.909 | 5,772.254 | 37.162 |

The model retrieved all three hidden facts. The raw harness did not get a clean
final answer because visible reasoning consumed the output limit. This is a
behavior failure, not a retrieval pass disguised as a complete coherence pass.

## INTBIT Comparison

The comparison uses the published projection-group INTBIT model on the same B70
and software stack.

| Input tokens | INTBIT prompt tok/s | INTERNARY prompt tok/s | INTBIT decode tok/s | INTERNARY decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,327.646 | 1,476.592 | 33.349 | 22.288 |
| 4,096 | 1,323.928 | 1,471.294 | 32.959 | 22.167 |
| 6,144 | 1,305.706 | 1,450.742 | 32.762 | 22.030 |

INTERNARY prefill is roughly 10 to 11 percent faster at these shapes. INTBIT
decode is roughly 49 percent faster. Ternary's W2A8 prefill maps directly to the
native signed-u2 instruction. Ternary decode moves twice the packed weight data
of u1 and currently uses an untuned W2A16 kernel.

## Native Signed-u2 Operator

The current operator winner is `gemm_u2_i8_m32n2_v2.cl` with an M32 by N32 tile,
256 GRF, packed A8 activations, and native `dpas.s2.s8`.

Three clean-card runs measured:

- Wide-N shape: 230.9 to 234.3 dense-equivalent TOPS
- Big-K shape: 214.0 to 214.2 dense-equivalent TOPS
- Square shape: 242.7 to 243.4 dense-equivalent TOPS

A separate full-size random `{-1,0,+1}` run measured 234.6, 219.7, and 244.3
dense-equivalent TOPS and passed every checked output. Zero codes were included.

Dense-equivalent TOPS is an operation-count comparison for the isolated GEMM.
It is not physical FP TFLOPS and it excludes activation packing, graph dispatch,
attention, KV-cache traffic, and every other model operation.

## Correctness

| Path | Rows | NRMSE | Cosine | Max absolute error ratio |
| --- | ---: | ---: | ---: | ---: |
| Decode W2A16 | 1 | 0.000199850 | 0.999999981 | 0.000260026 |
| Prefill W2A8 | 33 | 0.000198743 | 0.999999980 | 0.000287229 |

Two independent model loads produced the same fixed 128-token greedy sequence.
Sequence SHA-256:
`471f9237a017624ddfb36c3a007e11d49f746651a3ccec23e22e719a379b2241`.

## Skyrim Workload

The 18-case production workload had a 4.996-second median and 8.265-second
maximum. Its median was 1.019 times the corrected INTBIT workload.

It had no reasoning leak, AI-identity leak, Hangul fragments, repetition
failure, or empty response. It failed six likely-truncation checks and one exact
memory-recall check. The release reports that failure instead of calling the
workload a pass.

## Reproduce

Stop other models on the benchmark GPU first. Confirm the intended card is
clear, then run:

```bash
python runtime/benchmark.py \
  --model . \
  --device GPU.1 \
  --contexts 128,512,966,2048,4096,6144 \
  --decode-tokens 64
```

Do not compare a clean result against a card holding another model. Earlier
experiments proved that an idle resident can cut prompt processing badly without
obvious compute utilization.
