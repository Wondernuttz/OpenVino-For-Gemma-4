# Xe2 MoE Micro-GEMM N Selection

Date: 2026-07-25

Device: Arc Pro B70 GPU.2

Commit: `6306353b tune-MoE-micro-GEMM-selection-N`

Deployed plugin SHA-256:

`5dc9dacfad9c300c5e27643ac8bba18ffdd4fbca82fd422933a68d000904a696`

## Change

OpenVINO selected every prefill MoE microkernel with a fixed problem hint of
`N=32`. The runtime kernel supports variable token counts, but the selection N
changes its generated tile package.

The fork now defaults to `N=256` for this Xe2 prefill path and accepts
`MOE_MICRO_GEMM_N_HINT` values from 8 through 1024 for controlled races. The
selection N is part of the package cache key so differently tuned pipelines
cannot reuse the wrong package in one process.

No graph, weight, quantization, or model arithmetic changed.

## Qwen 35B Race

Fresh process, prefix cache disabled, 2,048 input tokens:

| N hint | PP tok/s |
|---:|---:|
| 8 | 3,333.401 |
| 16 | 3,303.678 |
| 32 | 3,326.130, 3,270.444, 3,265.605 |
| 64 | 3,333.289 |
| 128 | 3,301.656 |
| 256 | 3,373.040, 3,403.105, 3,334.310 |
| 512 | 3,358.371 |
| 1024 | 3,273.316 |

The three-run means were 3,287.393 for N32 and 3,370.152 for N256. N256 was
2.52 percent faster.

Fresh process, exact buddy configuration with prefix cache 6, 2,048 cold input
tokens:

| N hint | Run 1 | Run 2 | Run 3 | Mean |
|---:|---:|---:|---:|---:|
| 32 | 1,844.409 | 1,850.444 | 1,853.740 | 1,849.531 |
| 256 | 2,014.462 | 2,018.621 | 2,015.334 | 2,016.139 |

N256 was 9.01 percent faster. The final plugin, with no explicit hint in the
process environment, returned 2,018.087 tok/s and confirmed the new default.

At 512 tokens, after discarding the first N32 package-compilation outlier:

| N hint | Measurements | Mean |
|---:|---:|---:|
| 32 | 1,645.308, 1,665.840 | 1,655.574 |
| 256 | 1,716.586, 1,721.959, 1,693.679 | 1,710.741 |

N256 was 3.33 percent faster.

At 6,144 tokens, the compiled N32 control returned 3,994.669 tok/s. N256
returned 4,023.170 and 4,023.783 tok/s, about 0.72 percent faster. The initial
3,033.065 N32 result included first-package compilation and is discarded.

## Qwen Coherence Gate

Both candidates received the same 6,739-token chronicle with prefix cache 6.

| N hint | PP tok/s | Decode tok/s | Exact checks |
|---:|---:|---:|---:|
| 32 | 1,948.987 | 102.571 | 4 of 4 |
| 256 | 2,180.850 | 107.159 | 4 of 4 |

N256 improved matched coherence prompt processing by 11.90 percent. Both
outputs recovered the third barrel behind the Bannered Mare, all four Vigor of
the Nine ingredients, `shadow-hearth`, and the requested Entry 121 ending.

The separate N256 short-decode run returned 104.286 tok/s. This selector change
targets prefill, so no decode gain is claimed.

## Gemma 26B Check

Prefix cache disabled, one fresh-process result per shape:

| Input tokens | N32 | N256 | Change |
|---:|---:|---:|---:|
| 512 | 1,562.886 | 1,595.435 | +2.08% |
| 2,048 | 3,200.274 | 3,233.750 | +1.05% |
| 6,144 | 3,648.720 | 3,668.823 | +0.55% |

The N256 Gemma coherence run used 6,620 input tokens, returned 3,352.466 PP
tok/s, and passed all four exact checks.

## Rejected Selector Modes

- `slmPtr=false`: OpenCL build failure.
- `kParallelLocal=true`: OpenCL build failure.
- `localA=true`: selector returned no matching kernel.
- `localB=true`: selector returned no matching kernel.

These controls were removed from the final source. Do not re-walk them without
first changing the kernel interface or Gemmstone catalog.

## Deployment

The patched plugin is installed in the `ov-genai` environment. Qwen is live on
GPU.2 with prefix cache 6 and micro-GEMM prefill enabled. The registry records
`MOE_MICRO_GEMM_N_HINT=256` for Qwen35 and Gemma26. The plugin default also uses
256, so the running buddy service does not require a hub restart.

GPU.1 was not stopped or restarted during this tuning pass.
