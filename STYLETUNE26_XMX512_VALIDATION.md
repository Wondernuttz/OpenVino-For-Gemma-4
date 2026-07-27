# Gemma-4 26B-A4B StyleTune V2, Xe2 XMX validation

Date: 2026-07-26

## Result

The optimized Gemma-4 26B-A4B runtime transfers cleanly to the local StyleTune
V2 build. The model uses the same OpenVINO 2026.4 fork, 512-head micro-SDPA
route, grouped MoE path and DQ128 deployment settings as the Heretic build.

Tested model:

`/home/wondernutts/models/heretics/Gemma-4-26B-A4B-StyleTune-V2-int4-ov-lut131k`

GPU plugin SHA-256:

`5c2085a1f7a8edc86d782d8da805ffd98e47c5035dffb6722b8e0e0945a1959c`

## Fast 8K profile

Settings: DQ128, 8 GB prefix cache, grouped MoE, N128 and
`max_num_batched_tokens=8192`.

| Input tokens | Second PP result |
| ---: | ---: |
| 966 | 5,196.6 tok/s |
| 2,048 | 6,350.0 tok/s |
| 4,096 | 6,512.3 tok/s |
| 6,622 | 5,827.2 tok/s |

Short-context decode was 112.223 tok/s. Decode after the 6,622-token
coherence prompt was 94.855 tok/s.

The coherence gate passed 4/4: three exact retrieval checks plus the requested
style continuation. Output SHA-256:

`0274e854b8679e24e6a6f1ee3687ff1d02d59d1803f9f8270e30a5a2fe66a5fb`

## 16K-safe serving profile

With `max_num_batched_tokens=16384`, the second 6,622-token run measured
5,774.6 PP tok/s and short-context decode measured 112.55 tok/s. The larger
scheduler therefore retains 99.1% of the fast profile's sustained PP rate while
keeping the separately validated 16K scheduling rule.

The deployed local profile is:

```text
venv=gemma4pa
max_ctx=16000
OV_PREFIX_CACHE=8
OV_DQGS=128
OV_MAX_BATCHED_TOKENS=16384
MOE_USE_GROUPED_GEMM_PREFILL=1
MOE_MICRO_GEMM_N_HINT=128
```

The optimized continuous-batching path is not enabled beyond 16K. The 30K
allocation and Xe fault boundaries from the Heretic build still apply.

## Live state

After deployment, the model hub was restarted with persistent boot defaults:

- Discord card, GPU.1: Gemma-4 26B Heretic on the optimized profile
- Buddy card, GPU.2: Qwen3.6-27B on its validated fallback profile

Both endpoints passed exact post-restart smoke checks. The benchmark lock was
cleared after the buddy model was restored.
