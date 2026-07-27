# Qwen Prefix-Cache Scheduler Results

Date: 2026-07-25

Device: Arc Pro B70 GPU.2

Model: Qwen3.6 35B A3B INT4 OpenVINO

## Root Cause

OpenVINO GenAI's `SchedulerConfig` defaults `max_num_batched_tokens` to 256.
The prefix-cache backend therefore split a 2K prompt into at least eight
prefill scheduler chunks. The plain pipeline did not pay this chunking cost.

This setting is a scheduler chunk ceiling, not a model context limit. A 32K
prompt with a 4,096-token ceiling is processed in roughly eight chunks.

## 2K Scheduler Race

All tests used prefix caching, the N256 Xe2 micro-GEMM selector, and a fresh
process. First-package shape compilation was discarded where noted.

| Max batched tokens | PP tok/s |
|---:|---:|
| 256 | 2,016.139 three-run mean |
| 512 | 2,888.122 |
| 1,024 | 3,180.533 |
| 2,048 | 3,404.353 |
| 4,096 | 3,460.753 |
| 8,192 | 3,389.941 |

The first 1,024-token package launch returned 1,765.463 tok/s and is discarded
as one-time compilation. The compiled repeat is shown above.

With the final 9 GB cache, the 4,096-token setting returned 3,425.052 tok/s and
106.355 decode tok/s. Reducing `max_num_seqs` from 256 to 16 returned 3,423.566
tok/s, so the sequence limit was left unchanged for concurrency.

Compared with the old cached path at 2,016.139 tok/s, the final ordinary 2K
result is 69.88 percent faster. It also slightly exceeds the approximately
3,370 tok/s plain-pipeline mean while retaining prefix caching.

## Context Scaling

Prefix cache 6 GB, max batched tokens 4,096:

| Input tokens | PP tok/s | Result |
|---:|---:|---|
| 6,144 | 4,110.720 | valid |
| 16,384 | 3,934.112 | valid |
| 20,000 | 3,377.343 | valid |
| 24,000 | n/a | no output token, cache capacity exhausted |

The first 6K launch returned 3,070.786 tok/s while compiling the large shape
package. The compiled repeat is shown above.

Cache capacity tests with max batched tokens 4,096:

| Cache | Input tokens | PP tok/s | Result |
|---:|---:|---:|---|
| 6 GB | 28,672 | n/a | no output token |
| 6 GB | 30,000 | n/a | no output token |
| 8 GB | 28,672 | 3,323.542 | valid |
| 8 GB | 30,000 | 2,957.461 | valid |
| 8 GB | 32,000 | n/a | no output token |
| 9 GB | 32,000 | 2,962.500 | valid |
| 9 GB | 32,768 | 3,129.927 | valid |

The model advertises a 262K architectural limit. The failures above are KV
cache capacity limits for this one-card service, not model position limits.

## 31K Coherence Gate

The chronicle was padded to exactly 31,000 input tokens while keeping the three
hidden facts near their original early and middle positions. Configuration:

- Prefix cache: 9 GB
- Max batched tokens: 4,096
- Xe2 micro-GEMM selection N: 256

Result:

- Prompt processing: 2,951.568 tok/s
- Decode: 83.171 tok/s
- Output tokens: 152
- Exact checks: 4 of 4

The model recovered the third barrel behind the Bannered Mare, all four Vigor
of the Nine ingredients, `shadow-hearth`, and the requested Entry 121 style and
ending.

## Deployment

The live Qwen buddy service now uses:

```text
OV_PREFIX_CACHE=9
OV_MAX_BATCHED_TOKENS=4096
OV_MAX_CTX_TOKENS=32768
MOE_USE_GROUPED_GEMM_PREFILL=0
MOE_MICRO_GEMM_N_HINT=256
```

The hub registry contains the canonical settings. `ovserver_moe.py` also reads
`ovhub/runtime_overrides.json` at launch so this model could be updated without
restarting the hub and killing its GPU.1 child cgroup.

The server treats 32,768 as a combined input and requested-output budget. It
reserves each request's `max_tokens` before dropping old history, preventing a
full-length prompt from consuming the entire KV cache before generation.

GPU.1 was not stopped or restarted during this pass.
