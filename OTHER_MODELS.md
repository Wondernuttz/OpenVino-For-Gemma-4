# Other Models on Intel Arc

This page indexes the other OpenVINO models tested alongside the Gemma-4 work in this repository.
All performance numbers were measured on one Intel Arc Pro B70. They are single-card results and
do not use speculative decoding. Each linked report states its runtime, cache configuration,
measurement method, and coherence result.

The model weights live on Hugging Face. This repository carries the benchmark record and the
OpenVINO fork history.

## Validated results

| Model | Precision | Prompt processing | Decode | Report |
|---|---:|---:|---:|---|
| Qwen3.6 35B-A3B Heretic | INT4 | 3,392.8 at 2K; 4,108.0 at 6K; 3,128.2 at 32K | 106 short; 83.9 after 31K | [benchmark](OTHER_MODELS/QWEN35_BENCHMARKS.md) |
| Qwen3.6 27B Heretic | INT4 | 1,381.8 at 512; 1,897.8 at 2K; 2,018.1 at 6K | about 32 short; 24.2 after 24K | [DQGS report](OTHER_MODELS/GEMMA_QWEN_DQ128.md#qwen-27b-dense) |
| Gemma-4 12B Heretic | INT4 | 2,808.2 at 512; 3,834.7 at 2K; 2,956.0 at 6K | 56.1 short; about 26 after 6K | [DQGS report](OTHER_MODELS/GEMMA_QWEN_DQ128.md#gemma-12b-dense) |
| Gemma-4 31B Heretic | INT4 | 1,573.7 at 512; 1,655.6 at 2K; 1,191.2 at 6K | 27.5 short; 19.45 after 6.6K | [DQGS report](OTHER_MODELS/GEMMA_QWEN_DQ128.md#gemma-31b-dense) |
| Gemma-4 26B-A4B Heretic | INT4 | 5,827.2 sustained at 6.6K | 112.2 short; 94.9 after 6.6K | [full history](BENCHMARK_HISTORY_GEMMA4_26B.md) |
| Gemma-4 26B-A4B StyleTune V2 | INT4 | 5,774.6 at 6.6K on the 16K-safe profile | 112.6 short; 94.9 after 6.6K | [validation](STYLETUNE26_XMX512_VALIDATION.md) |
| Bonsai 27B INTBIT | packed u1 | 542.4-551.5 at 966; 1,327.6 at 2K; 1,305.7 at 6K | 33.93-34.03 short | [benchmark](OTHER_MODELS/INTBIT_BENCHMARKS.md) |
| Bonsai 27B INTERNARY | packed signed u2 | 555.9-564.5 at 966; 1,476.6 at 2K; 1,450.7 at 6K | about 22.5 short | [benchmark](OTHER_MODELS/INTERNARY_BENCHMARKS.md) |
| Magnum v4 12B Anthracite | INT4 | 5,894 at 512; 6,268 at 2K; 5,134 at 6K | 72.37 short | [Magnum report](OTHER_MODELS/MAGNUM_DQ128.md) |
| Lumimaid-Magnum v4 12B | INT4 | 5,912 at 512; 6,267 at 2K; 5,146 at 6K | 71.99 short | [Magnum report](OTHER_MODELS/MAGNUM_DQ128.md) |
| Magnum Diamond 24B | INT4 | 3,062 at 512; 3,202 at 2K; 2,974 at 6K | 41.97 short | [Magnum report](OTHER_MODELS/MAGNUM_DQ128.md) |
| Lumimaid-Magnum v4 12B | INT8 | 7,138 at 512; 7,105 at 2K; 5,533 at 6K | 43.27 short | [Magnum report](OTHER_MODELS/MAGNUM_DQ128.md) |
| Magnum Diamond 24B | INT8 | 4,017 at 512; 3,755 at 2K; 3,225 at 6K | 22.19 short | [Magnum report](OTHER_MODELS/MAGNUM_DQ128.md) |

Prompt processing is shown in tokens per second. Context size is the number of input tokens.
Decode measurements at long context are labeled separately from short-context decode.

## Qwen 35B tuning history

The Qwen 35B result came from three separate changes. The reports are preserved separately so the
gain history is not flattened into one unexplained final number.

- [Full-model grouped versus micro-GEMM results](OTHER_MODELS/QWEN35_GEMMA26_TUNING_HISTORY.md)
- [Xe2 micro-GEMM N-hint race and coherence gate](OTHER_MODELS/QWEN35_MICRO_GEMM_HISTORY.md)
- [Prefix-cache scheduler race and 31K gate](OTHER_MODELS/QWEN35_PREFIX_CACHE_HISTORY.md)

## Models without a standalone claim

These models are available, but a separate publishable benchmark has not been earned yet.

| Model | Current status |
|---|---|
| Ornith 1.0 35B | Available on Hugging Face. No independent cleared-card benchmark and coherence report yet. |
| Gemma-4 31B StyleTune | Blocked. Repeated plain-pipeline requests faulted and continuous batching produced incoherent output. |
| Qwen3.6 35B StyleTune | Available locally. No independent cleared-card benchmark and coherence report yet. |
| Bonsai 27B INT4 donor | Unpacked conversion source for INTBIT. It is not the packed u1 release runtime. |
| Ternary Bonsai 27B INT4 donor | Unpacked conversion source for INTERNARY. It is not the packed signed-u2 release runtime. |

## Model downloads

| Model | Hugging Face |
|---|---|
| Qwen3.6 35B-A3B Heretic | [Wondernutts/tvall43-Qwen3.6-35B-A3B-heretic-int4-ov](https://huggingface.co/Wondernutts/tvall43-Qwen3.6-35B-A3B-heretic-int4-ov) |
| Ornith 1.0 35B | [Wondernutts/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16-int4-ov](https://huggingface.co/Wondernutts/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16-int4-ov) |
| Qwen3.6 27B Heretic | [Wondernutts/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-int4-ov](https://huggingface.co/Wondernutts/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-int4-ov) |
| Gemma-4 26B-A4B Heretic | [Wondernutts/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov](https://huggingface.co/Wondernutts/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) |
| Gemma-4 31B Heretic | [Wondernutts/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov](https://huggingface.co/Wondernutts/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) |
| Gemma-4 12B Heretic | [Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov) |
| Bonsai 27B INTBIT | [Wondernutts/Bonsai-27B-INTBIT-OpenVINO-Arc](https://huggingface.co/Wondernutts/Bonsai-27B-INTBIT-OpenVINO-Arc) |
| Bonsai 27B INTERNARY | [Wondernutts/Bonsai-27B-INTERNARY-OpenVINO-Arc](https://huggingface.co/Wondernutts/Bonsai-27B-INTERNARY-OpenVINO-Arc) |
| Lumimaid-Magnum v4 12B INT4 | [Wondernutts/Lumimaid-Magnum-v4-12B-int4-ov](https://huggingface.co/Wondernutts/Lumimaid-Magnum-v4-12B-int4-ov) |
| Gemma-4 31B StyleTune | [Wondernutts/Gemma-4-31B-StyleTune-int4-ov](https://huggingface.co/Wondernutts/Gemma-4-31B-StyleTune-int4-ov) |
| Gemma-4 26B-A4B StyleTune V2 | [Wondernutts/Gemma-4-26B-A4B-StyleTune-V2-int4-ov](https://huggingface.co/Wondernutts/Gemma-4-26B-A4B-StyleTune-V2-int4-ov) |
| Qwen3.6 35B StyleTune | [Wondernutts/Qwen3.6-35B-A3B-StyleTune-int4-ov](https://huggingface.co/Wondernutts/Qwen3.6-35B-A3B-StyleTune-int4-ov) |
| Bonsai 27B INT4 donor | [Wondernutts/Binary-Bonsai-27B-int4-sym-ov](https://huggingface.co/Wondernutts/Binary-Bonsai-27B-int4-sym-ov) |
| Ternary Bonsai 27B INT4 donor | [Wondernutts/Ternary-Bonsai-27B-int4-sym-ov](https://huggingface.co/Wondernutts/Ternary-Bonsai-27B-int4-sym-ov) |
| Magnum v4 12B Anthracite INT4 | [Wondernutts/magnum-v4-12b-int4-ov](https://huggingface.co/Wondernutts/magnum-v4-12b-int4-ov) |
| Magnum Diamond 24B INT4 | [Wondernutts/MS3.1-24B-Magnum-Diamond-int4-ov](https://huggingface.co/Wondernutts/MS3.1-24B-Magnum-Diamond-int4-ov) |
| Lumimaid-Magnum v4 12B INT8 | [Wondernutts/Lumimaid-Magnum-v4-12B-int8-ov](https://huggingface.co/Wondernutts/Lumimaid-Magnum-v4-12B-int8-ov) |
| Magnum Diamond 24B INT8 | [Wondernutts/MS3.1-24B-Magnum-Diamond-int8-ov](https://huggingface.co/Wondernutts/MS3.1-24B-Magnum-Diamond-int8-ov) |

## Forks used by these reports

- [Gemma-4 26B paged-attention and Xe2 XMX branch](https://github.com/Wondernuttz/openvino/tree/arc-xe2-gemma4-pa-2026.4)
- [Qwen 35B Xe2 INT4 branch](https://github.com/Wondernuttz/openvino/tree/arc-xe2-int4-2026.2)
- [Gemma-4 toolkit and reproduction scripts](https://github.com/Wondernuttz/OpenVino-For-Gemma-4)
