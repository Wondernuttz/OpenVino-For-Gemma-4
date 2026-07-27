---
license: apache-2.0
library_name: openvino
pipeline_tag: image-text-to-text
base_model: CCSSNE/tvall43-Qwen3.6-35B-A3B-heretic
tags:
- openvino
- openvino-genai
- int4
- intel-arc
- qwen3_5
- mixture-of-experts
- roleplay
- uncensored
language:
- en
---

# Qwen3.6-35B-A3B Heretic OpenVINO INT4

This is the OpenVINO INT4 conversion of the tvall43 heretic Qwen3.6-35B-A3B tune. It is a 256-expert MoE with about 3B active parameters per token.

The graph can load with OpenVINO 2026.2, but the performance numbers below were measured with the custom Intel GPU plugin from [Wondernuttz/openvino](https://github.com/Wondernuttz/openvino/tree/arc-xe2-int4-2026.2). Do not attribute these numbers to the stock plugin.

## B70 benchmark

Measured 2026-07-25 on one cleared Intel Arc Pro B70 32 GB card:

| Input tokens | Prompt processing | Decode |
|---:|---:|---:|
| 2,048 | 3,392.8 tok/s | 106.2 tok/s |
| 6,144 | 4,108.0 tok/s | 106.0 tok/s |
| 16,384 | 3,914.6 tok/s | 106.0 tok/s |
| 32,768 | 3,128.2 tok/s | 102.8 tok/s on the separate short decode case |
| 31,000 retrieval prompt | 3,080.6 tok/s | 83.9 tok/s at long context |

Fresh pipeline construction took 27 to 29 seconds. The 9 GB scheduler cache configuration used about 28 GB of device memory.

The 31K coherence run generated 185 tokens and passed all four checks:

- Recovered Ysolda's key location exactly
- Recovered all four Vigor of the Nine ingredients
- Recovered the exact `shadow-hearth` password
- Wrote Entry 121 in the requested style and ending

### Measurement configuration

- Ubuntu 24.04.4 LTS
- Linux 7.0.0 with the `xe` driver
- Intel compute runtime 26.22.38646.6
- OpenVINO 2026.2.0
- OpenVINO GenAI 2026.2.0.0
- Custom fork implementation commit [`6306353b`](https://github.com/Wondernuttz/openvino/commit/6306353b0371c2bbb5fad05eea90e0fde3b45642)
- Custom plugin SHA-256 on the test system: `5dc9dacfad9c300c5e27643ac8bba18ffdd4fbca82fd422933a68d000904a696`
- `MOE_MICRO_GEMM_N_HINT=256`
- `MOE_USE_GROUPED_GEMM_PREFILL=0`
- `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`
- Prefix caching enabled with a 9 GB cache
- `max_num_batched_tokens=4096`
- Single stream

Each prompt shape was measured in a fresh process after a 64-token compilation warmup. The measured prompt was new, so prefix caching did not serve it from an earlier request. Prompt processing is input tokens divided by TTFT. Decode tests forced 128 output tokens. The benchmark configuration and fork build notes are recorded in [ARC-XE2-INT4.md](https://github.com/Wondernuttz/openvino/blob/arc-xe2-int4-2026.2/ARC-XE2-INT4.md).

## Install

The exact tested path is Linux x86-64 on Arc Pro B70. Windows and other Arc generations are not validated yet.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  openvino==2026.2.0 \
  openvino-genai==2026.2.0.0 \
  "huggingface_hub>=1.19,<2"
```

Build and install the custom Intel GPU plugin from the [Wondernuttz Arc Xe2 branch](https://github.com/Wondernuttz/openvino/tree/arc-xe2-int4-2026.2). The complete build and portable RPATH instructions are in [ARC-XE2-INT4.md](https://github.com/Wondernuttz/openvino/blob/arc-xe2-int4-2026.2/ARC-XE2-INT4.md).

Download the model:

```bash
hf download Wondernutts/tvall43-Qwen3.6-35B-A3B-heretic-int4-ov \
  --local-dir ./qwen36-heretic-ov
```

## Run

Set the tested kernel-selection properties before starting Python:

```bash
export MOE_MICRO_GEMM_N_HINT=256
export MOE_USE_GROUPED_GEMM_PREFILL=0
```

```python
import openvino_genai as ov_genai

model_dir = "./qwen36-heretic-ov"

scheduler = ov_genai.SchedulerConfig()
scheduler.enable_prefix_caching = True
scheduler.cache_size = 9
scheduler.max_num_batched_tokens = 4096

pipe = ov_genai.VLMPipeline(
    model_dir,
    "GPU",
    scheduler_config=scheduler,
    DYNAMIC_QUANTIZATION_GROUP_SIZE=0,
)

system = "You are Lydia, housecarl to the Dragonborn. Dry wit, fiercely loyal."
user = "We have walked this frozen pass for six hours. Say something."
prompt = (
    "<|im_start|>system\n" + system + "<|im_end|>\n"
    "<|im_start|>user\n" + user + "<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 256
config.do_sample = True
config.temperature = 0.9
config.top_p = 0.95
config.repetition_penalty = 1.2
config.apply_chat_template = False

result = pipe.generate(prompt, generation_config=config)
print(result.texts[0])
```

The preclosed `<think>...</think>` block requests a direct answer. Remove it when reasoning output is wanted and allow a larger generation budget.

To list the exact GPU IDs exposed by OpenVINO:

```bash
python - <<'PY'
import openvino as ov
core = ov.Core()
for device in core.available_devices:
    if device.startswith("GPU"):
        print(device, core.get_property(device, "DEVICE_PCI_INFO"))
PY
```

Use `GPU.0`, `GPU.1`, and so on when more than one Intel GPU is installed.

## Limits

- This exact 9 GB cache setup needs a 32 GB card.
- The published measurements are single-card and single-stream.
- Text generation is validated. Image input is not.
- The 31K retrieval run and 32,768-token PP case are the largest tested contexts. The architecture may support more, but this release does not claim it.
- This is an INT4 model. It does not use the Bonsai INTBIT or INTERNARY kernels.
- The model is uncensored and outputs are unfiltered. Users are responsible for lawful use.

## Conversion and attribution

The conversion uses INT4 asymmetric compression, group size 64, `backup_precision="int8_sym"`, no AWQ, and image-text-to-text export.

Base model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

Heretic tune: [CCSSNE/tvall43-Qwen3.6-35B-A3B-heretic](https://huggingface.co/CCSSNE/tvall43-Qwen3.6-35B-A3B-heretic)

OpenVINO conversion and Arc GPU work: Wondernutts
