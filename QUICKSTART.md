# Quickstart: run the models with zero OpenVINO experience

## What you need

- An Intel Arc GPU. For the 26B MoE: 16 GB VRAM minimum (A770 16GB is tight, B60/B70 comfortable).
  For the 31B: 24 GB minimum, 32 GB recommended.
- About 15 GB (26B) or 18 GB (31B) of disk.
- Python 3.10 or newer.
- A current Arc driver. Windows: update through Intel Arc Control. Linux: install the Intel
  compute runtime for your distro.
- A free HuggingFace account (click through the content warning on the model page once).

## Install

One package. No compiling, no llama.cpp, no CUDA.

```bash
pip install openvino-genai==2026.2.0 huggingface_hub
```

## Download

```bash
huggingface-cli download Wondernutts/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov --local-dir ./gemma4-26b-ov
```

## Run

Save as `chat.py`, then `python chat.py`:

```python
import openvino_genai as g

pipe = g.VLMPipeline("./gemma4-26b-ov", "GPU")

def chat(system, user):
    p = "<bos><|turn>system\n" + system + "<turn|>\n"
    p += "<|turn>user\n" + user + "<turn|>\n<|turn>model\n"
    p += "<|channel>thought\n<channel|>"   # fast mode, skips reasoning
    c = g.GenerationConfig()
    c.max_new_tokens = 512
    c.do_sample = True; c.temperature = 0.9; c.top_p = 0.95
    try: c.repetition_penalty = 1.2
    except Exception: pass
    try: c.apply_chat_template = False
    except Exception: pass
    return str(pipe.generate(p, generation_config=c))

print(chat("You are a witty tavern keeper in Whiterun.", "Rough night?"))
```

First load takes about 20 seconds. On a B70 you should see roughly 99 tokens a second from the
26B MoE.

## Troubleshooting

- `"GPU"` errors and you have more than one graphics device (e.g. a laptop iGPU): try `"GPU.1"`,
  and confirm which is which with
  `openvino.Core().get_property("GPU.1", "DEVICE_PCI_INFO")`. See README issue #10.
- Output repeats or rambles in long sessions: keep repetition_penalty at 1.2 and never use
  JSON grammar mode. See README issues #5 and #7.
- Want reasoning mode: put `<|think|>` in the system turn, do not pre-close the thought channel,
  and give it at least 1024 max_new_tokens. See README issue #12.

## Serve it as an OpenAI-compatible endpoint (SillyTavern, CHIM, bots)

`serving/ovserver_moe.py` exposes `/v1/chat/completions`:

```bash
OV_MODEL=./gemma4-26b-ov OV_DEVICE=GPU OV_PORT=8092 python serving/ovserver_moe.py
```

Point your client at `http://127.0.0.1:8092/v1/chat/completions`, model name
`gemma-4-26b-a4b-heretic`, blank API key, JSON/structured-output mode OFF.
