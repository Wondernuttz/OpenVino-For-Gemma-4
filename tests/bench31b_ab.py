import openvino as ov, openvino_genai as g, time, sys
MODEL = "/home/wondernutts/models/heretics/gemma-4-31B-heretic-int4-ov-lutfix"
DEV = "GPU.1"
core = ov.Core()
pci = str(core.get_property(DEV, "DEVICE_PCI_INFO"))
assert "bus: 4" in pci, "GPU.1 not bus 4: " + pci
def bench(tag, props):
    p = g.VLMPipeline(MODEL, DEV, **props)
    big = "The caravan wound slowly through the mountain pass while wary guards watched the ridgeline for wolves and worse. " * 300
    prompt = "<bos><|turn>user\n" + big + "\nSummarize in five words.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    c = g.GenerationConfig(); c.max_new_tokens = 32; c.do_sample = False
    try: c.apply_chat_template = False
    except Exception: pass
    r = p.generate(prompt, generation_config=c)  # warm-up
    r = p.generate(prompt, generation_config=c)
    m = r.perf_metrics
    ttft = m.get_ttft().mean / 1000.0
    n_in = m.get_num_input_tokens()
    print("%s: input=%d tok, TTFT=%.2fs -> PREFILL %.0f tok/s | decode %.1f tok/s | out: %r" % (
        tag, n_in, ttft, n_in/ttft, m.get_throughput().mean, str(r)[:60]), flush=True)
    del p
bench("DQ=0 (serving cfg)", {"DYNAMIC_QUANTIZATION_GROUP_SIZE": 0})
bench("DQ default        ", {})
print("AB DONE", flush=True)
