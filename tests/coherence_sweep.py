import openvino as ov, openvino_genai as g, time
MODEL = "/home/wondernutts/models/heretics/gemma-4-26B-A4B-heretic-int4-ov-lutfix"
DEV = "GPU.2"
core = ov.Core()
print("DEV %s PCI=%s" % (DEV, str(core.get_property(DEV, "DEVICE_PCI_INFO"))), flush=True)
pci = str(core.get_property(DEV, "DEVICE_PCI_INFO"))
assert "08:00" in pci or "bus: 8" in pci, "GPU.2 is not bus 8 in this runtime: " + pci
print(">>> loading LUT model (2026.2) ...", flush=True)
p = g.VLMPipeline(MODEL, DEV, **{"DYNAMIC_QUANTIZATION_GROUP_SIZE": 0})
base = [
 "The road to Whiterun turned to mud after the storm and the cart wheels groaned.",
 "Lydia sharpened her blade by the fire while the embers crackled in the dark.",
 "A pack of wolves was seen prowling near the western watchtower at dusk.",
 "The merchant from Riften swore his moonstone was genuine, but his eyes lied.",
 "Snow fell heavy over the Throat of the World and the pilgrims pressed on.",
 "A courier delivered a sealed letter bearing the mark of the Jarl of Falkreath.",
 "The mead in Whiterun tasted of honey and smoke and old arguments.",
 "Bandits had fortified the ruined fort and posted archers on the crumbling walls.",
 "A dragon was rumored to have burned a farmstead near the river two nights past.",
 "The blacksmith hammered a new shield while sparks leapt across the cold floor.",
 "Frost spiders nested deep in the barrow and the air smelled of rot and dust.",
 "A bard sang of the Dragonborn in the inn until the patrons begged him to stop.",
 "The hold guard eyed every traveler who passed beneath the gate at nightfall.",
 "An alchemist traded a frost-resist potion for three deathbell flowers.",
 "The river ran fast and gray and the bridge timbers creaked under the wagon.",
 "A hunter tracked an elk through the pines but lost it among the standing stones.",
 "The temple of Kynareth kept its doors open even in the worst of the blizzards.",
 "A thief slipped through the market with a coin purse and a guilty grin.",
 "The old soldier spoke of the war as if the dead still marched beside him.",
 "Lanterns swung over the docks and the smell of salt and tar filled the air.",
]
def make_ctx(tc):
    parts = []; n = 0; i = 0
    while n < tc:
        s = "Day %d: %s " % (i + 1, base[i % len(base)]); parts.append(s); n += len(s); i += 1
    return "".join(parts)
def test(ctx_tokens):
    sysp = "You are Lydia, housecarl to the Dragonborn. Stay in character, concise."
    user = "Our journey so far:\n" + make_ctx(ctx_tokens * 4) + "\n\nLydia, should we make camp here tonight? Answer in ONE sentence."
    prompt = "<bos><|turn>system\n" + sysp + "<turn|>\n<|turn>user\n" + user + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    c = g.GenerationConfig(); c.max_new_tokens = 70; c.do_sample = True; c.temperature = 0.9; c.top_p = 0.95
    try: c.apply_chat_template = False
    except Exception: pass
    try: c.repetition_penalty = 1.2
    except Exception: pass
    t0 = time.time(); r = p.generate(prompt, generation_config=c); dt = time.time() - t0
    out = str(r); w = out.split(); uniq = len(set(w)) / max(len(w), 1)
    flag = "<-- GIBBERISH?" if (uniq < 0.5 or len(w) < 3) else "ok"
    print("ctx ~%2dK: (%5.1fs) uniq=%.2f %s %r" % (ctx_tokens // 1000, dt, uniq, flag, out[:110]), flush=True)
print(">>> CORRECT p-RoPE model coherence (old: clean 16K / wobble 20K; pushing to 32K):", flush=True)
for k in [8000, 16000, 24000, 32000]:
    test(k)
