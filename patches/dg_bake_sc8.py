#!/usr/bin/env python3
"""Permanently bake the SC8 rewrite into the c32 IR and prune the dead int4 SC-embed
duplicate from the artifact (~370MB of trace-materialized, scope-evading weight that
violated the export's own D2 fp32-SC intent). save_model serializes reachable nodes
only, so the duplicate and its dequant chain vanish from the .bin.

Run ONLY after the load-time SC8 surgery has passed validation.
Usage: python dg_bake_sc8.py [SRC_DIR] [DST_DIR]
       (defaults: ~/dg_ov_c32_lut with the LUT already baked -> ~/dg_ov_c32_sc8)
"""
import json, os, shutil, sys
import numpy as np
import openvino as ov
from openvino import opset13 as ops

SRC = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/dg_ov_c32_lut")
DST = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/dg_ov_c32_sc8")
VOCAB, HID = 262144, 2816

core = ov.Core()
m = core.read_model(os.path.join(SRC, "ir_bb_int4.xml"))

sc_param = next(p for p in m.get_parameters()
                if "self_conditioning_logits" in p.output(0).get_names())
sm = None
frontier = [t.get_node() for t in sc_param.output(0).get_target_inputs()]
for _ in range(6):
    nxt = []
    for n in frontier:
        if n.get_type_name() == "Softmax":
            sm = n; break
        nxt += [t.get_node() for t in n.output(0).get_target_inputs()]
    if sm is not None:
        break
    frontier = nxt
assert sm is not None, "softmax not found"

PASSTHRU = ("Convert", "Reshape", "Unsqueeze", "Squeeze")
mm_sc = None
frontier = [sm]
for _ in range(4):
    nxt = []
    for fn in frontier:
        for t in fn.output(0).get_target_inputs():
            n = t.get_node()
            if n.get_type_name() == "MatMul":
                mm_sc = n
            elif n.get_type_name() in PASSTHRU:
                nxt.append(n)
        if mm_sc is not None:
            break
    if mm_sc is not None:
        break
    frontier = nxt
assert mm_sc is not None, "soft-embed MatMul not found"

# the clean bf16 embed table. The FULL IR may carry it once (tied+deduped) or twice
# (embed + lm_head); prefer the one feeding a Gather (the token-id lookup).
cands = []
for op in m.get_ops():
    if op.get_type_name() != "Constant":
        continue
    ps = op.get_output_partial_shape(0)
    if ps.rank.is_static and ps.rank.get_length() == 2 and list(ps.to_shape()) == [VOCAB, HID]:
        cands.append(op)
assert cands, "no [VOCAB, HID] constant found"
emb = None
for c in cands:
    consumers = {t.get_node().get_type_name() for t in c.output(0).get_target_inputs()}
    if "Gather" in consumers:
        emb = c; break
if emb is None:
    assert len(cands) == 1, f"{len(cands)} embed candidates, none Gather-fed"
    emb = cands[0]
print(f"embed table: {emb.get_friendly_name()[:70]}")

out_et = mm_sc.get_output_element_type(0)
p8 = ops.parameter(ov.PartialShape([1, -1, 8]), ov.Type.f32, name="sc_top_p")
p8.output(0).get_tensor().set_names({"sc_top_p"})
i8 = ops.parameter(ov.PartialShape([1, -1, 8]), ov.Type.i64, name="sc_top_i")
i8.output(0).get_tensor().set_names({"sc_top_i"})
g8 = ops.gather(emb.output(0), i8.output(0), ops.constant(np.int64(0)).output(0))
g8f = ops.convert(g8.output(0), ov.Type.f32)
pw = ops.unsqueeze(p8.output(0), ops.constant(np.int64(2)).output(0))
soft = ops.matmul(pw.output(0), g8f.output(0), False, False)
softs = ops.squeeze(soft.output(0), ops.constant(np.array([2], np.int64)).output(0))
soft_out = softs.output(0)
if out_et != ov.Type.f32:
    soft_out = ops.convert(soft_out, out_et).output(0)

for t in list(mm_sc.output(0).get_target_inputs()):
    t.replace_source_output(soft_out)
ph = ops.constant(np.zeros((1, 1, VOCAB), np.float32))
for t in list(sc_param.output(0).get_target_inputs()):
    t.replace_source_output(ph.output(0))
m.remove_parameter(sc_param)
m.add_parameters([p8, i8])
m.validate_nodes_and_infer_types()
mask_note = "kept"
for p in list(m.get_parameters()):
    if "self_conditioning_mask" in p.output(0).get_names() and not p.output(0).get_target_inputs():
        m.remove_parameter(p); mask_note = "dead, removed"
print(f"sc8 baked (sc_mask {mask_note}); saving...")

shutil.rmtree(DST, ignore_errors=True)
os.makedirs(DST, exist_ok=True)
ov.save_model(m, os.path.join(DST, "ir_bb_int4.xml"), compress_to_fp16=False)
for f in ("sampler_manifest.json", "tokenizer.json", "tokenizer_config.json",
          "config.json", "chat_template.jinja"):
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        shutil.copy(p, DST)
man = json.load(open(os.path.join(DST, "sampler_manifest.json")))
man["sc8_baked"] = {
    "inputs": {"sc_top_p": "[1, L, 8] f32 softmax of the top-8 TEMPERATURE-SCALED logits",
               "sc_top_i": "[1, L, 8] i64 top-8 token ids"},
    "why": ("self-conditioning reads 8 gathered bf16 embed rows instead of the full-vocab "
            "softmax @ embed matmul. The original traced graph carried an int4-quantized, "
            "hidden-major DUPLICATE of the tied embedding for this path (materialized at "
            "trace time, missed by every ignored_scope pattern, violating the fp32-SC "
            "intent); it has been pruned from this artifact. Top-8 truncation is "
            "cross-engine validated (Arcaine soft_next=topk:8, 2026-07-12)."),
}
json.dump(man, open(os.path.join(DST, "sampler_manifest.json"), "w"), indent=2)
old = sum(os.path.getsize(os.path.join(SRC, f)) for f in os.listdir(SRC)
          if f.startswith("ir_bb")) / 1e9
new = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)
          if f.startswith("ir_bb")) / 1e9
print(f"BAKE DONE: IR {old:.2f} GB -> {new:.2f} GB ({(old-new)*1000:.0f} MB pruned)")
