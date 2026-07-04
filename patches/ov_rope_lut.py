
# Replace Gemma4 rope angle computation (position x inv_freq -> Sin/Cos) with
# precomputed sin/cos lookup tables + Gather(position_ids). Sin/cos VALUES are
# in [-1,1] (fp16-safe); the fp16 blowup only happens computing the big angle
# at runtime, and a Gather has no arithmetic to corrupt.
import json, os, sys
import numpy as np
import openvino as ov
from openvino import op, Type, Shape
try:
    from openvino import opset13 as ops
except Exception:
    import openvino.opset13 as ops

SRC = sys.argv[1]
DST = sys.argv[2]
MAXPOS = int(os.environ.get("LUT_MAXPOS", "40960"))  # position_ids are clamped to this

core = ov.Core()
m = core.read_model(SRC + "/openvino_language_model.xml")

byname = {n.get_friendly_name(): n for n in m.get_ops()}
pos_param = None
for p in m.get_parameters():
    if p.get_friendly_name() == "position_ids":
        pos_param = p
assert pos_param is not None, "no position_ids parameter"

# clamped indices shared by all gathers: Minimum(position_ids, MAXPOS-1)
cap = op.Constant(Type.i64, Shape([1]), [MAXPOS - 1])
idx = ops.minimum(pos_param, cap)
idx.set_friendly_name("rope_lut/clamped_positions")
axis0 = op.Constant(Type.i64, Shape([]), [0])

def trace_inv_freq(sin_node):
    """walk Sin <- Concat <- Transpose <- MatMul <- Broadcast <- Constant[1,F,1]"""
    concat = sin_node.input_value(0).get_node()
    transpose = concat.input_value(0).get_node()
    matmul = transpose.input_value(0).get_node()
    inv = None
    for i in range(matmul.get_input_size()):
        n = matmul.input_value(i).get_node()
        if n.get_type_name() == "Broadcast":
            c = n.input_value(0).get_node()
            assert c.get_type_name() == "Constant", "Broadcast input not Constant: " + c.get_type_name()
            inv = np.array(c.get_data()).astype(np.float64).flatten()
    assert inv is not None, "no inv_freq const found behind " + sin_node.get_friendly_name()
    # Gemma-4 global attention uses PROPORTIONAL (partial) RoPE: only the first
    # rope_angles frequencies are real, the rest are ZERO BY DESIGN (nope dims,
    # cos=1/sin=0 = identity). Preserve them -- do NOT fill the spectrum.
    print(">>> %d real freqs + %d nope (zero) dims -- zeros preserved (p-RoPE)" % (
        int((inv != 0).sum()), int((inv == 0).sum())), flush=True)
    return concat, inv

patched = 0
for name in list(byname):
    node = byname[name]
    if node.get_type_name() != "Sin" or "rotary_emb" not in name:
        continue
    sin_node = node
    concat, inv = trace_inv_freq(sin_node)
    # find the sibling Cos fed by the same Concat
    cos_node = None
    for ti in concat.output(0).get_target_inputs():
        t = ti.get_node()
        if t.get_type_name() == "Cos":
            cos_node = t
    assert cos_node is not None, "no sibling Cos for " + name

    F = inv.shape[0]
    print(">>> path %s: %d freqs, inv_freq[min,max]=[%.3g, %.3g], out dim %d" % (
        name.split("/")[-1], F, float(inv.min()), float(inv.max()), 2 * F), flush=True)
    ang = np.arange(MAXPOS, dtype=np.float64)[:, None] * inv[None, :]   # [MAXPOS, F]
    ang = np.concatenate([ang, ang], axis=1)                            # cat(x,x) -> [MAXPOS, 2F]
    sin_t = np.sin(ang).astype(np.float32)
    cos_t = np.cos(ang).astype(np.float32)

    sin_c = op.Constant(sin_t)
    cos_c = op.Constant(cos_t)
    g_sin = ops.gather(sin_c, idx, axis0)   # [B, S, 2F] f32
    g_cos = ops.gather(cos_c, idx, axis0)
    g_sin.set_friendly_name(name + "/lut_gather")
    g_cos.set_friendly_name(cos_node.get_friendly_name() + "/lut_gather")

    for ti in list(sin_node.output(0).get_target_inputs()):
        ti.replace_source_output(g_sin.output(0))
    for ti in list(cos_node.output(0).get_target_inputs()):
        ti.replace_source_output(g_cos.output(0))
    patched += 1

assert patched >= 2, "expected >=2 rope paths, patched %d" % patched
m.validate_nodes_and_infer_types()
print(">>> both rope paths now table-driven; saving to %s ..." % DST, flush=True)
os.makedirs(DST, exist_ok=True)
ov.save_model(m, DST + "/openvino_language_model.xml")

# copy the small sidecar files + link the big untouched bins
import shutil, glob
for f in glob.glob(SRC + "/*"):
    b = os.path.basename(f)
    if b.startswith("openvino_language_model."):
        continue
    dst = os.path.join(DST, b)
    if os.path.exists(dst):
        continue
    if os.path.getsize(f) > 200 * 1024 * 1024:
        os.symlink(f, dst)
    else:
        shutil.copy2(f, dst)
print(">>> DONE: " + DST, flush=True)
