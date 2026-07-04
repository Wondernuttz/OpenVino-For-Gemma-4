import numpy as np, openvino as ov, sys
try:
    from openvino.runtime import op, Type, Shape
except Exception:
    from openvino import op, Type, Shape
ORIG = "/home/wondernutts/models/heretics/gemma-4-26B-A4B-heretic-int4-ov"
NEW = ORIG + "-ropefix"
core = ov.Core()
m = core.read_model(ORIG + "/openvino_language_model.xml")
# 1) full-spectrum rope const (best value config)
correct = (1e6 ** (-(2.0 * np.arange(256)) / 512)).astype(np.float32)
for node in m.get_ops():
    if node.get_type_name() == "Constant" and "rotary_emb" in node.get_friendly_name():
        shp = list(node.get_output_shape(0))
        if int(np.prod(shp)) != 256: continue
        arr = None
        for g in ("get_data", "get_vector"):
            try: arr = np.array(getattr(node, g)()).flatten(); break
            except Exception: pass
        if arr is not None and (arr == 0).sum() > 100:
            nc = op.Constant(Type.f32, Shape([int(x) for x in shp]), [float(x) for x in correct])
            for ti in list(node.output(0).get_target_inputs()):
                ti.replace_source_output(nc.output(0))
            break
# 2) mark all rope ops to stay full-precision (don't downcast to fp16 at runtime)
marked = 0
for node in m.get_ops():
    if "rotary_emb" in node.get_friendly_name():
        rt = node.get_rt_info(); rt["precise"] = 1
        try: rt["disable_fp16_compression"] = True
        except Exception: pass
        marked += 1
print(">>> rope const full-spectrum + %d ops marked precise; saving ..." % marked, flush=True)
ov.save_model(m, NEW + "/openvino_language_model.xml")
print(">>> DONE", flush=True)
