#!/usr/bin/env python3
"""
DiffusionGemma OpenVINO block-diffusion sampler -- Phase 2 payoff.

Drives the single unified IR (ir_bb_int4) on one Arc card to generate text:
  ENCODER pass (apply_sc=0, causal): build the prefix KV cache (prompt, then each committed block)
  DECODER denoise loop (apply_sc=1, bidirectional, <=48 steps): refine a noise canvas into tokens

NATIVE CANVAS 256: the Arc onednn gemm writes garbage logits rows >=225 at M=256, so the
lm_head+softcap tail is SPLIT OUT of the backbone at load time and run as its own static
fp32 model (exact same weights and math; also restores the fp32 tail the export intended,
which the GPU plugin otherwise downcasts to f16). Masks assume total length <
sliding_window (1024) -> no windowing (DG_MAX_PROMPT must keep totals under it).

STAGE-2 FAST PATH (DG_FAST=1, default): the legacy loop moved ~950MB/step over PCIe to do
~115ms of device math (268MB SC upload + ~300MB prefix-KV re-upload + 268MB logits download
+ ~110MB unused new-KV download + full-vocab argpartition). Fast path:
  - lm tail rebuilt at M=256 fp32 with temperature Divide + TopK(64) IN-GRAPH: host reads
    only [256,64] values+indices (128KB). Load-time parity probe vs the proven M=128 tail
    (the bf16 M=256 row>=225 bug must not exist in fp32; if it does -> fall back).
  - scaled logits stay on device: the tail writes a RemoteTensor that IS the backbone's
    self_conditioning_logits input next step (zeroed for step 1 by running the tail on a
    zeros hidden state: softcap(0) == 0 exactly).
  - prefix KV uploaded ONCE per block into RemoteTensors (constant during denoise).
  - the 60 unused new_key/new_value outputs bound to RemoteTensors so they never copy to host.
Any probe/API failure at load prints the reason and falls back to the legacy path.

Usage:  python dg_sampler.py ~/dg_ov GPU.1 "Why is the sky blue?" [max_blocks]
"""
import json, os, sys, time
import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor

REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else "./dg_ov"
DEVICE   = sys.argv[2] if len(sys.argv) > 2 else "GPU.1"
PROMPT   = sys.argv[3] if len(sys.argv) > 3 else "Why is the sky blue?"
MAX_NEW_BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
CANVAS = 256
LM_CHUNK = 128   # static M of the legacy split lm_head model; must divide CANVAS
# Arc SDPA kernel page-faults (CL_OUT_OF_RESOURCES) for seq len 9..32 (except exactly 8/16);
# lengths >=33 verified clean up to 256. Pad every encode to >=48, 16-aligned, for margin.
PAD_FLOOR, PAD_ALIGN = 48, 16
NEG = float(np.finfo(np.float32).min)

man = json.load(open(os.path.join(REPO_DIR, "sampler_manifest.json")))
LAYER_TYPES = man["layer_types"]; N_LAYERS = len(LAYER_TYPES)
KVS = man["per_layer_type_kv_shapes"]; VOCAB = int(man["dims"]["vocab_size"])
SD = man["sampler_defaults"]
# tuning knobs (env overrides for the coherence-repair loop)
MAX_STEPS = int(os.environ.get("DG_STEPS", SD["max_denoising_steps"]))
EBOUND = float(os.environ.get("DG_EBOUND", SD["entropy_bound"]))
T_MIN = float(os.environ.get("DG_TMIN", SD["t_min"]))
T_MAX = float(os.environ.get("DG_TMAX", SD["t_max"]))
CONF = float(SD["confidence_threshold"])
TOPK = int(os.environ.get("DG_TOPK", 0))          # 0 = off; else restrict sampling to top-k
WARM = int(os.environ.get("DG_WARM", 4))          # steps before locking starts (let SC signal form)
REV_H = float(os.environ.get("DG_REVH", 0.05))    # unlock a locked pos if model confidently disagrees
FREQPEN = float(os.environ.get("DG_FREQPEN", 0.1))  # logit penalty per locked occurrence past the floor
FREQ_FLOOR = 8                                    # occurrences a token gets free (normal ' the' rates)
# adjacent-duplicate penalty (2026-07-12): parallel denoising lets NEIGHBORING positions
# independently lock the same content (off-by-one alignment ambiguity) -> "the the",
# "(I I", "you you" artifacts. Dock a candidate's logit when it equals an already-locked
# neighbor's token. Targeted at the diffusion-specific failure; distant repeats untouched
# (a global rep-pen would strangle legit ' the' x40 replies). EOS/PAD exempt (canvas
# padding repeats by design). 0 disables.
ADJPEN = float(os.environ.get("DG_ADJPEN", 3.0))
THINK = os.environ.get("DG_THINK", "0") == "1"    # thinking mode (default off, like the MoE server)
SLIDING = 1024                                    # sliding-layer window (enforced in the masks)
MAX_PROMPT_TOK = int(os.environ.get("DG_MAX_PROMPT", 660))  # keeps cache+canvas inside the sliding window
# capacity-dispatch encode contract (C=24 exports): each expert has only C static slots per
# forward, verified drop-free at T=256 only. Prompt encodes above the chunk bound must be
# split into causal chunks against the growing prefix KV (exact math; chunk k attends
# chunks <k through the cache). 0 = off (C=48-era models without the manifest key).
ENC_CHUNK = int(os.environ.get("DG_ENC_CHUNK",
                man.get("capacity_dispatch", {}).get("encode_chunk_max", 0) or 0))
MAX_TOTAL_TOK = 2352                              # the ladder's proven top; never extrapolate past it
DUMMY = int(os.environ.get("DG_DUMMY", 48))       # masked dummy cache prefix (48 = safe; 768 = POISON)
WARMBIG = os.environ.get("DG_WARMBIG", "1") == "1"  # warm the big-context ladder rungs
DIAG = bool(os.environ.get("DG_DIAG"))
FAST = os.environ.get("DG_FAST", "1") == "1"      # Stage-2 device-resident fast path
# DG_SC8: feed self-conditioning from the TOP-8 candidates only instead of the full
# 262144-wide softmax @ embedding matmul (an lm_head-sized matmul INSIDE the backbone,
# every step). Cross-engine validated 2026-07-12: Arcaine's soft_next=topk:8 produces
# persona replies indistinguishable from full-SC quality at 2x the speed. Surgery swaps
# softmax(sc_logits) @ embed for gather(embed, top8_idx) weighted by host-softmaxed
# top-8 probs; the sc input shrinks from 268MB to 10KB per step. 0 = original path.
SC8 = os.environ.get("DG_SC8", "1") == "1"
PROF = bool(os.environ.get("DG_PROF"))            # per-phase step timing
# ADAPTIVE STOPPING v2 (2026-07-12): the model is designed to finish blocks in ~13-17
# forwards (google card: "15-20 tokens per forward pass"). v1's global argmax-stability
# criterion can NEVER fire here: the discard zone is re-randomized every step by design,
# so full-canvas convergence does not exist in the sticky sampler. What converges is the
# thing we actually ship -- the confident prefix. Stops:
#   prefix-complete: a locked EOS with everything before it locked (clean finishes).
#   boundary-stall: the trim boundary (lock-density collapse point) is solid (<= 2
#     unlocked holes, argmax-filled at the horizon anyway) and has not GROWN for
#     STABLE_N consecutive steps -> locking has stalled, later steps only polish
#     tokens the trim discards.
# Both respect MIN_STEPS (argmax-fill is temperature-invariant, stopping mid-anneal is
# safe). MIN_STEPS=0 disables adaptive stopping entirely.
STABLE_N = int(os.environ.get("DG_STABLE_N", 3))   # consecutive boundary-stable steps
MIN_STEPS = int(os.environ.get("DG_MIN_STEPS", 10))
SEED_TEXT = os.environ.get("DG_SEED_TEXT", "")    # canvas seeding: pre-lock the reply's first
                                                  # tokens (diffusion-native prefill; kills
                                                  # "I am Gemma" persona refusals structurally)
_seed_ids = None                                  # armed per generate() for block 0 only
TOK = man["token_ids"]; EOS = TOK.get("eos")
def kv_hd(lt):
    d = KVS["full_attention"] if lt == "full_attention" else KVS["sliding_attention"]
    return int(d["kv_heads"]), int(d["head_dim"])

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(REPO_DIR)

core = ov.Core()
core.set_property({"CACHE_DIR": os.path.join(REPO_DIR, ".ovcache")})
m = core.read_model(os.path.join(REPO_DIR, "ir_bb_int4.xml"))

# ---- TAIL SPLIT: cut at the lm_head MatMul. Main model outputs last_hidden instead of
# logits (broken tail pruned entirely); the MatMul..softcap chain becomes its own model. ----
def _nonconst_input(node):
    for i in range(node.get_input_size()):
        prod = node.input_value(i).get_node()
        while prod.get_type_name() == "Convert":
            prod = prod.input_value(0).get_node()
        if prod.get_type_name() not in ("Constant",):
            return i, node.input_value(i)
    raise RuntimeError(f"no non-const input on {node.get_friendly_name()}")

logits_res = None
for r in m.get_results():
    if "logits" in r.output(0).get_names():
        logits_res = r; break
assert logits_res is not None, "logits result not found"
node = logits_res.input_value(0).get_node()
hops = []
while node.get_type_name() != "MatMul":
    hops.append(node.get_type_name())
    node = _nonconst_input(node)[1].get_node()
mm = node
hid_idx, hidden_out = _nonconst_input(mm)
print(f"tail split: logits <- {' <- '.join(hops[::-1])} <- MatMul({mm.get_friendly_name()[:60]}) "
      f"<- hidden {hidden_out.get_partial_shape()} {hidden_out.get_element_type()}")
try:
    _w = mm.input_value(1 - hid_idx).get_node()
    _wchain = []
    while _w.get_type_name() != "Constant" and len(_wchain) < 12:
        _wchain.append(_w.get_type_name()); _w = _w.input_value(0).get_node()
    print(f"lm_head weights: {' <- '.join(_wchain) or 'direct'} <- {_w.get_type_name()} "
          f"{_w.get_output_partial_shape(0)} {_w.get_output_element_type(0)}")
except Exception as e:
    print("lm_head weight probe failed:", repr(e)[:80])

try:
    import openvino.opset13 as ops
except ImportError:
    from openvino.runtime import opset13 as ops
_hs = hidden_out.get_partial_shape()
HID = _hs[_hs.rank.get_length() - 1].get_length()
lm_param = ops.parameter(ov.PartialShape([1, LM_CHUNK, HID]),
                         hidden_out.get_element_type(), name="last_hidden_in")
mm.input(hid_idx).replace_source_output(lm_param.output(0))
lm_model = ov.Model([logits_res.input_value(0)], [lm_param], "lm_tail")
lm_model.outputs[0].get_tensor().set_names({"logits"})

res_h = ops.result(hidden_out)
keep = [r for r in m.get_results() if r is not logits_res]
main_model = ov.Model(keep + [res_h], m.get_parameters(), "backbone_nolm")
main_model.outputs[-1].get_tensor().set_names({"last_hidden"})

# ---- SC8 SURGERY: replace softmax(sc_logits [1,L,262144]) @ embed with an 8-row
# gather + weighted sum. Discovery first, mutation last; ANY discovery failure falls
# back to the original graph (SC8=False). Mutation failures abort the load loudly. ----
HAS_SCMASK = True
_sc8_baked = any("sc_top_p" in p.output(0).get_names() for p in main_model.get_parameters())
if _sc8_baked:
    SC8 = True                                    # the IR ships the rewrite; nothing to do
    HAS_SCMASK = any("self_conditioning_mask" in p.output(0).get_names()
                     for p in main_model.get_parameters())
    print(f"sc8: baked into IR (sc_mask {'present' if HAS_SCMASK else 'absent'})")
if SC8 and not _sc8_baked:
    try:
        sc_param = None
        for p in main_model.get_parameters():
            if "self_conditioning_logits" in p.output(0).get_names():
                sc_param = p; break
        assert sc_param is not None, "sc parameter not found"
        # walk consumers to the Softmax (may pass through Convert/Multiply)
        sm = None
        frontier = [t.get_node() for t in sc_param.output(0).get_target_inputs()]
        hops = []
        for _ in range(6):
            nxt = []
            for n in frontier:
                if n.get_type_name() == "Softmax":
                    sm = n; break
                hops.append(n.get_type_name())
                nxt += [t.get_node() for t in n.output(0).get_target_inputs()]
            if sm is not None: break
            frontier = nxt
        assert sm is not None, f"softmax not found from sc param (walked {hops})"
        # the soft-embedding MatMul: consumer of softmax whose other input resolves to
        # the big tied-embedding Constant [262144, HID]
        # softmax -> (Convert/Reshape...) -> MatMul whose weight side resolves (through a
        # possible int4-decompression chain) to a Constant with a VOCAB-sized axis
        PASSTHRU = ("Convert", "Reshape", "Unsqueeze", "Squeeze")
        WCHAIN = ("Convert", "Multiply", "Subtract", "Reshape", "Transpose")
        def _wchain_dump(out, max_hops=6):
            prod = out.get_node(); chain = []
            for _ in range(max_hops):
                shp = prod.get_output_partial_shape(0)
                chain.append(f"{prod.get_type_name()}{shp}")
                if prod.get_input_size() == 0:
                    break
                prod = prod.input_value(0).get_node()
            return " <- ".join(chain)
        def _find_const(out, max_hops=8):
            prod = out.get_node(); seen = 0
            while prod.get_type_name() in WCHAIN and seen < max_hops:
                prod = prod.input_value(0).get_node(); seen += 1
            if prod.get_type_name() == "Constant":
                shp = list(prod.get_output_partial_shape(0).to_shape())
                if len(shp) == 2 and VOCAB in shp and HID in shp:   # the embed table, not a scales tensor
                    return prod.output(0), shp
            return None, None
        # the soft-embed MatMul: first MatMul reachable from the softmax through passthru
        # nodes. Its weight side is a hidden-major INT4-dequant DUPLICATE of the tied embed
        # (materialized by the trace, silently missed by every ignored_scope pattern), so we
        # do NOT gather from it: the backbone's true bf16 embed_tokens table [VOCAB, HID]
        # (scope-protected, already present for the token-id lookup) is both cleaner math
        # (closer to reference than the int4 copy SC has been using) and vocab-major.
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
        emb_out = None; emb_shape = None
        for op in main_model.get_ops():
            if op.get_type_name() != "Constant":
                continue
            ps = op.get_output_partial_shape(0)
            if ps.rank.is_static and ps.rank.get_length() == 2:
                shp = list(ps.to_shape())
                if shp == [VOCAB, HID]:
                    assert emb_out is None, "multiple [VOCAB, HID] constants; embed table ambiguous"
                    emb_out = op.output(0); emb_shape = shp
        assert emb_out is not None, "bf16 embed_tokens constant [VOCAB, HID] not found in backbone"
        if mm_sc is None:
            tree = []
            for t in sm.output(0).get_target_inputs():
                n = t.get_node()
                for t2 in n.output(0).get_target_inputs():
                    n2 = t2.get_node()
                    if n2.get_type_name() == "MatMul":
                        tree.append(f"MatMul in0: {_wchain_dump(n2.input_value(0))}")
                        tree.append(f"MatMul in1: {_wchain_dump(n2.input_value(1))}")
                if not tree:
                    tree.append(f"{n.get_type_name()} -> "
                                f"[{','.join(t2.get_node().get_type_name() for t2 in n.output(0).get_target_inputs())}]")
            raise AssertionError(f"soft-embed MatMul not found; {' | '.join(tree)}")
        assert list(emb_out.get_partial_shape().to_shape())[0] == VOCAB, \
            "embed constant not vocab-major; gather axis assumption invalid"
        out_et = mm_sc.get_output_element_type(0)
        print(f"sc8 surgery: sc -> {'/'.join(hops) or 'direct'} -> Softmax -> MatMul("
              f"{mm_sc.get_friendly_name()[:60]}) out {mm_sc.get_output_partial_shape(0)} {out_et}")
        # new inputs + replacement subgraph (all built BEFORE any mutation)
        p8_par = ops.parameter(ov.PartialShape([1, -1, 8]), ov.Type.f32, name="sc_top_p")
        p8_par.output(0).get_tensor().set_names({"sc_top_p"})
        i8_par = ops.parameter(ov.PartialShape([1, -1, 8]), ov.Type.i64, name="sc_top_i")
        i8_par.output(0).get_tensor().set_names({"sc_top_i"})
        gath8 = ops.gather(emb_out, i8_par.output(0), ops.constant(np.int64(0)).output(0))
        gath8f = ops.convert(gath8.output(0), ov.Type.f32)        # [1,L,8,HID] f32
        pw = ops.unsqueeze(p8_par.output(0), ops.constant(np.int64(2)).output(0))  # [1,L,1,8]
        soft8 = ops.matmul(pw.output(0), gath8f.output(0), False, False)           # [1,L,1,HID]
        soft8s = ops.squeeze(soft8.output(0), ops.constant(np.array([2], np.int64)).output(0))
        soft_out = soft8s.output(0)
        if out_et != ov.Type.f32:
            soft_out = ops.convert(soft_out, out_et).output(0)
        # MUTATIONS (abort load on failure rather than serve a corrupt graph)
        for t in list(mm_sc.output(0).get_target_inputs()):
            t.replace_source_output(soft_out)
        # sever + drop the dead full-vocab input (tiny placeholder keeps the dead chain typed)
        ph = ops.constant(np.zeros((1, 1, VOCAB), np.float32))
        for t in list(sc_param.output(0).get_target_inputs()):
            t.replace_source_output(ph.output(0))
        main_model.remove_parameter(sc_param)
        main_model.add_parameters([p8_par, i8_par])
        main_model.validate_nodes_and_infer_types()
        # is the (B,) sc mask still alive? (it survives if it scales soft_emb downstream)
        mask_par = None
        for p in main_model.get_parameters():
            if "self_conditioning_mask" in p.output(0).get_names():
                mask_par = p; break
        if mask_par is not None and not mask_par.output(0).get_target_inputs():
            main_model.remove_parameter(mask_par)
            HAS_SCMASK = False
        print(f"sc8 armed: SC input 268MB -> 10KB/step, full-vocab soft-embed matmul removed "
              f"(sc_mask {'kept' if HAS_SCMASK else 'dead, removed'})")
    except AssertionError as e:
        print(f"SC8 DISARMED ({e}) -> full-vocab SC path")
        SC8 = False

def _fp32_io(model):
    ppp = PrePostProcessor(model)
    for p in model.inputs:
        if p.get_element_type() == ov.Type.bf16:
            ppp.input(next(iter(p.get_names()))).tensor().set_element_type(ov.Type.f32)
    for p in model.outputs:
        if p.get_element_type() == ov.Type.bf16:
            ppp.output(next(iter(p.get_names()))).tensor().set_element_type(ov.Type.f32)
    return ppp.build()

# DG_DQ0=1: disable the plugin's dynamic int8 activation quantization. The DQ path picks
# an s8-activation x s4-weight batched-gemm kernel that FAULTS at launch for expert
# capacities other than 48 (onednn_verbose 2026-07-12: matmul src:s8 wei:s4
# 128x32x2816:128x2816x1408 -> CL_OUT_OF_RESOURCES; same at C=24; C=48 selects fine).
# DQ off -> f16 x s4 kernels, the family every other matmul in this graph already uses.
_BB_CFG = {}
_dqgs = os.environ.get("DG_DQGS", "")            # explicit group size: 32/64/... ("0" = off)
if os.environ.get("DG_DQ0", "0") == "1" and _dqgs == "":
    _dqgs = "0"                                  # legacy knob: DQ fully off
if _dqgs != "":
    _BB_CFG["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = _dqgs
    print(f"dynamic quantization group size = {_dqgs}" + (" (DISABLED)" if _dqgs == "0" else ""))
    # REMEMBER: .ovcache does NOT rekey on compile-config changes; wipe it when flipping this.
t0 = time.time(); ir = core.compile_model(_fp32_io(main_model), DEVICE, _BB_CFG)
print(f"compiled backbone (no lm tail) on {DEVICE} in {time.time()-t0:.1f}s | canvas={CANVAS} steps<={MAX_STEPS}")
t0 = time.time()
lm = core.compile_model(_fp32_io(lm_model), DEVICE, {"INFERENCE_PRECISION_HINT": "f32"})
print(f"compiled lm tail (static M={LM_CHUNK}, fp32) in {time.time()-t0:.1f}s")

def lm_logits(h):
    """[1, CANVAS, hidden] -> [CANVAS, VOCAB] softcapped fp32 logits via static M=128 chunks."""
    outs = [lm(h[:, c:c + LM_CHUNK].astype(np.float32))[lm.output("logits")]
            for c in range(0, h.shape[1], LM_CHUNK)]
    return np.concatenate([o.reshape(LM_CHUNK, -1) for o in outs], axis=0)

# ---- STAGE-2 FAST PATH (all failures -> legacy). Built and probed strictly AFTER the
# warmup ladder: running ANY kernel before the tiny-causal-first ladder order trips the
# Arc first-inference allocation bug (CL_OUT_OF_RESOURCES on the first ladder rung). ----
K_S = TOPK or 64
lmf = None; rctx = None; sc_rt = None; kvout_rt = None; breq = None; lreq = None

def _run(req, feeds):
    """infer() marshals ALL outputs to numpy, which explodes on remote-bound outputs
    ("Not Implemented"). Set inputs and run async+wait; read outputs selectively."""
    holds = []
    for k, v in feeds.items():
        t = ov.Tensor(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v
        holds.append(t); req.set_tensor(k, t)
    req.start_async(); req.wait()

def _rung(P, L, bidir):
    """Input dict for one warmup ladder rung (P cache cols, L new tokens)."""
    fm = np.zeros((1, 1, L, P + L), np.float32)
    if not bidir:
        fm[0, 0, :, P:] = np.where(np.arange(L)[None] <= np.arange(L)[:, None], 0.0, NEG)
    d_ = {"current_ids": np.zeros((1, L), np.int64),
          "apply_sc": np.array([1.0 if bidir else 0.0], np.float32),
          "position_ids": np.arange(P, P + L)[None].astype(np.int64),
          "full_mask": fm, "sliding_mask": fm}
    if SC8:
        d_["sc_top_p"] = np.zeros((1, L, 8), np.float32)
        d_["sc_top_i"] = np.zeros((1, L, 8), np.int64)
    else:
        d_["self_conditioning_logits"] = np.zeros((1, L, VOCAB), np.float32)
    if HAS_SCMASK:
        d_["self_conditioning_mask"] = np.array([0.0], np.float32)
    for i, lt in enumerate(LAYER_TYPES):
        h, d = kv_hd(lt)
        d_[f"prefix_key_{i}"] = np.zeros((1, h, P, d), np.float32)
        d_[f"prefix_value_{i}"] = np.zeros((1, h, P, d), np.float32)
    return d_

_LADDER = [(4, 8, False), (8, 256, True)]
if ENC_CHUNK:
    _LADDER += [(DUMMY, ENC_CHUNK, False)]   # chunked prompt-encode shape (total stays ascending)
if WARMBIG:
    _LADDER += [(48, 672, False), (720, 256, True), (48, 2048, False), (2096, 256, True)]

def _warmup():
    """Arc first-inference allocation bug: cold-starting most shapes page-faults
    (CL_OUT_OF_RESOURCES). Replay the proven order (tiny causal, then max-shape bidir)
    so internal buffers are sized before real work.
    THE PROVEN LADDER, do not reorder: gently ascending totals 12->264->720->976->2096->2352.
    Aggressive jumps (12->864) and descents both page-fault in oneDNN kernel launch."""
    for (P, L, bidir) in _LADDER:
        if PROF:
            print(f"    [warmup] rung P={P} L={L} bidir={bidir}", flush=True)
        ir(_rung(P, L, bidir))
    lm(np.zeros((1, LM_CHUNK, HID), np.float32))
t0 = time.time(); _warmup(); print(f"warmup done in {time.time()-t0:.1f}s")

if FAST:
    try:
        lm256 = lm_model.clone()
        # remote tensors: unused new-KV outputs never copy to host; prefix KV device-resident
        rctx = core.get_default_context(DEVICE)
        kvout_rt = []
        for i, lt in enumerate(LAYER_TYPES):
            h_, d_ = kv_hd(lt)
            kvout_rt.append((rctx.create_tensor(ov.Type.f32, ov.Shape([1, h_, CANVAS, d_]), {}),
                             rctx.create_tensor(ov.Type.f32, ov.Shape([1, h_, CANVAS, d_]), {})))
        # KV-remote fill probe (copy_from host -> device)
        _probe = rctx.create_tensor(ov.Type.f32, ov.Shape([4]), {})
        _probe.copy_from(ov.Tensor(np.arange(4, dtype=np.float32)))

        # warm the persistent fast request on the ladder's bidir rungs (ascending totals)
        breq = ir.create_infer_request()
        for i in range(N_LAYERS):
            breq.set_tensor(f"new_key_{i}", kvout_rt[i][0])
            breq.set_tensor(f"new_value_{i}", kvout_rt[i][1])
        for (P, L, bidir) in _LADDER:
            if bidir:
                _run(breq, _rung(P, L, bidir))
        print("fast path armed: remote prefix-KV + remote new-KV sinks (hybrid tail)")
    except Exception as e:
        print(f"FAST PATH DISARMED ({repr(e)[:160]}) -> legacy loop")
        FAST = False

# ---- V3 tail: gemm -> divide(temp) -> topk as THREE separate models chained through
# RemoteTensors. The fused single-graph version plans a ~1.1s kernel; separately each is
# fast (bench: gemm 39ms incl download, topk 33ms). Backbone SC input reads the scaled
# logits remote directly -> zero host logits traffic; host receives only [256,64] v+i.
V3 = False

def _topk_tail(src_out, tag=""):
    """(tv, ti) outputs for exact top-K_S along the vocab axis of src_out [.., CANVAS, VOCAB].

    2-STAGE EXACT TOP-K (probe 2026-07-11): the monolithic 262144-wide sorted TopK costs
    80-95ms/step interleaved with the backbone sweep (33ms benched back-to-back; the
    in-loop penalty scales with kernel work, K=16 -> 40ms). Segment-max form runs 2ms/step
    in-loop. Exactness: a global top-K element x in segment s means at most K-1 values
    exceed x, so at most K-1 segment maxes exceed s's max -> s ranks in the top-K segments
    -> gathering those K whole segments keeps x. Two small sorted TopKs replace the huge
    one; indices reconstructed as seg_id*W + col. Falls back to the monolithic node on
    any build error (compile errors bubble to the caller's fallback chain)."""
    try:
        S, W = VOCAB // 64, 64
        assert S * W == VOCAB
        c = lambda v: ops.constant(v).output(0)
        r = ops.reshape(src_out, c(np.array([CANVAS, S, W], np.int64)), False)
        segmax = ops.reduce_max(r.output(0), c(np.array([2], np.int64)), False)
        t1 = ops.topk(segmax.output(0), c(np.int64(K_S)), 1, "max", "value",
                      index_element_type=ov.Type.i32)
        gath = ops.gather(r.output(0), t1.output(1), c(np.int64(1)), batch_dims=1)
        flat = ops.reshape(gath.output(0), c(np.array([CANVAS, K_S * W], np.int64)), False)
        t2 = ops.topk(flat.output(0), c(np.int64(K_S)), 1, "max", "value",
                      index_element_type=ov.Type.i32)
        Wc = c(np.int32(W))
        slot = ops.divide(t2.output(1), Wc)      # indices non-negative: trunc == floor
        col = ops.mod(t2.output(1), Wc)
        seg = ops.gather_elements(t1.output(1), slot.output(0), axis=1)
        ti = ops.add(ops.multiply(seg.output(0), Wc).output(0), col.output(0))
        return t2.output(0), ti.output(0)
    except Exception as te:
        print(f"    2-stage topk build failed{tag} ({repr(te)[:100]}) -> monolithic")
        vt = ops.topk(src_out, ops.constant(np.int64(K_S)).output(0),
                      2, "max", "value", index_element_type=ov.Type.i32)
        return vt.output(0), vt.output(1)
if FAST:
    try:
        lh_rt = rctx.create_tensor(ov.Type.f32, ov.Shape([1, CANVAS, HID]), {})
        lg_rt = rctx.create_tensor(ov.Type.f32, ov.Shape([1, CANVAS, VOCAB]), {})
        g256 = lm_model.clone()
        g256.reshape({g256.inputs[0]: ov.PartialShape([1, CANVAS, HID])})
        gm = core.compile_model(_fp32_io(g256), DEVICE, {"INFERENCE_PRECISION_HINT": "f32"})
        greq = gm.create_infer_request()
        greq.set_tensor("last_hidden_in", lh_rt)
        greq.set_tensor("logits", lg_rt)
        if not SC8:
            # full-vocab SC path: device divide writes sc_rt, which IS the backbone SC input
            sc_rt = rctx.create_tensor(ov.Type.f32, ov.Shape([1, CANVAS, VOCAB]), {})
            sx = ops.parameter(ov.PartialShape([1, CANVAS, VOCAB]), ov.Type.f32, name="x")
            st = ops.parameter(ov.PartialShape([1]), ov.Type.f32, name="temp")
            sm_model = ov.Model([ops.result(ops.divide(sx.output(0), st.output(0)).output(0))],
                                [sx, st], "sc_scale")
            sm_model.outputs[0].get_tensor().set_names({"y"})
            smc = core.compile_model(sm_model, DEVICE)
            sreq = smc.create_infer_request()
            sreq.set_tensor("x", lg_rt)
            sreq.set_tensor("y", sc_rt)
        v_in = ops.parameter(ov.PartialShape([1, CANVAS, VOCAB]), ov.Type.f32, name="x")
        v_tv, v_ti = _topk_tail(v_in.output(0), " (tk3)")
        v_tkm = ov.Model([ops.result(v_tv), ops.result(v_ti)], [v_in], "tk3")
        for o, nm in zip(v_tkm.outputs, ("tv", "ti")):
            o.get_tensor().set_names({nm})
        v_tkc = core.compile_model(v_tkm, DEVICE)
        tkreq = v_tkc.create_infer_request()
        # SC8: topk reads RAW logits (order is temperature-invariant; host scales [256,64])
        tkreq.set_tensor("x", lg_rt if SC8 else sc_rt)

        _tprof = {"g": 0.0, "m": 0.0, "n": 0}   # in-loop tail split: gemm vs scale/topk

        if SC8:
            def _chain8(temp):
                """gemm -> 2-stage topk on RAW logits; caller scales by temp on host."""
                _t0 = time.time()
                greq.start_async(); greq.wait()
                _t1 = time.time()
                tkreq.start_async(); tkreq.wait()
                tv = np.array(tkreq.get_tensor("tv").data.reshape(CANVAS, K_S))
                ti = np.array(tkreq.get_tensor("ti").data.reshape(CANVAS, K_S)).astype(np.int64)
                _tprof["g"] += _t1 - _t0; _tprof["m"] += time.time() - _t1; _tprof["n"] += 1
                return tv / temp, ti
            _chain = _chain8
        else:
            def _chain3(temp):
                _t0 = time.time()
                greq.start_async(); greq.wait()
                _t1 = time.time()
                _run(sreq, {"temp": np.array([temp], np.float32)})
                tkreq.start_async(); tkreq.wait()
                tv = np.array(tkreq.get_tensor("tv").data.reshape(CANVAS, K_S))
                ti = np.array(tkreq.get_tensor("ti").data.reshape(CANVAS, K_S)).astype(np.int64)
                _tprof["g"] += _t1 - _t0; _tprof["m"] += time.time() - _t1; _tprof["n"] += 1
                return tv, ti
            _chain = _chain3

        # zero-warm the chain (also leaves finite values in sc_rt for the first masked step)
        lh_rt.copy_from(ov.Tensor(np.zeros((1, CANVAS, HID), np.float32)))
        _chain(1.0)

        try:
            if SC8:
                raise StopIteration   # no divide stage exists; nothing to merge
            # v4: divide+topk merged into ONE model (2 round trips instead of 3). The slow-plan
            # pathology needs the gemm in the same graph; divide+topk alone plans fine.
            mx = ops.parameter(ov.PartialShape([1, CANVAS, VOCAB]), ov.Type.f32, name="x")
            mt = ops.parameter(ov.PartialShape([1]), ov.Type.f32, name="temp")
            mdiv = ops.divide(mx.output(0), mt.output(0))
            m_tv, m_ti = _topk_tail(mdiv.output(0), " (merged)")
            m4 = ov.Model([ops.result(mdiv.output(0)), ops.result(m_tv),
                           ops.result(m_ti)], [mx, mt], "sc_topk")
            for o, nm in zip(m4.outputs, ("y", "tv", "ti")):
                o.get_tensor().set_names({nm})
            m4c = core.compile_model(m4, DEVICE)
            mreq = m4c.create_infer_request()
            mreq.set_tensor("x", lg_rt)
            mreq.set_tensor("y", sc_rt)
            def _chain2(temp):
                _t0 = time.time()
                greq.start_async(); greq.wait()
                _t1 = time.time()
                _run(mreq, {"temp": np.array([temp], np.float32)})
                tv = np.array(mreq.get_tensor("tv").data.reshape(CANVAS, K_S))
                ti = np.array(mreq.get_tensor("ti").data.reshape(CANVAS, K_S)).astype(np.int64)
                _tprof["g"] += _t1 - _t0; _tprof["m"] += time.time() - _t1; _tprof["n"] += 1
                return tv, ti
            _chain2(1.0)                                   # warm
            t0 = time.time()
            for _ in range(3): _chain3(1.0)
            t3ms = (time.time() - t0) / 3 * 1000
            t0 = time.time()
            for _ in range(3): _chain2(1.0)
            t2ms = (time.time() - t0) / 3 * 1000
            print(f"tail chain: 3-model {t3ms:.0f}ms vs merged {t2ms:.0f}ms -> using {'merged' if t2ms < t3ms else '3-model'}")
            if t2ms < t3ms:
                _chain = _chain2
        except StopIteration:
            pass                                              # SC8: no merged stage
        except Exception as me:
            print(f"merged sc+topk unavailable ({repr(me)[:100]}) -> 3-model chain")
        # parity probe vs the proven host path (top-64 SET equality per row + value error)
        rng = np.random.default_rng(0)
        h_probe = (rng.standard_normal((1, CANVAS, HID)) * 0.05).astype(np.float32)
        lh_rt.copy_from(ov.Tensor(h_probe))
        tv, ti = _chain(1.0)
        ref = lm_logits(h_probe)
        ref_ti = np.argpartition(ref, -K_S, axis=-1)[:, -K_S:]
        overlap = np.array([len(np.intersect1d(ti[r], ref_ti[r])) for r in range(CANVAS)])
        verr = float(np.max(np.abs(np.sort(tv, -1) - np.sort(np.take_along_axis(ref, ref_ti, -1), -1))))
        print(f"v3 chain parity: top{K_S} set overlap min {overlap.min()}/{K_S} | value err {verr:.2e}")
        if overlap.min() < K_S - 2 or verr > 5e-3:
            raise RuntimeError(f"v3 parity FAILED (overlap {overlap.min()}, err {verr:.2e})")
        breq.set_tensor("last_hidden", lh_rt)                 # backbone writes the chain input
        d_w = _rung(8, CANVAS, True)                          # warm breq with the remote bindings
        if not SC8:
            breq.set_tensor("self_conditioning_logits", sc_rt)  # backbone reads scaled logits as SC
            del d_w["self_conditioning_logits"]               # (keep sc_rt bound, don't override)
        _run(breq, d_w)
        V3 = True
        print("v3 tail armed: device-resident gemm/scale/topk chain (host gets [256,64] only)")
    except Exception as e:
        print(f"V3 TAIL DISARMED ({repr(e)[:160]}) -> hybrid tail")
        try:
            breq.set_tensor("last_hidden", ov.Tensor(ov.Type.f32, ov.Shape([1, CANVAS, HID])))
        except Exception:
            pass
        V3 = False

if FAST and PROF:
    # one-shot micro-bench of device-TopK candidates (informational; serving uses the hybrid).
    # Measured 2026-07-11: lm128x2+download 60ms | lmf256 sorted-topk 1137ms | unsorted CRASHES.
    try:
        def _bench(tag, fn, n=3):
            fn(); t0 = time.time()
            for _ in range(n): fn()
            print(f"    [bench] {tag}: {(time.time()-t0)/n*1000:.0f} ms")
        rng = np.random.default_rng(0)
        hb = (rng.standard_normal((1, CANVAS, HID)) * 0.05).astype(np.float32)
        _bench("lm128x2 (legacy tail, full logits down)", lambda: lm_logits(hb))
        sc_bench = rctx.create_tensor(ov.Type.f32, ov.Shape([1, CANVAS, VOCAB]), {})
        tk_in = ops.parameter(ov.PartialShape([1, CANVAS, VOCAB]), ov.Type.f32, name="x")
        tk2 = ops.topk(tk_in.output(0), ops.constant(np.int64(K_S)).output(0),
                       2, "max", "value", index_element_type=ov.Type.i32)
        tkm = ov.Model([ops.result(tk2.output(0)), ops.result(tk2.output(1))], [tk_in], "tk_only")
        for o, nm in zip(tkm.outputs, ("tv", "ti")):
            o.get_tensor().set_names({nm})
        b_tkc = core.compile_model(tkm, DEVICE)
        b_tkreq = b_tkc.create_infer_request()
        b_tkreq.set_tensor("x", sc_bench)      # device-resident input: pure TopK cost
        def _tk():
            b_tkreq.start_async(); b_tkreq.wait()
        _bench("topk256-only sorted (input stays device)", _tk)
        b_g256 = lm_model.clone()
        b_g256.reshape({b_g256.inputs[0]: ov.PartialShape([1, CANVAS, HID])})
        b_gm = core.compile_model(_fp32_io(b_g256), DEVICE, {"INFERENCE_PRECISION_HINT": "f32"})
        b_greq = b_gm.create_infer_request()
        _bench("gemm256-only (full logits down)", lambda: _run(b_greq, {"last_hidden_in": hb}))
        del b_tkc, b_tkreq, b_gm, b_greq, sc_bench
    except Exception as be:
        print(f"    [bench] extras failed: {repr(be)[:120]}")

# ---- KV cache: per-layer [1,h,S,d] fp32; DUMMY masked dummy prefix (see knobs above) ----
def zeros_kv(P):
    kv = []
    for lt in LAYER_TYPES:
        h, d = kv_hd(lt); kv += [np.zeros((1, h, P, d), np.float32), np.zeros((1, h, P, d), np.float32)]
    return kv
cache = zeros_kv(DUMMY); cache_len = DUMMY

def _sc_zero_feed(L):
    """SC feed for encoder passes / step 1 (mask 0 or zeros = the step-1 zeros path)."""
    if SC8:
        return {"sc_top_p": np.zeros((1, L, 8), np.float32),
                "sc_top_i": np.zeros((1, L, 8), np.int64)}
    return {"self_conditioning_logits": np.zeros((1, L, VOCAB), np.float32)}

def infer(current_ids, sc_feed, apply_sc, sc_mask, full_m, slid_m):
    base = cache_len - DUMMY
    L = current_ids.shape[1]
    pos = np.arange(base, base + L)[None].astype(np.int64)
    d = {"current_ids": current_ids.astype(np.int64),
         "apply_sc": np.array([apply_sc], np.float32),
         "position_ids": pos, "full_mask": full_m.astype(np.float32),
         "sliding_mask": slid_m.astype(np.float32)}
    d.update(sc_feed)
    if HAS_SCMASK:
        d["self_conditioning_mask"] = np.array([sc_mask], np.float32)
    for i in range(N_LAYERS):
        d[f"prefix_key_{i}"] = cache[2 * i]; d[f"prefix_value_{i}"] = cache[2 * i + 1]
    return ir(d)

def _positions(L):
    """absolute positions of (cache columns, new columns, new queries)"""
    base = cache_len - DUMMY
    k_pos = np.concatenate([np.arange(cache_len) - DUMMY, base + np.arange(L)])
    q_pos = base + np.arange(L)
    return k_pos, q_pos

def _windowed(full, L):
    """sliding variant: NEG where the key is >= SLIDING positions behind the query
    (bidirectional/canvas keys ahead of the query have negative distance -> always visible)"""
    k_pos, q_pos = _positions(L)
    slid = full.copy()
    slid[0, 0][(q_pos[:, None] - k_pos[None, :]) >= SLIDING] = NEG
    return slid

def causal_mask(L):
    """(full, sliding) [1,1,L, cache_len+L]: attend cache (dummies masked) + causal among new."""
    full = np.zeros((1, 1, L, cache_len + L), np.float32)
    full[..., :DUMMY] = NEG
    nc = np.where(np.arange(L)[None] <= np.arange(L)[:, None], 0.0, NEG).astype(np.float32)
    full[0, 0, :, cache_len:] = nc
    return full, _windowed(full, L)

def bidir_mask(L):
    full = np.zeros((1, 1, L, cache_len + L), np.float32); full[..., :DUMMY] = NEG
    return full, _windowed(full, L)

def append_kv(new):
    global cache, cache_len
    for i in range(N_LAYERS):
        cache[2 * i]     = np.concatenate([cache[2 * i],     new[2 * i]],     axis=2)
        cache[2 * i + 1] = np.concatenate([cache[2 * i + 1], new[2 * i + 1]], axis=2)
    cache_len = cache[0].shape[2]

PAD_ID = TOK.get("pad") or TOK.get("eos") or 0

def encode(ids):
    """ENCODER pass, chunk-aware wrapper: C=24 capacity exports overflow their static
    expert slots above ENC_CHUNK tokens per forward (drops corrupt the prefix KV), so
    long prompt encodes split into causal chunks against the growing cache. Exact math:
    chunk k attends chunks <k through prefix KV, positions/masks per chunk are the same
    as any committed-block encode. ENC_CHUNK=0 (no manifest key) = single-shot as before."""
    n = ids.shape[1]
    if not ENC_CHUNK or n <= ENC_CHUNK:
        return _encode_one(ids)
    for s in range(0, n, ENC_CHUNK):
        _encode_one(ids[:, s:s + ENC_CHUNK])

def _encode_one(ids):
    """One causal encode forward: apply_sc=0 -> new per-layer KV appended to cache.
    Pads L up to >=PAD_FLOOR, PAD_ALIGN-aligned (real rows never attend pads; pad KV sliced off)."""
    true_L = ids.shape[1]
    L = ((max(true_L, PAD_FLOOR) + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
    if L != true_L:
        ids = np.concatenate([ids, np.full((1, L - true_L), PAD_ID, np.int64)], axis=1)
    cmf, cms = causal_mask(L)
    if L != true_L:
        cmf[0, 0, :true_L, cache_len + true_L:] = NEG
        cms[0, 0, :true_L, cache_len + true_L:] = NEG
    out = infer(ids, _sc_zero_feed(L), 0.0, 0.0, cmf, cms)
    new = []
    for i in range(N_LAYERS):
        new += [out[ir.output(f"new_key_{i}")][:, :, :true_L, :],
                out[ir.output(f"new_value_{i}")][:, :, :true_L, :]]
    append_kv(new)

def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))

def _sample_step(top_v, top_i, locked, committed):
    """Shared lock/sample math on [CANVAS, K] scaled logits. Returns (denoiser, new_argmax, H)."""
    rows = np.arange(CANVAS)
    if FREQPEN and locked.any():                          # break repetition spirals: penalize
        cnt = np.bincount(committed[locked], minlength=VOCAB).astype(np.float32)
        if EOS is not None: cnt[EOS] = 0.0                # EOS/pad repeat by design (canvas padding)
        cnt[PAD_ID] = 0.0
        top_v = top_v - FREQPEN * np.maximum(cnt[top_i] - FREQ_FLOOR, 0.0)
    if ADJPEN and locked.any():                           # adjacent-duplicate penalty (see knob)
        ln = np.concatenate([[-1], committed[:-1]])       # locked left-neighbor token per row
        lm = np.concatenate([[False], locked[:-1]])
        rn = np.concatenate([committed[1:], [-1]])        # locked right-neighbor token per row
        rm = np.concatenate([locked[1:], [False]])
        dup = ((top_i == ln[:, None]) & lm[:, None]) | ((top_i == rn[:, None]) & rm[:, None])
        if EOS is not None:
            dup &= top_i != EOS
        dup &= top_i != PAD_ID
        top_v = top_v - ADJPEN * dup
    lp = top_v - top_v.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    p = np.exp(lp)
    H = -(p * lp).sum(-1)                                 # token entropy (nats, top-K mass)
    g = -np.log(-np.log(np.random.uniform(size=top_v.shape).clip(1e-20, 1.0)))
    denoiser = top_i[rows, (top_v + g).argmax(-1)]        # categorical via Gumbel-max on top-K
    new_argmax = top_i[rows, top_v.argmax(-1)]
    return denoiser, new_argmax, H

def _lock_trim_loop(get_logits):
    """Core sticky denoise loop; get_logits(canvas, sc_ready, temp) -> (top_v, top_i) [CANVAS,K]
    scaled. sc feedback is handled by the caller (host array or device remote)."""
    global _seed_ids
    canvas = np.random.randint(0, VOCAB, (CANVAS,)).astype(np.int64)
    locked = np.zeros(CANVAS, bool)
    committed = np.zeros(CANVAS, np.int64)
    seed_n = 0
    if _seed_ids:                                 # canvas seeding (block 0): reply MUST start here
        seed_n = min(len(_seed_ids), 32)
        committed[:seed_n] = _seed_ids[:seed_n]
        locked[:seed_n] = True
        canvas[:seed_n] = committed[:seed_n]
        _seed_ids = None
    new_argmax = canvas; total_rev = 0
    W = 16; last_c = -1; stable = 0
    for k, step in enumerate(range(MAX_STEPS, 0, -1)):        # cur_step counts DOWN
        temp = T_MIN + (T_MAX - T_MIN) * (step / MAX_STEPS)
        top_v, top_i = get_logits(canvas, k > 0, temp)
        denoiser, new_argmax, H = _sample_step(top_v, top_i, locked, committed)
        rev = locked & (new_argmax != committed) & (H < REV_H)  # confident disagreement -> unlock
        if seed_n:
            rev[:seed_n] = False                  # seeds are non-negotiable
        if rev.any():
            locked[rev] = False; total_rev += int(rev.sum())
        if k >= WARM:                                         # no locking until SC signal forms
            cand = np.where(~locked)[0]
            order = cand[np.argsort(H[cand])]
            cum = np.cumsum(H[order])
            newly = order[(cum - H[order]) <= EBOUND]
            if len(newly) == 0 and len(order):                # always make progress
                newly = order[:1]
            committed[newly] = denoiser[newly]; locked[newly] = True
        renoise = np.random.randint(0, VOCAB, (CANVAS,))
        canvas = np.where(locked, committed, renoise).astype(np.int64)
        if locked.all():
            print(f"    [denoise] all {CANVAS} locked at step {k+1}/{MAX_STEPS} (mean H={H.mean():.4f})")
            break
        if MIN_STEPS and k + 1 >= MIN_STEPS:
            if EOS is not None:                               # prefix-complete: reply exists in full
                e_lk = np.where(locked & (committed == EOS))[0]
                if len(e_lk) and locked[:e_lk[0]].all():
                    print(f"    [denoise] prefix-complete stop at step {k+1}/{MAX_STEPS} "
                          f"(EOS locked at {e_lk[0]}, prefix fully locked)")
                    break
            # boundary-stall: the confident prefix is solid and stopped growing
            dens = np.convolve(locked.astype(np.float32), np.ones(W) / W, mode="valid")
            cc = np.where(dens < 0.5)[0]
            c = int(cc[0]) if len(cc) else CANVAS
            if c >= W and int((~locked[:c]).sum()) <= 2:
                stable = stable + 1 if c == last_c else 1
                last_c = c
                if stable >= STABLE_N:
                    print(f"    [denoise] boundary-stall stop at step {k+1}/{MAX_STEPS} "
                          f"(prefix {c} solid, stable x{stable})")
                    break
            else:
                stable = 0; last_c = c
    n_locked = int(locked.sum())
    committed = np.where(locked, committed, new_argmax)       # argmax-fill any stragglers
    # confident-prefix trim: when the model runs out of content it stops locking and the
    # argmax-filled tail degenerates (repetition spirals / junk). Lock density marks the
    # boundary; everything past the first collapsed window is untrusted.
    W = 16
    dens = np.convolve(locked.astype(np.float32), np.ones(W) / W, mode="valid")
    collapse = np.where(dens < 0.5)[0]
    trim = int(collapse[0]) if len(collapse) else CANVAS
    if DIAG:
        cnt = np.bincount(committed, minlength=VOCAB)
        top = np.argsort(cnt)[::-1][:8]
        print(f"    [diag] locked {n_locked}/{CANVAS} | revisions {total_rev} | mean H last step {H.mean():.3f} | top committed: " +
              ", ".join(f"{tok.decode([int(t)])!r}x{int(cnt[t])}" for t in top if cnt[t] > 1))
        rep = int((committed[1:] == committed[:-1]).sum())
        print(f"    [diag] adjacent-duplicate tokens: {rep}/{CANVAS-1} | confident-prefix trim at {trim}/{CANVAS}")
    return committed, trim

def denoise():
    """DECODER denoise loop over a CANVAS noise canvas (legacy host path).

    STICKY entropy-bound acceptance (arXiv 2505.24857): each step locks the lowest-entropy
    unlocked positions within the EBOUND nats budget, PERMANENTLY. Only never-locked
    positions are re-noised."""
    fmf, fms = bidir_mask(CANVAS)
    sc_holder = {"feed": _sc_zero_feed(CANVAS)}
    def get_logits(canvas, sc_ready, temp):
        out = infer(canvas[None], sc_holder["feed"], 1.0, 1.0 if sc_ready else 0.0, fmf, fms)
        raw = lm_logits(out[ir.output("last_hidden")])        # [CANVAS, V] softcapped fp32
        processed = raw / temp
        top_i = np.argpartition(processed, -K_S, axis=-1)[:, -K_S:]      # [CANVAS, K_S]
        top_v = np.take_along_axis(processed, top_i, axis=-1).astype(np.float32)
        if SC8:
            i8 = np.argpartition(processed, -8, axis=-1)[:, -8:]
            v8 = np.take_along_axis(processed, i8, axis=-1)
            e = np.exp(v8 - v8.max(-1, keepdims=True))
            sc_holder["feed"] = {"sc_top_p": (e / e.sum(-1, keepdims=True)).astype(np.float32)[None],
                                 "sc_top_i": i8.astype(np.int64)[None]}
        else:
            sc_holder["feed"] = {"self_conditioning_logits": processed[None]}
        return top_v, top_i
    return _lock_trim_loop(get_logits)

def denoise_fast():
    """Stage-2 HYBRID denoise: prefix KV uploaded once per block as RemoteTensors and the
    60 unused new-KV outputs sinked on device (the two big per-step thieves); the lm tail
    stays the proven M=128 pair with host top-K (60ms incl. the full logits download --
    measured CHEAPER than any Arc device-TopK variant: sorted TopK ~1s, unsorted crashes).
    SC feedback returns host->device as a per-step input upload (~268MB, the remaining cost;
    kill it later with a 2-stage device TopK if the PROF bench proves it fast)."""
    fmf, fms = bidir_mask(CANVAS)
    base = cache_len - DUMMY
    # prefix KV -> device once (constant for the whole block)
    kv_rt = []
    for i in range(N_LAYERS):
        for j in (0, 1):
            src = np.ascontiguousarray(cache[2 * i + j])
            rt = rctx.create_tensor(ov.Type.f32, ov.Shape(list(src.shape)), {})
            rt.copy_from(ov.Tensor(src))
            kv_rt.append(rt)
            breq.set_tensor(f"prefix_{'key' if j == 0 else 'value'}_{i}", rt)
        breq.set_tensor(f"new_key_{i}", kvout_rt[i][0])    # unused sinks stay on device
        breq.set_tensor(f"new_value_{i}", kvout_rt[i][1])
    for nm, arr in (("full_mask", fmf), ("sliding_mask", fms)):   # masks device-resident too
        rt = rctx.create_tensor(ov.Type.f32, ov.Shape(list(arr.shape)), {})
        rt.copy_from(ov.Tensor(np.ascontiguousarray(arr)))
        kv_rt.append(rt)
        breq.set_tensor(nm, rt)
    breq.set_tensor("apply_sc", ov.Tensor(np.array([1.0], np.float32)))
    breq.set_tensor("position_ids",
                    ov.Tensor(np.arange(base, base + CANVAS)[None].astype(np.int64)))
    sc_host = None if SC8 else np.zeros((1, CANVAS, VOCAB), np.float32)
    sc_p8 = np.zeros((1, CANVAS, 8), np.float32)          # SC8 feedback (10KB/step)
    sc_i8 = np.zeros((1, CANVAS, 8), np.int64)
    t_bb = t_lm = t_host = 0.0
    g0 = dict(_tprof) if V3 else None
    def _sc8_stash(scaled_v, idx):
        """next-step SC from the top-8 of the UNPENALIZED scaled logits (reference
        semantics: SC sees softmax(logits/temp); penalties only shape sampling)."""
        nonlocal sc_p8, sc_i8
        v8 = scaled_v[:, :8] if scaled_v.shape[1] >= 8 else scaled_v   # tk2 output is sorted desc
        e = np.exp(v8 - v8.max(-1, keepdims=True))
        sc_p8 = (e / e.sum(-1, keepdims=True)).astype(np.float32)[None]
        sc_i8 = idx[:, :8].astype(np.int64)[None]
    def get_logits(canvas, sc_ready, temp):
        nonlocal t_bb, t_lm, t_host, sc_host
        t0 = time.time()
        breq.set_tensor("current_ids", ov.Tensor(canvas[None].astype(np.int64)))
        if SC8:
            breq.set_tensor("sc_top_p", ov.Tensor(sc_p8))
            breq.set_tensor("sc_top_i", ov.Tensor(sc_i8))
        elif not V3:
            breq.set_tensor("self_conditioning_logits", ov.Tensor(sc_host))
        if HAS_SCMASK:
            breq.set_tensor("self_conditioning_mask",
                            ov.Tensor(np.array([1.0 if sc_ready else 0.0], np.float32)))
        breq.start_async(); breq.wait()
        t1 = time.time(); t_bb += t1 - t0
        if V3:                                             # device chain: gemm -> (scale) -> topk
            top_v, top_i = _chain(temp)
            t_lm += time.time() - t1
            if SC8:
                _sc8_stash(top_v, top_i)                   # tk2 rows are sorted desc
            return top_v, top_i
        h = breq.get_tensor(ir.output("last_hidden")).data
        raw = lm_logits(np.ascontiguousarray(h, dtype=np.float32))
        t2 = time.time(); t_lm += t2 - t1
        processed = raw / temp
        top_i = np.argpartition(processed, -K_S, axis=-1)[:, -K_S:]
        top_v = np.take_along_axis(processed, top_i, axis=-1).astype(np.float32)
        if SC8:
            order = np.argsort(top_v, axis=-1)[:, ::-1]    # argpartition is unsorted
            _sc8_stash(np.take_along_axis(top_v, order, -1),
                       np.take_along_axis(top_i, order, -1))
        else:
            sc_host = processed[None]                      # SC feedback (uploaded next step)
        t_host += time.time() - t2
        return top_v, top_i
    out = _lock_trim_loop(get_logits)
    if PROF:
        extra = ""
        if V3 and g0 is not None and _tprof["n"] > g0["n"]:
            n = _tprof["n"] - g0["n"]
            extra = (f" | gemm {(_tprof['g']-g0['g'])/n*1000:.0f}ms/step"
                     f" + sc/topk {(_tprof['m']-g0['m'])/n*1000:.0f}ms/step x{n}")
        print(f"    [prof] backbone {t_bb:.2f}s | tail {t_lm:.2f}s | host-math {t_host:.2f}s (block, v3={V3}){extra}")
    return out

# ================= GENERATE =================
def reset_cache():
    global cache, cache_len
    cache = zeros_kv(DUMMY); cache_len = DUMMY

if os.environ.get("DG_ENCCHECK") == "1":
    # chunked-vs-oneshot prefill parity (the OV PR#36708 invariant). Runs on top of a
    # 256-token filler cache so every forward's total stays in the proven >=300 warm band
    # (bare totals 96-208 page-faulted the tail probe). The compared lengths (96+64 vs 160)
    # keep even a C=24 model under capacity in BOTH paths, so any difference is chunk
    # bookkeeping (positions/masks/KV offsets), not intended capacity drops. Report-only.
    _rng = np.random.default_rng(3)
    _fill = _rng.integers(0, VOCAB, (1, 256)).astype(np.int64)
    _tids = _rng.integers(0, VOCAB, (1, 160)).astype(np.int64)
    reset_cache(); _encode_one(_fill)
    _base = [a.copy() for a in cache]; _base_len = cache_len
    _encode_one(_tids[:, :96]); _encode_one(_tids[:, 96:])
    _kv_c = [a.copy() for a in cache]
    cache = [a.copy() for a in _base]; cache_len = _base_len
    _encode_one(_tids)
    _err = max(float(np.abs(_kv_c[i] - cache[i]).max()) for i in range(len(cache)))
    _ref = max(float(np.abs(a).max()) for a in cache)
    reset_cache()
    print(f"    [encheck] chunked-vs-oneshot prefill KV: max abs err {_err:.2e} "
          f"(KV magnitude {_ref:.1f}) over 160 tok on a 256-tok base")

def generate(prompt, max_blocks=None):
    """prompt: str (single user turn) or list of OpenAI-style {role, content} messages."""
    reset_cache()
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    try:
        try:
            text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=THINK)
        except TypeError:
            text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        style = os.environ.get("DG_THOUGHTSTYLE", "none")  # none (proven) | channel (AR-style; derails personas) | plain
        if not THINK and style == "channel":
            text += "<|channel>thought\n<channel|>"   # AR gemma-4 pre-closed thought channel
        elif not THINK and style == "plain":
            text += "thought\n\n"                     # emulate an empty plain thought block
        ids = tok(text, add_special_tokens=False)["input_ids"]
    except Exception as e:
        print("chat template unavailable, raw encode:", repr(e)[:80])
        ids = tok(str(prompt))["input_ids"]
    if len(ids) > MAX_PROMPT_TOK:                     # keep persona head + the template's model-turn tail
        ids = ids[:MAX_PROMPT_TOK - 64] + ids[-64:]
        print(f"  prompt truncated to {len(ids)} tokens (safe-envelope cap)")
    prompt_ids = np.array(ids, np.int64)[None]
    print(f"prompt: {str(prompt)[:70]!r} -> {prompt_ids.shape[1]} tokens")
    t0 = time.time(); encode(prompt_ids); print(f"prefill in {time.time()-t0:.2f}s (cache_len={cache_len})")
    global _seed_ids
    _seed_ids = tok(SEED_TEXT, add_special_tokens=False)["input_ids"] if SEED_TEXT else None
    out_tokens = []; trimmed = False
    for blk in range(max_blocks or MAX_NEW_BLOCKS):
        t0 = time.time()
        committed, trim = denoise_fast() if FAST else denoise()
        print(f"  block {blk}: {time.time()-t0:.1f}s")
        cut = trim                                   # lock-collapse = model ran out of content
        if EOS is not None:
            e = np.where(committed[:cut] == EOS)[0]
            if len(e): cut = int(e[0]) + 1
        out_tokens.extend(committed[:cut].tolist())
        if cut < CANVAS:
            hit_eos = EOS is not None and cut and committed[cut - 1] == EOS
            trimmed = not hit_eos
            print(f"  {'EOS' if hit_eos else 'lock-collapse trim'} -> stop")
            break
        if cache_len + 2 * CANVAS > MAX_TOTAL_TOK:    # memory ceiling (dispatch buffers scale with T)
            print("  total-length ceiling -> stop")
            break
        encode(committed[None].astype(np.int64))
    text_out = tok.decode(out_tokens, skip_special_tokens=True)
    if SEED_TEXT:                                    # model often rewrites the seed opener: "(I (I stand" -> "(I stand"
        st = SEED_TEXT.strip()
        if st and text_out.startswith(f"{st} {st}"):
            text_out = text_out[len(st) + 1:]
    if trimmed:                                      # snap a mid-sentence trim to the last sentence end
        m = max(text_out.rfind(c) for c in ".!?\n")
        if m > len(text_out) // 2:
            text_out = text_out[:m + 1].rstrip()
    return text_out

if __name__ == "__main__":
    prompts = ([l.strip() for l in open(os.path.expanduser(PROMPT[1:])) if l.strip()]
               if PROMPT.startswith("@") else [PROMPT])
    for pr in prompts:
        txt = generate(pr)
        print(f"\n===== GENERATED ({pr[:50]!r}) =====\n{txt}\n=====================")
