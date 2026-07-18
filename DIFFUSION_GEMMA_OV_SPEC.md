# DiffusionGemma 26B-A4B -> OpenVINO Port Spec (2026-07-07)

Target: DuoNeural/diffusiongemma-26B-A4B-it-abliterated (base: google/diffusiongemma-26B-A4B-it)
Goal: FIRST diffusion LLM running on OpenVINO / Intel Arc. Hand-rolled sampler over exported IRs
(same pattern as the 12B AV pipeline). Modeling source: transformers main,
src/transformers/models/diffusion_gemma/ (modeling 1689 lines, generation 1327 lines).

## Checkpoint facts (read from the repo config, 2026-07-07)

- canvas_length 256 (the diffusion block), use_bidirectional_attention: "vision" (text encoder stays causal)
- hidden 2816, 30 layers, 16 heads / 8 kv, head_dim 256; GLOBAL layers: global_head_dim 512, only 2 global kv heads
- Hybrid MoE: 128 experts top-8 (moe_intermediate 704) PLUS a dense MLP (intermediate 2112) in EVERY layer, summed
- sliding_window 1024, 5:1 sliding:full layer pattern
- RoPE: full_attention = proportional, partial 0.25, theta 1e6 (OUR LUT-PATCH TERRITORY, fp16 wall expected);
  sliding = default theta 10k
- vocab 262144, final_logit_softcapping 30.0, weights tied (lm_head = embeddings)
- NO generation_config.json -> sampler defaults from code (below)

## Architecture (the surprise)

NOT masked diffusion, NOT single-stack. Encoder-decoder with tied weights:
- ENCODER (causal, standard Gemma-style): its ONLY inference job is writing the prefix KV cache.
  Runs once per block (prefill: whole prompt; later: the 256 tokens just committed).
- DECODER (fully bidirectional, is_causal=False everywhere, even sliding layers see everything):
  denoises a 256-token canvas up to 48x per block, reading the cache WITHOUT writing it
  (out-of-place concat per step). Sliding layers only see the last sliding_window prefix keys.
- Noise = uniform random token ids over the full vocab. There is NO mask token.
- Self-conditioning: previous step's temperature-scaled logits -> fp32 softmax -> @ embedding matrix
  (scaled) -> SC MLP (gate/up/down with pre/post RMSNorm-no-scale) added to token embeddings.
  Step 1 uses zeros for the soft embeddings; post_norm STILL applies. Export knob: keep the SC input
  always present + a (B,) sc_mask input; mask=0 exactly reproduces step 1 (never feed zero logits:
  softmax(0) is uniform).
- Committed block = the ARGMAX canvas, then it is re-forwarded through the ENCODER causally to
  produce cache entries (decoder K/V are bidirectional, unusable for the cache; encoder pass is
  mandatory each block).

## The sampler (reimplement exactly, all numpy between forwards)

Defaults: max_denoising_steps 48, entropy_bound 0.1, t_min 0.4, t_max 0.8,
stability_threshold 1, confidence_threshold 0.005, canvas 256.

Per block: canvas = randint(0, V, (B, 256)); sc_logits = None; then per step (cur_step counts DOWN N..1):
1. temperature = t_min + (t_max - t_min) * (cur_step / N); processed = raw_logits / temperature
   (raw logits already softcapped in-graph: tanh(l/30)*30, fp32)
2. probs = softmax(processed, fp32); denoiser_canvas = per-position categorical sample
3. new_argmax = argmax(processed)
4. ENTROPY-BOUND ACCEPTANCE (arXiv 2505.24857): H_i per position (fp32);
   sort ascending; cum = cumsum(sorted_H); accept sorted positions where (cum - sorted_H) <= 0.1;
   unsort mask back. At least the lowest-entropy position always accepted.
   accepted = where(mask, denoiser_canvas, current_canvas)
5. RENOISE: rejected positions get FRESH randint(0, V) every step (not the sampled token)
6. Frozen rows (finished) keep old canvases/logits
7. ADAPTIVE STOP per row: stable (argmax canvas identical to previous step, ring buffer depth 1)
   AND confident (mean token entropy < 0.005) -> finished; break when all finished
8. sc_logits = processed (temperature-scaled!), cast to embed dtype, feeds next step
Commit argmax canvas; EOS scan on the committed block (keep first EOS, pad after);
positions: encoder pos = arange over its tokens, decoder pos = arange(cur_len, cur_len+256).

## Export plan (two stateless IRs, numpy owns cache + sampler)

IR-E (cache builder, once per block): input_ids (B,S_dyn), position_ids, additive 4D masks
  (full + sliding), per-layer past k/v (B,H_kv,P_dyn,D) -> per-layer NEW k/v for the S new tokens only.
IR-D (denoiser, <=48x per block, static except prefix dim): decoder_input_ids (B,256),
  sc_logits (B,256,V), sc_mask (B,), decoder_position_ids, per-layer prefix k/v ->
  logits (B,256,V) fp32 with softcap in-graph.
Stateless v1 is fine: per-step compute is a 256-token forward (prefill-class), KV copy overhead modest.
v2 optimization = keep prefix KV device-resident (states/remote tensors), not stateful export.

## Export risks (agent-verified against source)

1. MoE experts loop is untraceable (nonzero + python loop): replace with dense all-experts
   grouped-matmul formulation before trace (same class of fix as optimum's gemma-4 MoE).
   Router itself traces fine (RMSNorm-no-scale -> *scale*H^-0.5 -> linear -> fp32 softmax ->
   top8 -> renorm -> * per_expert_scale gather).
2. EVERY layer = dense MLP + MoE branch SUMMED with extra norms (post_ffw_norm(post1(mlp(pre(x)))
   + post2(experts(pre2(x))))) + layer_scalar buffer multiply. Do not reuse a vanilla converter blindly.
3. GLOBAL layers: v_proj IS NONE, values = k_proj output with v-norm (shared KV projection);
   keys additionally get k_norm + rope. scaling = 1.0 (NOT head_dim^-0.5). QK-norm + V-norm(no scale).
4. Per-layer-type rope buffers; proportional rope on global layers (LUT patch transfers).
5. Always pass explicit cache tensors (encoder builds DynamicCache internally if none).
6. SC matmul (B,256,262k)@(262k,2816) runs EVERY step, lm_head cost class, keep fp16.
7. Softcap stays in-graph. RMSNorm fp32 upcast. Embedding scale hidden^0.5 (also on SC matmul).
8. Text-only export: skip vision tower + masked_scatter path entirely.

## Pipeline plan

Phase 1 (Colab A100, proven flow): load bf16 (50GB), monkeypatch experts to dense form,
  trace/export IR-E + IR-D f16, NNCF int4 (AWQ + ignored_scope router/gates like the 26B recipe),
  upload artifact to HF private.
Phase 2 (box): numpy sampler (this spec section 'sampler'), validate vs Colab-side HF reference
  generations (same seed impossible across frameworks; validate distributions/quality instead),
  LUT-patch global-layer rope if the fp16 wall shows (expect it: theta 1e6 partial 0.25 = the 26B case).
Phase 3: bench (tokens/sec effective = 256 * blocks / wall; adaptive stopping does heavy lifting),
  publish model + writeup as the first diffusion LLM on OpenVINO/Arc.

Cost estimate per 256-token block on B70 int4: <=48 denoise steps x 256-token forward
(prefill-class ~85-120ms at short prefix) + 1 encoder pass. Ballpark 4-8s/block worst case,
much less with adaptive stopping -> effective 30-60+ tok/s plausible. Competitive with AR decode.

## VERIFIED CORRECTIONS + REVIEW FIXES (2026-07-07)

The original spec above was misleading in four places. All four are corrected below, verified
against modeling/modular/generation/convert_diffusion_gemma_weights.py, and folded into the
finalized export script (COLAB_DiffusionGemma.txt).

### Four verified spec corrections
1. Sliding layers have 8 KV heads, NOT 4. head_dim 256, num_key_value_heads 8 (groups = 16/8 = 2),
   separate k_proj + v_proj. The `4` in config is a stale library default for a different-sized block
   (2304/8heads); this checkpoint is 2816/16heads and convert L102-104 gives 8. KV is heterogeneous:
   sliding = [B, 8, S, 256], full/global = [B, 2, S, 512].
2. Self-conditioning pre_norm HAS a learnable scale; only post_norm is with_scale=False (no weight).
   The wiring is post_norm(inputs_embeds + SC_MLP(pre_norm(soft_emb))) (modeling L1286 / L822-824), so
   the ENTIRE embedding sum is renormalized every step by a no-scale RMSNorm. It is NOT "SC added on top
   of already-normed embeddings". Do not look for post_norm.weight (it does not exist); pre_norm.weight
   does exist and loads from the checkpoint.
3. IR-D must be traced as DiffusionGemmaForBlockDiffusion.forward with input_ids=None (encoder skipped at
   modeling L1549), which includes lm_head + the fp32 final softcap tanh(l/30)*30 (L1666-1670). The bare
   DiffusionGemmaDecoderModel returns only last_hidden_state (no logits, no softcap), so tracing the
   submodule would drop the tail. This is exactly what generation does (decoder_forward = self.forward).
4. The SC soft-embedding matmul dtype is bf16 (tied embed/lm_head weight dtype), not fp16. Reference:
   softmax(fp32) -> .to(bf16) -> matmul(probs_bf16, weight_bf16) * embed_scale. The next-step SC input is
   processed_logits.to(bf16) (generation L1070-1071), so a bf16 SC input is CORRECT parity, not a break.

### Eight design decisions (authoritative; resolve 3 blockers + 7 majors)
- D1 (blocker: IR-D sliding window + blocker: IR-E cannot represent it + major: None-mask may be causal):
  IR-D takes an EXPLICIT {full_attention, sliding_attention} dict of 4D additive masks, NOT None. The
  decoder consumes a dict decoder_attention_mask directly (modeling L1301 walrus check) and indexes it per
  layer_type (L1319); the mask is added to scores (L269). full_mask = all-zeros (bidirectional over
  prefix+canvas). sliding_mask = zeros except -inf on prefix columns j < (cache_len - 1024); canvas-to-canvas
  fully visible. VERIFIED that the reference achieves windowing by physically capping the sliding KV cache to
  sliding_window (generation returns {full: None, sliding: None} for a DynamicCache with no padding, L1380-1385,
  so windowing lives in the cache, not a mask). Masking old keys is numerically identical to evicting them,
  so this keeps a UNIFORM full-length cache and IR-E's per-layer slice stays valid end to end. A post-build
  assertion confirms full = all-attend, sliding = windowed, and rows are identical (no causal triangle); a
  functional smoke test proves the mask input controls attention (bidirectional vs canvas-causal logits differ).
- D2 (major: fp16 tail is the largest parity deviation): IR-D is exported fp32 (compress_to_fp16=False) so the
  lm_head, the final softcap, and the 262144-wide SC softmax execute in fp32. int4 shrinks the bulk weights;
  the tie-sensitive tail stays fp32 via ignored_scope + no fp16 compression. A measurement cell reports
  max|logit delta| and argmax-flip rate of a naive fp16 tail vs the fp32 reference so parity is quantified.
- D3 (blocker: MoE unit test crashes): the equivalence test runs in ONE dtype, fp32 copies of the expert
  weights (gate_up_proj.float(), down_proj.float(), x fp32), because torch matmul/einsum do not promote
  fp32<->bf16. The ~1e-3 assertion stays meaningful and the notebook no longer halts in cell 2.
- D4 (major: experts class hardcoded to decoder layer 0): discover experts by iterating named_modules for
  every module exposing gate_up_proj + down_proj + act_fn (3D gate_up_proj), assert at least one under the
  encoder and one under the decoder, and patch every distinct class. Verified: every encoder AND decoder layer
  has .experts (modular L471-472, L546-547), same class both stacks.
- D5 (major: mask trick may bake the zeros branch): verified self_conditioning_mask enters as a
  value-independent broadcast MULTIPLY (modeling L1282-1283), gated only by `is not None` presence checks, no
  .item()/bool/value branch. Passing a real (B,) mask bakes the multiply; a post-trace smoke test confirms
  mask=0 vs mask=1 logits differ. No monkeypatch needed.
- D6 (numerics + SC dtype): token entropy, the multinomial draw, and SC feedback all operate on the SAME
  processed = (raw softcapped IR-D logits) / temperature; divide by temperature FIRST. Only argmax is
  temperature-invariant. Keep the SC-input bf16 cast (matches generation L1070-1071); note the OV fp32 caveat.
- D7 (manifest completeness): the manifest carries token_ids {eos, pad, bos}, per-layer-type KV shapes
  (sliding 8x256, full 2x512), layer_types, canvas_length 256, block_length 256, all sampler defaults
  (max_denoising_steps 48, entropy_bound 0.1, t_min 0.4, t_max 0.8, stability_threshold 1,
  confidence_threshold 0.005), the constants (final_logit_softcapping 30.0, embed_scale sqrt(hidden),
  rms_norm_eps), and the fp32-tail note.
- D8 (blocker: IR-E per-layer slice): the new-token KV slice is computed PER LAYER from that layer's own past
  length (past_kv[2*i].shape[-2]), never a single global layer-0 length. The IR-E trace example uses DIFFERENT
  prefix lengths for sliding vs full layers so the dynamic axis is captured per layer, and a two-block cache
  round-trip smoke test confirms the slice returns exactly the new tokens.
