#!/usr/bin/env python3
"""Build openvino_audio_embeddings_model.xml for Gemma-4 12B unified.

The unified model has NO audio tower: audio is raw 16 kHz waveform chunked into
640-sample frames (40 ms per soft token), RMSNorm-ed (no learned scale, so the
waveform's absolute level cancels out), and lifted into LM space by one
Linear(640 -> 3840, no bias). That single weight ships as audio_projection.npy
in the 12B model repo; this script wraps it into an OpenVINO IR that the
genai audio patch (gemma4-unified-audio.patch) compiles next to the vision model.

Usage: python make_audio_ir.py <model_dir_with_audio_projection.npy>
"""
import sys

import numpy as np
import openvino as ov

model_dir = sys.argv[1]
ops = ov.opset16

W = np.load(model_dir + "/audio_projection.npy")  # [3840, 640] f32 (from BF16)
inp = ops.parameter([-1, -1, 640], np.float32, name="input_features")
mean_sq = ops.add(ops.reduce_mean(ops.multiply(inp, inp), np.array([-1]), keep_dims=True),
                  ops.constant(np.float32(1e-6)))
normed = ops.multiply(inp, ops.power(mean_sq, ops.constant(np.float32(-0.5))))
out = ops.matmul(normed, ops.constant(W.T.astype(np.float32).copy()), False, False)
out.set_friendly_name("audio_embeds")
model = ov.Model([out], [inp], "gemma4_unified_audio_embedder")
ov.save_model(model, model_dir + "/openvino_audio_embeddings_model.xml", compress_to_fp16=False)
print("wrote", model_dir + "/openvino_audio_embeddings_model.xml")
