# Quickstart: choose the right model and runtime

## B50 or another 16 GB Arc: start with 12B text

Start with [the Docker/server guide](serving/README.md#b50--12b-start-here).
It includes pinned runtime dependencies, GPU inventory, model mounting, and a
chat test. The default profile uses **DQ128, a plain VLMPipeline, prefix caching
off, and 4K total context**. This is a starting configuration requiring validation
on your hardware, not a guarantee of B50 speed or freedom from OOM.

Download the full [Gemma 4 12B Heretic OpenVINO export](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov).
The model's approximately 7.5 GB of weights is not its complete runtime memory use.

**This replaces the older quickstart's blanket 16 GB minimum for 26B and its
OpenVINO 2026.2 installation recipe.** Do not use that old recipe for the new
runtime gains or the 12B compatibility path.

## B70 / 32 GB: optimized 26B

Use [Wondernuttz's custom OpenVINO fork](https://github.com/Wondernuttz/openvino/tree/arc-xe2-gemma4-pa-2026.4)
with matching GenAI/tokenizers. An ordinary pip install does not contain the
custom grouped-MoE lookup patch.

- [Measured configuration, source revisions, and build notes](https://github.com/Wondernuttz/openvino/blob/7b27ac8eb88faec682d4c96dfba749b93df32285/WONDERNUTTZ_GEMMA4_PREFILL_20260906.md).
- [Docker packaging of a matching custom runtime](serving/README.md#upstream-image-versus-the-custom-fork).
- [Per-tune validation and StyleTune hold](https://github.com/Wondernuttz/openvino/blob/7b27ac8eb88faec682d4c96dfba749b93df32285/WONDERNUTTZ_GEMMA4_RELEASE_GATE_20260906.md).

Our 24K reference test sampled about **26.5 GiB** card memory on one **B70 under
Linux**. This is not a B50/16 GB profile or a validated offload setup. Dense 31B
and 12B do not use the 26B MoE optimization.

## Already have a validated native runtime?

No Docker is required. From the repository root:

```bash
export OV_MODEL=/absolute/path/to/model
export OV_DEVICE=GPU.0  # Example only: verify device identity first.
export OV_PROFILE=gemma12-text
/path/to/validated-runtime/bin/python serving/launch.py
```

The native endpoint binds to localhost:8000 by default. See the
[server guide](serving/README.md) for API requests, bearer-key support, explicit
B70 settings, and the optional legacy-named foreground launchers.

## Important distinctions

- Text-only 12B: **DQ128** in the documented plain VLMPipeline.
- 12B vision/audio: separate **DQ0** recipes on the model card; the portable
  server rejects multimedia instead of silently discarding it.
- Exact GPU indexes vary. Container drivers do not replace the host kernel.
- Total context includes output. Prefix-cache allocation is memory, not a token count.
- The portable server is not OpenArc and does not modify an existing OpenArc container.
- SSE responses are buffered, not token-by-token. Tool calls and forced JSON
  output are unsupported by this small server.

See [validation status](serving/VALIDATION.md) before treating a build check as a
hardware/coherence benchmark.
