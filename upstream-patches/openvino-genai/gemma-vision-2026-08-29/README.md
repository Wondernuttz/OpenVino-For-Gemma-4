# Gemma Vision local edits — 2026-08-29

This folder preserves a focused three-file Gemma vision change set developed against the public [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) source tree.

## Provenance

- Upstream repository: `https://github.com/openvinotoolkit/openvino.genai.git`
- Upstream branch used locally: `master`
- Local base commit: `8981d6f848f17985979be0a9224251d181f68c56`
- Edited paths:
  - `src/cpp/src/visual_language/gemma4/classes.cpp`
  - `src/cpp/src/visual_language/gemma4/classes.hpp`
  - `src/cpp/src/visual_language/vision_encoder.hpp`

`gemma-vision-uncommitted.patch` is a binary-safe unified patch containing only those changes. Apply it to a compatible OpenVINO GenAI checkout with `git apply gemma-vision-uncommitted.patch`, then review and adapt it as upstream evolves.

No model weights, generated binaries, credentials, private runtime configuration, or user data are included.
