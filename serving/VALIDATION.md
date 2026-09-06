# Portable server validation — September 6, 2026

## Completed

- 19 CPU-only unit/HTTP contract tests passed on Windows Python 3.14 and inside
  the custom Ubuntu/Python 3.12 Docker image. These cover profiles, context-budget
  trimming, reasoning boundaries, authentication, chat, model mismatch, unsupported
  multimodal/tools requests, and buffered SSE. Generation is mocked in these tests.
- Python syntax compilation passed.
- `server` Docker target built using the pinned upstream wheel set.
- `custom-server` Docker target built using wheels exported from the accepted
  live runtime. `pip check`, runtime imports, and the custom lookup-marker check passed.
- No GPU device was passed to the image build/import/contract tests. No model was
  loaded and no live service was stopped or restarted.
- The active bots and buddy worker PIDs remained present through validation.

## Verified runtime identities inside containers

Upstream OpenVINO: `2026.4.0-22527-2c3fe6fa4f1`.

Custom OpenVINO: `2026.4.0-105-31c98eec8ab-prefill-upstream-integration`.
This version string reflects the build base; the plugin hash identifies the
accepted binary with the published lookup patch.

Both GenAI: `2026.4.0.0-3308-79bc2469701`.

| Image | GPU plugin SHA256 | Custom lookup marker |
|---|---|---|
| Upstream | `f137499ece481013c7cfb439849cafed5fdee250bc159650e6d7458aabf8cd20` | Absent |
| Custom | `2818d86aa36e34cf7ce3bd5dd2ab27d3660c21baef39ebeb7a360a9c5dea9388` | Present |

The custom plugin hash matches the accepted native serving runtime. This proves
packaging identity, **not equivalent Docker inference speed or correctness**.

## Not yet established

- B50 model loading, GPU kernel compilation, OOM behavior, coherence, or speed.
- An end-to-end 12B or 26B inference run using this portable Docker package.
- OpenArc integration: this is a separate server, not an OpenArc patch.
- Compose execution: the test host has Docker Engine 29.1.3 but no Compose plugin.
  The documented plain `docker build` path was used for image validation.
- A downloadable custom image or binary release. Runtime wheels used for testing
  remain local; users need a matching custom runtime build for `custom-server`.

Do not report the 19 mocked tests as a model intelligence evaluation. The native
B70 measurements and per-checkpoint limitations remain in the linked benchmark
reports. The B50/12B profile is an initial test configuration, not certification.
