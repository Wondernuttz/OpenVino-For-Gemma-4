# Portable Gemma server and Docker

This is **Wondernuttz's standalone server**, not OpenArc. The server is derived from
the September 6 live implementation, with portable paths, explicit profiles,
text-only input, bounded requests, optional bearer authentication, and no
machine-specific service/process management. It does not install or replace OpenArc.

## Runtime updates and B50 / 12B setup

### September 6 wide-query update (26B / B70 only)

The corrected cached-prefill tile is now in the
[custom fork](https://github.com/Wondernuttz/openvino/commit/80c431dfd426bcff914886d6f9a9cef83b44b0c4).
The Heretic reference gained16.8% at24K uncached input; see
[benchmarks and validation caveats](../WIDEQ_24K_20260906.md).
The default12B/B50 image is unchanged.

After rebuilding **matching custom runtime wheels from that revision or later**
and following the custom image instructions below:

```bash
export OV_BUILD_TARGET=custom-server
export OV_PROFILE=gemma26-b70
export GEMMA_MIXED_512_TILE=wideq
docker compose -f serving/compose.yaml build
docker compose -f serving/compose.yaml up -d --force-recreate
```

Keep your existing `MODEL_PATH` and verified `OV_DEVICE`. These commands restart
your container. `git pull` alone does not upgrade its OpenVINO binary; the launcher
rejects this opt-in when the selector is missing. No new prebuilt image/wheel
download is published here. To use the old tile, unset `GEMMA_MIXED_512_TILE` and
recreate the container. Do not enable the rejected `64` or `compact` experiments.

### B50 / 12B setup

The Intel Arc Pro B50 has **16 GB VRAM and 224 GB/s memory bandwidth**
([Intel specifications](https://www.intel.com/content/www/us/en/products/sku/242615/intel-arc-pro-b50-graphics/specifications.html)).
Our published speed measurements are on a **B70**, not a B50. Same Xe2 family
does not establish equal throughput or identical driver behavior.

- Start with the **12B INT4 text model**, approximately 7.5 GB of weights.
- The default profile is **DQ128, plain VLMPipeline, no prefix-cache allocation,
  4,096 total tokens**, 512 minimum output reserve and 128 margin tokens.
- This is a conservative starting configuration, **not a B50 OOM/coherence certification**.
  KV state and execution workspace still consume memory even with prefix reuse off.
- **Do not use the 26B/B70 profile on the B50.** Its reference 24K test sampled
  about 26.5 GiB card usage. We have no validated 26B offload recipe for the B50.
- Debian is fine as a host conceptually: Docker shares its Linux kernel and supplies
  its own userspace driver/runtime. Kernel `6.18.12` alone does not certify the full stack.

Download the full
[12B export](https://huggingface.co/Wondernutts/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov)
to a host directory first (including tokenizer and embedding files).
Do not mount a partial download or a Transformers/GGUF model. Credentials remain
in your normal host download workflow; they are not baked into the image.

From the repository root, with Docker Engine and the Compose plugin installed:

```bash
export MODEL_PATH=/absolute/path/to/gemma4-12b-heretic-ov
docker compose -f serving/compose.yaml build

# Read-only GPU inventory; this does not load model weights.
docker compose -f serving/compose.yaml run --rm --no-deps --entrypoint python gemma \
  -c 'import openvino as ov; c=ov.Core(); print([(d,c.get_property(d,"FULL_DEVICE_NAME")) for d in c.available_devices])'

# Select the B50 from the inventory, not an assumed index. Example only:
export OV_DEVICE=GPU.0
docker compose -f serving/compose.yaml up
```

The API is published on **127.0.0.1:8000** by default. Do not change that to a
public bind without appropriate access controls. Set `OV_API_KEY` before `up`
to require `Authorization: Bearer <key>` on model/chat endpoints.
There is no automatic restart loop on failed model loading.

### Without the Compose plugin

Docker Engine alone is sufficient. From the repository root:

```bash
docker build --target server -t wondernuttz-gemma:local -f serving/Dockerfile .
docker run --rm --device /dev/dri --entrypoint python wondernuttz-gemma:local \
  -c 'import openvino as ov; c=ov.Core(); print([(d,c.get_property(d,"FULL_DEVICE_NAME")) for d in c.available_devices])'

# Replace the model path and GPU index with your actual values.
docker run --rm --init --name gemma12-text \
  --device /dev/dri -p 127.0.0.1:8000:8000 \
  --mount type=bind,src=/absolute/path/to/gemma4-12b-heretic-ov,dst=/model,readonly \
  -e OV_DEVICE=GPU.0 -e OV_PROFILE=gemma12-text \
  wondernuttz-gemma:local
```

This starts a separate container; it does not replace OpenArc. Choose a different
host port if OpenArc already occupies 8000 (for example `127.0.0.1:8001:8000`).

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b-heretic","messages":[{"role":"user","content":"You are a Whiterun innkeeper. Greet a traveler in two sentences."}],"max_tokens":128,"temperature":0.7}'
```

If an API key is configured, add the bearer header to the last two commands.
Allow initial compilation to finish. For failures, collect the error and runtime
versions before changing DQ or context. Do not assume a failed load means broken weights.

## Upstream image versus the custom fork

**`server` (default):** pinned upstream July 23 OpenVINO/GenAI 2026.4 wheels.
This makes a reproducible dependency starting point for 12B text, not the custom
26B speedup. This particular container still needs B50/model end-to-end validation.
Runtime versions are fixed at build time; restarting does not upgrade them.

**`custom-server`:** requires all three compatible **Linux x86_64 / Python 3.12**
runtime wheels in `serving/wheels/`: OpenVINO, GenAI, and tokenizers. It checks that
the GPU plugin contains `MOE_GROUPED_BINARY_LOOKUP` and refuses stock substitutes.
No prebuilt custom image or wheel download is published by this change.

The accepted source is [Wondernuttz's custom OpenVINO fork](https://github.com/Wondernuttz/openvino/tree/arc-xe2-gemma4-pa-2026.4),
revision `7b27ac8eb88faec682d4c96dfba749b93df32285`; matching tested GenAI is
`79bc246970146922a385b6c0342f185b45478f4b`.
Follow the [build/provenance notes](https://github.com/Wondernuttz/openvino/blob/7b27ac8eb88faec682d4c96dfba749b93df32285/WONDERNUTTZ_GEMMA4_PREFILL_20260906.md#provenance-build-and-credits).
Build matching components together; never drop a lone GPU plugin into an unrelated runtime.

If you already have a validated installed custom runtime, the included exporter
repackages its runtime distributions into wheels, with new RECORD checksums and
a SHA256 manifest. It copies no models, tokens, user configuration, or services:

```bash
python3.12 -m venv /tmp/ov-wheel-tools
/tmp/ov-wheel-tools/bin/pip install wheel
/path/to/validated-runtime/bin/python serving/export_runtime_wheels.py \
  --wheel-python /tmp/ov-wheel-tools/bin/python \
  --output serving/wheels

export OV_BUILD_TARGET=custom-server
docker compose -f serving/compose.yaml build
```

Repacked wheels retain distribution version labels; the **GPU plugin hash** records
the actual custom binary. Do not infer patch identity from `pip list` alone.
The exporter is packaging, not a source compiler or a substitute for validating a new build.

On a **B70**, after selecting that GPU and mounting a qualified 26B export:

```bash
export OV_BUILD_TARGET=custom-server
export OV_PROFILE=gemma26-b70
docker compose -f serving/compose.yaml up --build
```

This enables DQ128, U4 KV, 8 GiB prefix cache, batch16384, one sequence, and both
grouped-prefill/lookup switches. The total-token cap is 24,576, including output;
it is not identical to a benchmark with 24,576 input tokens plus output.
It rejects a non-B70 device and a missing custom lookup marker. **StyleTune's 24K
rollout remains held; this generic profile is not approval to use it.**

## Native launch / endpoint limitations

Use `OV_MODEL`, `OV_DEVICE`, and `OV_PROFILE` explicitly with `serving/launch.py`.
The two legacy-named shell launchers now run it in the foreground on ports 8002
or 8092. They no longer kill other processes, choose your GPU for you, or use
Wondernuttz's home-directory paths. `OV_PYTHON` can select your validated Python runtime.

- Gemma formatting, token-counted context trimming, and reasoning-boundary filtering
  are inherited from the live server. Thinking defaults OFF (`OV_THINK=1` to test ON).
- Generation is serialized. **SSE is buffered until the answer is complete**, not
  token-by-token streaming. An incomplete reasoning trace can cause one no-think retry.
- Text chat only. Images/audio, tool calls, and constrained JSON/grammar requests
  are rejected explicitly. The 12B vision/audio paths still require their separate DQ0 setup.
- Token usage is omitted rather than reported using the old character-count estimate.
  Native counts and timing are available in the custom `performance` field below.
  This is a small compatibility server, not the full OpenAI API.
- Token limits reduce oversized requests; they do not guarantee all allocations stay
  in VRAM or eliminate driver/model-switch faults. Do not disable caches on other services.

## Reading response speed

Each successful response includes a top-level `performance` object, also logged as
`[ov] performance {...}`. Buffered SSE includes it in the final JSON chunk before
`[DONE]`. No sampling, model precision, prompt, or kernel settings are changed by
measurement. JSON escapes such as `\u2014` are normal; a JSON parser displays `—`.

If `jq` is installed, show the answer and measurements with:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b-heretic","messages":[{"role":"user","content":"You are a Whiterun innkeeper. Greet a traveler in four sentences."}],"max_tokens":512,"temperature":0.7}' \
  | jq '{answer: .choices[0].message.content, performance}'
```

Without `jq`, pipe the curl response to `python3 -m json.tool --no-ensure-ascii`.
Retain your Authorization header if you configured an API key. For the named
container from the Docker example, recent measurements are also in:

```bash
docker logs --tail 100 gemma12-text 2>&1 | grep '\[ov\] performance'
```

Each entry in `performance.attempts` describes one native generation call:

| Field | Meaning |
|---|---|
| `input_tokens` | Native input count for the rendered prompt after context trimming; not a count of cache misses. |
| `generated_tokens` | Native generated count, including hidden reasoning/special tokens reported by GenAI; not just visible answer tokens. |
| `ttft_ms` | GenAI time to first generated token, which may be hidden reasoning; not first visible text over HTTP. |
| `pp_tokens_per_ttft_second` | Input tokens / TTFT in seconds, matching our benchmark convention. Includes first-token/startup overhead; not pure GPU prefill time. Suppressed when prefix caching is enabled, because cache hits inflate this ratio. |
| `decode_tokens_per_second` | GenAI throughput metric (tokens/s), including hidden generation. Unavailable for fewer than two generated tokens. |
| `pipeline_wall_seconds` | Wall time around this native generation call, excluding the server lock wait. |

`request_wall_seconds` includes prompt preparation, lock waiting, generation and
answer filtering, but excludes HTTP upload/serialization/transmission. A reasoning
boundary fallback produces two attempts and `retry_count: 1`; the discarded attempt
is not hidden or merged into a misleading answer-only rate. Missing/invalid native
metrics are `null`, never estimated from character counts. Standard OpenAI `usage`
is still omitted. Metrics log no prompts or response text.

Repeat a request to separate first-run compilation from warmed execution. A tiny
innkeeper prompt is useful for responsiveness, not a comparable sustained 2K/4K
prefill benchmark. SSE remains buffered: curl's `time_starttransfer` is not TTFT.

## Updating an existing install

Run these from your existing repository checkout to obtain the metrics update.
Do not discard local edits if `git pull` reports a conflict.

For Compose, retain the same `MODEL_PATH`, `OV_PROFILE`, `OV_THINK`, device, port,
API key and build-target settings (including any `.env` file) used originally:

```bash
git pull --ff-only
docker compose -f serving/compose.yaml build gemma
docker compose -f serving/compose.yaml up -d --no-deps gemma
```

This recreates the model container, briefly interrupting inference and reloading
the model. The read-only model directory is retained; no weights are downloaded.

For the plain `docker run --name gemma12-text` example:

```bash
git pull --ff-only
docker build -f serving/Dockerfile --target server -t wondernuttz-gemma:local .
```

Only after the build succeeds, stop the old container, remove that stopped container
(not its model directory), and rerun your original `docker run` command with the
same options. A simple `docker restart` keeps the old image and does not update code.
If using custom wheels, preserve `--target custom-server` and your existing image tag
instead of switching to the upstream `server` target. Without Docker, update the
checkout and restart only this server using the original environment and launcher.

## Checks

```bash
python serving/test_portable.py
docker compose -f serving/compose.yaml config
docker compose -f serving/compose.yaml run --rm --no-deps --entrypoint python gemma \
  -c 'import openvino as ov, openvino_genai as g; print(ov.__version__,g.__version__)'
```

Validation status is recorded in [VALIDATION.md](VALIDATION.md). No B50 benchmark
or Docker inference-quality claim should be inferred from an image-build/import test.
