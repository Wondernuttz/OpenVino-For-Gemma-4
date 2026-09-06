# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gpg python3 python3-venv libtbb12 libnuma1 libgomp1 \
    && curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
       | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client' \
       > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update && apt-get install -y --no-install-recommends intel-opencl-icd libze-intel-gpu1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# 12B text path: upstream runtime, NOT the custom 26B kernel build.
FROM base AS upstream
ARG OV_VERSION=2026.4.0.dev20260723
ARG GENAI_VERSION=2026.4.0.0.dev20260723
RUN pip install --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly \
    "openvino==${OV_VERSION}" "openvino-genai==${GENAI_VERSION}" "openvino-tokenizers==${GENAI_VERSION}"
ENV OV_RUNTIME_FLAVOR=upstream

FROM upstream AS server
COPY serving/ovserver_moe.py serving/launch.py /app/
ENV OV_MODEL=/model OV_HOST=0.0.0.0 OV_PORT=8000 OV_PROFILE=gemma12-text
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
ENTRYPOINT ["python", "/app/launch.py"]

# Keep this stage after server for compatibility with Docker's legacy builder.
# Supply Linux x86_64 / Python 3.12 wheels built together from the custom fork
# and matching GenAI/tokenizers. Fail rather than silently using stock wheels.
FROM base AS custom
COPY serving/wheels/ /wheels/
RUN python -c "from pathlib import Path; p=list(Path('/wheels').glob('*.whl')); assert all(any(x.name.startswith(n+'-') for x in p) for n in ['openvino','openvino_genai','openvino_tokenizers']), 'Supply all three matching custom runtime wheels; see serving/README.md'" \
    && pip install /wheels/*.whl && pip check \
    && python -c "import openvino,pathlib; p=pathlib.Path(openvino.__file__).parent/'libs/libopenvino_intel_gpu_plugin.so'; assert b'MOE_GROUPED_BINARY_LOOKUP' in p.read_bytes(), 'Not the patched GPU plugin'"
ENV OV_RUNTIME_FLAVOR=wondernuttz-custom

FROM custom AS custom-server
COPY serving/ovserver_moe.py serving/launch.py /app/
ENV OV_MODEL=/model OV_HOST=0.0.0.0 OV_PORT=8000 OV_PROFILE=gemma12-text
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
ENTRYPOINT ["python", "/app/launch.py"]
