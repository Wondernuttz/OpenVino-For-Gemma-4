#!/bin/sh
set -eu
: "${OV_MODEL:?Set OV_MODEL to a complete local model export}"
: "${OV_PROFILE:?Choose gemma12-text or gemma26-b70}"
: "${OV_DEVICE:?Set OV_DEVICE after checking the GPU inventory}"
export OV_PORT="${OV_PORT:-8002}"
exec "${OV_PYTHON:-python3}" "$(dirname "$0")/launch.py"
