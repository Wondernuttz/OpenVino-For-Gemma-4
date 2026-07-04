#!/bin/bash
# Buddy/CHIM endpoint: LUT-fixed 26B MoE heretic on the PROD card (GPU.1 = bus 4), no-think,
# same recipe as the Discord bots. Rollback to SYCL GGUF: /tmp/restore8002.sh
pkill -TERM -f "llama-server.*--port 8002" 2>/dev/null
pkill -f "OV_PORT=8002" 2>/dev/null
sleep 3
nohup env OV_MODEL=/home/wondernutts/models/heretics/gemma-4-26B-A4B-heretic-int4-ov-lutfix \
  OV_DEVICE=GPU.1 OV_PORT=8002 OV_EXPECT_BUS="bus: 4" OV_MAX_CTX_TOKENS=30000 \
  /home/wondernutts/ov-genai/bin/python /home/wondernutts/ovserver_moe.py > /tmp/ovserver_buddy.log 2>&1 &
echo "launched pid $!"
