#!/bin/bash
pkill -f "ovserver_moe\.py" 2>/dev/null
sleep 1
nohup env OV_MODEL=/home/wondernutts/models/heretics/gemma-4-26B-A4B-heretic-int4-ov-lutfix \
  OV_DEVICE=GPU.2 OV_PORT=8092 OV_EXPECT_BUS="bus: 8" OV_MAX_CTX_TOKENS=30000 \
  /home/wondernutts/ov-genai/bin/python /home/wondernutts/ovserver_moe.py > /tmp/ovserver_moe.log 2>&1 &
echo "launched pid $!"
