#!/bin/bash
# stop both servers for max free RAM
pkill -f "ovserver_moe\.py" 2>/dev/null
pkill -TERM -f "llama-server.*--port 8002" 2>/dev/null
sleep 4
free -g | head -2
# babysitter: kill the test if MemAvailable < 2GB
( while true; do
    avail=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)
    if [ "$avail" -lt 2000000 ]; then echo "[guard] LOW MEM ${avail}kB -- killing test"; pkill -f "ov_31b_test\.py"; break; fi
    sleep 1
  done ) &
GUARD=$!
/home/wondernutts/ov-genai/bin/python /tmp/ov_31b_test.py 2>&1 | grep -vE "^\[|Warning|warn"
kill $GUARD 2>/dev/null
echo "--- restoring servers ---"
setsid /tmp/restore8002.sh
setsid /tmp/start8092.sh
