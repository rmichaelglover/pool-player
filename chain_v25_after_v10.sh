#!/usr/bin/env bash
# Auto-chain: wait for the v10 8-ball depth-2 run (wrapper PID below) to finish,
# then launch the 14.1 v25 geometry retrain. Chained because both are heavy
# multi-env CPU training jobs — running them at once would thrash the CPU and
# slow both. v25 runs regardless of v10's outcome (independent experiments).
set -u
V10_PID=1897301
ROOT=/home/r-m-glover/claude_projects/pool_player

echo "chain: waiting for v10 (PID $V10_PID) to finish... $(date)"
while kill -0 "$V10_PID" 2>/dev/null; do sleep 120; done
echo "chain: v10 wrapper exited at $(date); launching v25 in 30s"
sleep 30
cd "$ROOT/14.1"
bash launch_v25_geomfix.sh
echo "chain: v25 launcher returned at $(date)"
