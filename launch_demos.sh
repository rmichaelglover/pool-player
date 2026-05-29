#!/usr/bin/env bash
# Launch the pool demo servers with the CORRECT policy/checkpoint args.
#
# Gotcha this script exists to prevent: demo_phase6b_server.py's --policy
# defaults to 'heuristic', which ignores the network checkpoint entirely and
# plays a hand-coded policy. The 14.1 demo MUST be launched with
# `--policy phase7 --p7_ckpt <ckpt>` to actually use the trained transformer
# (the bare `--ckpt` arg is only read in 'network' mode). v24's architecture
# matches the demo's phase-7 defaults (embed=128, heads=8, layers=4, ff=256),
# so no extra arch flags are needed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

# 14.1 straight-pool demo — phase-7 transformer (v24), banks on — :8001
( cd "$HERE/14.1" && nohup python3 -u demo_phase6b_server.py --policy phase7 \
    --p7_ckpt checkpoints/phase7_p7_v24_valuefocus_best.pt --port 8001 \
    > demo_server.log 2>&1 & echo "14.1 demo (phase7 v24) -> http://localhost:8001  pid $!" )

# 8-ball demo — v6 search-distilled — :8002
( cd "$HERE/eight_ball" && nohup python3 -u demo_eight_ball_server.py \
    --ckpt checkpoints/eight_ball_8ball_v6_distill_final.pt --port 8002 \
    > demo_server.log 2>&1 & echo "8-ball demo (v6) -> http://localhost:8002  pid $!" )

echo "Tailscale Funnel (public URL) proxies one local port; see: tailscale funnel status"
