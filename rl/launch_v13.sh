#!/bin/bash
# v13: value-head refresh after the 3-body rail-frozen physics fix in
# pool_sim.c (2026-05-11). v12 learned that rail-frozen shots miss often
# — that prior is stale now. Warm-start from v12 best and run rollouts
# under the corrected physics so the value head rebalances. Bump
# rail_drill_share well above v12 so rail-frozen examples dominate the
# refresh signal. Lower entropy_coef (0.03 → 0.015) so the policy
# refines rather than re-explores from scratch.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v13_railfix \
    --iters 2000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.015 \
    --warm checkpoints/phase7_p7_v12_eor3ball_best.pt \
    --movement_penalty_weight 1.5 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 5.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    --mixed \
    --mix_ratio 0.4 \
    --rail_drill_share 0.55 \
    --threeball_drill_share 0.25 \
    > phase7_p7_v13_railfix.log 2>&1
