#!/bin/bash
# Phase 7 fresh-init v6: clean-isolation play.
#
# Changes vs v5:
#   - movement_penalty_weight = 1.5 (NEW: OB-scatter penalty restored)
#   - eor_bonus_max = 2.0 (was 4.0; reduce "scatter is good" learned bias)
#   - entropy_coef = 0.02 (between v4 and v5)
#
# Goal: agent learns to take clean isolation shots by default, only breaking
# clusters when worthwhile, planning shape for the next shot.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v6_isolation \
    --iters 5000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.02 \
    --shape_bonus_max 0 \
    --movement_penalty_weight 1.5 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 2.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    > phase7_p7_v6_isolation.log 2>&1
