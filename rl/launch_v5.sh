#!/bin/bash
# Phase 7 fresh-init v5: same recipe as v4 but with entropy_coef=0.05 (5x higher)
# to encourage policy diversity and prevent the "1.00 / 0.0000" overconfidence
# we saw in v4 demo runs.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v5_entropy \
    --iters 5000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.05 \
    --shape_bonus_max 0 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 4.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    > phase7_p7_v5_entropy.log 2>&1
