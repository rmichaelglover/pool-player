#!/bin/bash
# Phase 7 fresh-init v3: NO shape formula. Pocket reward + simulator noise
# carry the shot-quality signal. Cue-control penalties keep clean play.
#
# shape_bonus_max=0 disables the formula entirely.
# Penalties (independent of shape bonus): cue path 0.1, ricochet 0.5
# EOR bonus: 4.0
# Noise: 0.2°/2%/0.02 (skilled player)
# Iters: 5000 (~10 hr CPU)

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v3_noformula \
    --iters 5000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.01 \
    --shape_bonus_max 0 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 4.0 \
    --aim_noise_deg 0.2 \
    --force_noise_pct 0.02 \
    --spin_noise 0.02 \
    > phase7_p7_v3_noformula.log 2>&1
