#!/bin/bash
# Phase 7 fresh-init v2: new empirical shape formula + skilled-player noise.
#
# Shape: cut_func(θ) × dist_scale(x, y), [0, 1] → [-1, 1] × shape_bonus_max
#   cut_func: peaks at 0°, smooth descent, asymptotic to 0 at 90°
#   dist_scale: 1.0 at zero dist → 0.75 at table diagonal (gentle floor)
# Penalties: cue path length (0.1) + cue ricochets (0.5)
# EOR bonus: 4.0
# Noise: 0.2°/2%/0.02 (skilled player)
# Iters: 5000 (~10 hr CPU)

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v2_skilled \
    --iters 5000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.01 \
    --shape_bonus_max 2.0 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 4.0 \
    --aim_noise_deg 0.2 \
    --force_noise_pct 0.02 \
    --spin_noise 0.02 \
    > phase7_p7_v2_skilled.log 2>&1
