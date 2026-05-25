#!/bin/bash
# Phase 7 fresh-init v4: realistic skilled-player noise (0.03°/0.5%/0.01)
# + new cushion physics (CUSH_R=0.70, no spin-reset hack)
# + shape formula DISABLED (let pocket reward + sim noise carry shot quality)
# + cue-control penalties active (path 0.1, ricochet 0.5)
# + EOR bonus 4.0 for break-ball preserve / rerack-scatter
#
# Iters: 5000 (~10 hr CPU)

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v4_realistic \
    --iters 5000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.01 \
    --shape_bonus_max 0 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 4.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    > phase7_p7_v4_realistic.log 2>&1
