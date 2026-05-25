#!/bin/bash
# v14: continuation of v13. v13 ended at 2000 iters with VL still bouncing
# 500–870 and rolling avg flat at ~4.8 — value head wasn't converged.
# Warm-start from v13 best, same env mix and hparams, run 4000 more iters
# to let VL settle and rolling avg push higher. Refinement only — no
# hparam changes.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v14_longer \
    --iters 4000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.015 \
    --warm checkpoints/phase7_p7_v13_railfix_best.pt \
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
    > phase7_p7_v14_longer.log 2>&1
