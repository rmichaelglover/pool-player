#!/bin/bash
# Phase 7 fresh-init training with all "clean play" penalties active from iter 0.
# Goal: a policy that pockets well AND plays cleanly (controlled cue, single
# contacts, minimal scatter). 567K-param small net, ~10 hr CPU.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_fresh_clean \
    --iters 5000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.01 \
    --shape_bonus_max 2.0 \
    --movement_penalty_weight 1.5 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 4.0 \
    > phase7_p7_fresh_clean.log 2>&1
