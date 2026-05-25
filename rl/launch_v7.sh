#!/bin/bash
# Phase 7 fresh-init v7: curriculum + higher EOR + higher entropy.
#
# Changes vs v6:
#   --mixed (Phase9MixedEnv: 30% curriculum episodes with key/break ball drill)
#   --mix_ratio 0.3
#   eor_bonus_max 2.0 → 5.0 (higher break-ball preservation reward)
#   entropy_coef 0.02 → 0.04 (more exploration of side pockets, rail shots)
#
# Goal: emergence of keyball/breakball strategy through repeated end-of-rack
# practice, plus broader shot-type exploration to fix v6's side-pocket aversion.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v7_curriculum \
    --iters 5000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.04 \
    --shape_bonus_max 0 \
    --movement_penalty_weight 1.5 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 5.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    --mixed \
    --mix_ratio 0.3 \
    > phase7_p7_v7_curriculum.log 2>&1
