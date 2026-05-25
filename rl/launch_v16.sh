#!/bin/bash
# v16: search-distillation. Address the policy/value gap surfaced by the
# v15+search demo on 2026-05-19 — value head has learned EOR sequencing
# and blocker removal, but the policy's argmax hides it. Pure
# distillation (no PPO ratio) trains the policy to imitate search-chosen
# actions.
#
# Teacher schedule: first 500 iters use frozen v15 best as the search
# teacher (so the value head used to compute Q targets is stable while
# the student catches up). After iter 500, switch to self-search.
#
# Search uses K=4 candidates (cheaper than inference K=8 but enough for
# training signal). Reward shaping unchanged from v15.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v16_distill \
    --iters 2500 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm checkpoints/phase7_p7_v15_railshape_best.pt \
    --teacher_ckpt checkpoints/phase7_p7_v15_railshape_best.pt \
    --frozen_teacher_iters 500 \
    --search_k 4 \
    --softmax_temp 1.0 \
    --ce_weight 1.0 \
    --mse_force_weight 0.1 \
    --mse_spin_weight 0.1 \
    --value_weight 0.5 \
    --distill_entropy 0.005 \
    --movement_penalty_weight 1.5 \
    --cue_movement_penalty_weight 0.1 \
    --cue_ricochet_penalty_weight 0.5 \
    --eor_bonus_max 5.0 \
    --rail_shot_bonus_weight 1.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    --mixed \
    --mix_ratio 0.4 \
    --rail_drill_share 0.40 \
    --threeball_drill_share 0.25 \
    > phase7_p7_v16_distill.log 2>&1
