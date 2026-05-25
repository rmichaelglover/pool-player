#!/bin/bash
# v20: stronger shape penalty to drive the policy away from extreme cut
# shots (78°+) that real players hesitate on but the deterministic sim
# rewards perfectly. v19 enabled --shape_bonus_max 1.0 with a hard
# difficulty clamp at 1.0 — so a 50° and an 80° cut both saturated to
# the same penalty, leaving no gradient against extreme cuts.
#
# Changes from v19:
# 1. _shape_bonus clamp raised: 1.0 → 3.0. Now cut_norm scales past
#    1.0 (78° = 1.73, 90° = 2.0), giving real gradient against extreme
#    cuts. Worst-case penalty bounded at 3*shape_bonus_max.
# 2. --shape_bonus_max 1.0 → 2.0. Combined with the new clamp, a 78°
#    shot gets ~-3.66 penalty (vs -1.0 in v19), 35% of pocket_reward.
#
# Distillation continuation from v19 best, 2500 iters, self-search.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v20_hardcut \
    --iters 2500 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm checkpoints/phase7_p7_v19_easyshape_best.pt \
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
    --next_shape_bonus_max 1.5 \
    --shape_bonus_max 2.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    --mixed \
    --mix_ratio 0.4 \
    --rail_drill_share 0.40 \
    --threeball_drill_share 0.25 \
    > phase7_p7_v20_hardcut.log 2>&1
