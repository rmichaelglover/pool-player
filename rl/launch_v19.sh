#!/bin/bash
# v19: two simultaneous improvements over v18 (the run that did the
# 56-ball multi-rack sequence with consistent EOR).
#
# 1. Multi-criterion break-ball selection in _eor_bonus (n=2 and n=3).
#    Previous logic: nearest-to-apex = the ball to preserve. New logic:
#    score each ball by (a) sweet-spot distance to apex (peak d=9″), and
#    (b) clear line of sight from ball to apex. Pick the higher-quality
#    candidate. Targets the 5-vs-13 break-ball selection error.
#
# 2. --shape_bonus_max 1.0 (never enabled before). Penalizes the current
#    shot's difficulty (cut_norm + dist_norm). Targets the "system
#    assigns non-trivial prob to 78° cut shots" pattern observed in the
#    v18 demo even with --p7_search_prob_threshold filtering.
#
# Distillation continuation from v18 best, 2500 iters, self-search,
# overnight-ish ~9h.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v19_easyshape \
    --iters 2500 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm checkpoints/phase7_p7_v18_nextshape_best.pt \
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
    --shape_bonus_max 1.0 \
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    --mixed \
    --mix_ratio 0.4 \
    --rail_drill_share 0.40 \
    --threeball_drill_share 0.25 \
    > phase7_p7_v19_easyshape.log 2>&1
