#!/bin/bash
# v17: continuation of v16 distillation with corrected shot_enumerator.
# Fix applied 2026-05-19: shot_enumerator.py:114 now skips obstacles whose
# forward-component along the cue motion is <= 0.5*BALL_R. This unblocks
# the frozen-ball-perpendicular shot case (another ball touching the cue
# at right angles to the shot line was incorrectly flagged as a blocker).
#
# Goal: train v17 against a slightly larger legal-shot space so the policy
# learns to weight previously-invisible shots like "12 → TR when 7 is
# touching the cue from above."
#
# Warm-start: v16 best. Self-search throughout (no frozen teacher — v16-best
# is already a reasonable teacher and we want the student to incorporate
# the new shots into its own preferences).

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v17_enumfix \
    --iters 1200 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm checkpoints/phase7_p7_v16_distill_best.pt \
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
    > phase7_p7_v17_enumfix.log 2>&1
