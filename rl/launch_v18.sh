#!/bin/bash
# v18: address the post-break cue-position failure mode from v17's
# 58-ball run. After pocketing the break ball, the cue ended up out of
# position and the new rack stalled. Turn on --next_shape_bonus_max so
# every shot's reward includes a "leave easy next shot" component, which
# directly teaches the agent to land the cue somewhere with a clean
# follow-up — most relevant on the break shot when the rack scatters.
#
# next_shape_bonus_max = 1.5 (between rail_shot_bonus 1.0 and
# eor_bonus_max 5.0). Big enough to compete with pocket reward (10),
# small enough not to override EOR shaping.
#
# Distillation continuation from v17 best with corrected enumerator,
# 3500 iters (~10h on CPU) for overnight.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v18_nextshape \
    --iters 3500 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm checkpoints/phase7_p7_v17_enumfix_best.pt \
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
    --aim_noise_deg 0.03 \
    --force_noise_pct 0.005 \
    --spin_noise 0.01 \
    --mixed \
    --mix_ratio 0.4 \
    --rail_drill_share 0.40 \
    --threeball_drill_share 0.25 \
    > phase7_p7_v18_nextshape.log 2>&1
