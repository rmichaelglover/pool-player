#!/bin/bash
# Fine-tune from phase7_p7_eor_x2_best (567K small net, rolling 5.50)
# with the new cue-ball path-length penalty.
#
#   cue_movement_penalty_weight = 0.3
#   normalization 100" cap 1.5  (so 100" cue path ≈ -0.3 shape units,
#                                  300"+ saturates at -0.45)
#
# Other settings match the eor_x2 run: shape_bonus 2.0, mov_pen 1.0,
# eor 4.0, mild noise, lr 5e-5.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --warm checkpoints/phase7_p7_eor_x2_best.pt \
    --tag p7_cuepen01 \
    --iters 2000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 5e-5 \
    --entropy_coef 0.01 \
    --shape_bonus_max 2.0 \
    --movement_penalty_weight 1.0 \
    --cue_movement_penalty_weight 0.1 \
    --eor_bonus_max 4.0 \
    --aim_noise_deg 0.5 \
    --force_noise_pct 0.05 \
    --spin_noise 0.05 \
    > phase7_p7_cuepen01.log 2>&1
