#!/bin/bash
# v15: address two v14 demo issues.
# 1. Rail-shot aversion: turn on rail_shot_bonus_weight (1.0 = 10% of
#    pocket_reward). The flag has existed since v8 but was never enabled.
# 2. End-of-rack sequence: _eor_bonus now also grades n=3 at half strength,
#    rewarding pocketing the ball furthest from rack apex (preserves
#    key-ball-1 and break-ball candidate).
# Lower rail_drill_share 0.55 → 0.40 since rail shots are now rewarded in
# regular episodes too.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --tag p7_v15_railshape \
    --iters 3500 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --entropy_coef 0.015 \
    --warm checkpoints/phase7_p7_v14_longer_best.pt \
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
    > phase7_p7_v15_railshape.log 2>&1
