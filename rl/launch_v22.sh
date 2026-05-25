#!/bin/bash
# v22: Option E — heavily emphasize rail-shot drills in the env mix to
# fix the easy-rail-miss pattern, WITHOUT adding global execution noise.
# v21's noise approach improved rail calibration but degraded EOR depth.
# v22 isolates the rail-shot training problem from the noise-vs-strategy
# trade-off: more rail-drill exposure, deterministic execution preserved.
#
# Drill mix change: regular Phase7 episodes 40% (was 60%), curriculum 60%
# (was 40%). Within curriculum, rail drills are 70% (was 40%). Net:
# 42% of episodes are rail drills (was 16% in v18-v20). The rail drill
# places a single OB within 1.5″ of a rail with cue mid-table, exactly
# the pattern the model has been miscalibrating.
#
# Everything else identical to v20: warm-start from v20-best preserves
# EOR/break sequencing, all rewards unchanged.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v22_railheavy \
    --iters 2500 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm checkpoints/phase7_p7_v20_hardcut_best.pt \
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
    --mix_ratio 0.60 \
    --rail_drill_share 0.70 \
    --threeball_drill_share 0.15 \
    > phase7_p7_v22_railheavy.log 2>&1
