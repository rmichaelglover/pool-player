#!/bin/bash
# v24: value-head focus + wider training search
#
# Diagnosis: v20's rolling avg plateaued at ~7 from iter 10 through 2500.
# The policy was already good at warm-start — distillation couldn't improve
# it because the value head is the bottleneck. VL bounced 70–270 all run,
# making the K=4/M=1/MC=1 (4-sim) search targets noisy. Meanwhile v21–v23
# each tried curriculum/noise tweaks and all degraded overall play.
#
# Changes from v20:
# 1. value_weight 0.5 → 1.5: triple emphasis on value head training.
# 2. search_k 4 → 6, search_m 1 → 2: 12 sims vs 4 — wider shot coverage
#    and force exploration give cleaner distillation targets.
# 3. softmax_temp 1.0 → 0.7: sharper targets given higher-quality search.
# 4. mix_ratio 0.4 → 0.25: less curriculum, more real-game signal.
# 5. iters 2500 → 1500: sized for overnight at ~35s/iter (~14.5h).
#
# Distillation continuation from v20 best.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v24_valuefocus \
    --iters 1500 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm checkpoints/phase7_p7_v20_hardcut_best.pt \
    --search_k 6 \
    --search_m 2 \
    --softmax_temp 0.7 \
    --ce_weight 1.0 \
    --mse_force_weight 0.1 \
    --mse_spin_weight 0.1 \
    --value_weight 1.5 \
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
    --mix_ratio 0.25 \
    --rail_drill_share 0.25 \
    --threeball_drill_share 0.25 \
    > phase7_p7_v24_valuefocus.log 2>&1
