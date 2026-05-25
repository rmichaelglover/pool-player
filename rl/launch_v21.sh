#!/bin/bash
# v21: address the "easy rail miss" pattern observed in v20 demo.
# Diagnosis: in deterministic sim, easy shots pocket for a wide band of
# (force, spin) choices, so the agent learned a vague default that
# happens to fail on certain rail geometries. The fix is execution
# noise — even easy shots must demand robust calibration, not just
# any-pocket calibration.
#
# Noise bumped 3-5× from v20's tiny defaults:
#   aim_noise_deg:   0.03 → 0.10   (~real pro execution)
#   force_noise_pct: 0.005 → 0.01
#   spin_noise:      0.01 → 0.02
#
# Smoke test confirmed the model is initially MUCH more brittle than
# expected: AvgRun drops 8 → ~2 at iter 10 when noise turns on. Banking
# on distillation closing the gap over 2500 iters as the policy learns
# robust force/spin choices and the value head learns expected-return
# Q under noise.
#
# All other rewards and hyperparameters identical to v20. Warm-start
# from v20-best to preserve the EOR/break/shape learning.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v21_noise \
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
    --aim_noise_deg 0.10 \
    --force_noise_pct 0.01 \
    --spin_noise 0.02 \
    --mixed \
    --mix_ratio 0.4 \
    --rail_drill_share 0.40 \
    --threeball_drill_share 0.25 \
    > phase7_p7_v21_noise.log 2>&1
