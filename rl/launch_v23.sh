#!/bin/bash
# v23: address the rail-ball break-shot weak-force pattern. Diagnosis:
# when the break ball is on a rail, the network's learned rail-shot
# prior (low force = reliable pocket) overrides the post-rerack scatter
# bonus, producing forces ~60 instead of the ~240 needed to scatter the
# rack. The fix is targeted exposure: a new "railbreak" curriculum drill
# that sets up 14 reracked balls + 1 break ball near a rail + cue
# positioned for the break, with _post_rerack_break_pending=True at
# episode start so the existing scatter bonus fires on the first shot.
#
# Drill mix (within curriculum):
#   rail_drill:      35%   (lighter than v22's 70%)
#   threeball_drill: 25%
#   railbreak_drill: 25%   (NEW)
#   key+break:       15%
# Total curriculum: 40% (same as v20). Net railbreak prevalence: 10%.
#
# Warm-start from v20-best so all EOR/strategy is preserved. No noise
# added.

cd /home/r-m-glover/claude_projects/pool_player/rl

python3 -u train_phase7.py \
    --distill \
    --tag p7_v23_railbreak \
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
    --mix_ratio 0.40 \
    --rail_drill_share 0.35 \
    --threeball_drill_share 0.25 \
    --railbreak_drill_share 0.25 \
    > phase7_p7_v23_railbreak.log 2>&1
