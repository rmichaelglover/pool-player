#!/bin/bash
# v25: re-distill v24's recipe on the CORRECTED side-pocket geometry.
#
# v24 (phase7_p7_v24_valuefocus) was trained on the OLD enumerator, which
# wrongly rejected makeable side-pocket cuts/banks (rear-facing over-restriction,
# fixed in shared/table_geometry.py, merged in PR #1). Phase7Env already
# enumerates with include_banks=True, so it now sees the corrected action space
# automatically — v25 just *trains* on it instead of only generalizing.
#
# This mirrors v24's recipe EXACTLY (value_weight 1.5, search_k 6 / search_m 2,
# temp 0.7, the reward-shaping stack, noise, 25% mixed drills) so the corrected
# geometry is the only variable vs v24. Warm-starts from v24_best (the champion)
# and fine-tunes for 1000 iters (~10h at ~35s/iter) to optimize over the
# now-correct side shots. NOTE: cd's to this script's dir (14.1) -- launch_v24.sh
# had a stale `cd .../rl`.
set -u
cd "$(dirname "$0")"

BASE=checkpoints/phase7_p7_v24_valuefocus_best.pt
TAG=p7_v25_geomfix
STAMP=$(date +%Y%m%dT%H%M%S)
SUMMARY=eval_v25_${STAMP}_summary.log

echo "=== v25 geometry-fix re-distill started $(date) ===" | tee "$SUMMARY"
echo "warm=$BASE  v24 recipe + corrected side-pocket enumerator  iters=1000" | tee -a "$SUMMARY"

python3 -u train_phase7.py \
    --distill \
    --tag "$TAG" \
    --iters 1000 \
    --envs 16 \
    --steps_per_update 32 \
    --lr 1e-4 \
    --warm "$BASE" \
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
    > phase7_${TAG}.log 2>&1
echo "  train $TAG exit=$? ($(date))" | tee -a "$SUMMARY"

echo "=== EVAL v25 vs v24 (paired seeds; eval labels: v20=v24 baseline, v23=v25) $(date) ===" | tee -a "$SUMMARY"
NEW=checkpoints/phase7_${TAG}_best.pt
if [ -f "$NEW" ]; then
  python3 -u eval_v20_vs_v23.py --n 100 --v20 "$BASE" --v23 "$NEW" 2>&1 \
    | grep -E "mean=|^  v20:|^  v23:|Balls pocketed|Reracks" | tee -a "$SUMMARY"
else
  echo "  $NEW missing, eval skipped" | tee -a "$SUMMARY"
fi

echo "=== v25 complete $(date) ===" | tee -a "$SUMMARY"
echo "Read the eval as: v23 line = v25, v20 line = v24 baseline. If v25 mean balls > v24, promote and point the demo at v25." | tee -a "$SUMMARY"
