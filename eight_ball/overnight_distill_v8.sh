#!/usr/bin/env bash
# v8 search-distillation WITH value-based ball-in-hand placement search
# (2026-05-29). Warm-starts from v6_best (current champion; v7 round-2 was flat).
# New vs v6/v7: placement is no longer advanced by the net's raw mean with no
# target — placement_search_distill samples candidate cue positions, scores each
# by the net's value of the best follow-up shot (depth-1 search), and distills
# the head toward the best. Heuristic-free: the network learns placement from
# its own value, so novel placements can emerge. _placement_reward was also
# trimmed to a thin rule-floor (penalize only no-legal-shot), removing the
# hand-coded "easiest cut / shape" strategy prior.
#
# After training, evaluates the distilled RAW policy (no search, deterministic)
# vs v6_best. Uses sweep_*_summary.log naming so the watch loop keeps working.
set -u
cd "$(dirname "$0")"

BASE=checkpoints/eight_ball_8ball_v6_best.pt
TAG=8ball_v8_distill
STAMP=$(date +%Y%m%dT%H%M%S)
SUMMARY=sweep_${STAMP}_distill_summary.log

echo "=== v8 placement-search distillation started $(date) ===" | tee "$SUMMARY"
echo "warm_start=$BASE  search-distill K=6 M=1 gamma=0.99  place_n=8  iters=500" | tee -a "$SUMMARY"

echo "--- TRAIN $TAG (search-distill + value-based placement) $(date) ---" | tee -a "$SUMMARY"
python3 -u train_eight_ball_distill.py \
  --tag "$TAG" --iters 500 --envs 8 --steps 24 \
  --search_k 6 --search_m 1 --search_mc 1 --gamma 0.99 \
  --place_n 8 --place_weight 1.0 \
  --log_std_min -3.0 --warm_start "$BASE" \
  > "train_${TAG}.log" 2>&1
echo "  train $TAG exit=$? ($(date))" | tee -a "$SUMMARY"

echo "=== EVAL phase $(date) ===" | tee -a "$SUMMARY"
for ck in best final; do
  CKPT=checkpoints/eight_ball_${TAG}_${ck}.pt
  if [ -f "$CKPT" ]; then
    echo "--- EVAL ${TAG}_${ck} vs v6_best (raw deterministic, no search) ---" | tee -a "$SUMMARY"
    python3 -u eval_eight_ball.py "$CKPT" \
      --model_b "$BASE" --games 100 2>&1 \
      | grep -E "Win rate|Elo|Avg game|Avg fouls" | tee -a "$SUMMARY"
  else
    echo "--- EVAL ${TAG}_${ck}: checkpoint missing, skipped ---" | tee -a "$SUMMARY"
  fi
done

echo "=== v8 complete $(date) ===" | tee -a "$SUMMARY"
echo "Watch placement quality in the demo (search OFF) after promoting if v8 >= v6." | tee -a "$SUMMARY"
