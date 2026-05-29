#!/usr/bin/env bash
# Round-2 8-ball SEARCH-DISTILLATION (2026-05-29).
# Round 1 (v6_distill) baked depth-1 search into the policy and beat v5_best
# 77% / +210 Elo raw. This round repeats the distillation, warm-starting from
# the NEW champion (v6_best == v6_distill_final) to compound the gain.
# After training, evaluates the distilled RAW policy (no search, deterministic)
# vs v6_best — the test of whether a second round adds anything.
#
# Uses the sweep_*_summary.log naming so the existing watch loop keeps working.
set -u
cd "$(dirname "$0")"

BASE=checkpoints/eight_ball_8ball_v6_best.pt
TAG=8ball_v7_distill
STAMP=$(date +%Y%m%dT%H%M%S)
SUMMARY=sweep_${STAMP}_distill_summary.log

echo "=== Distillation round-2 started $(date) ===" | tee "$SUMMARY"
echo "warm_start=$BASE  search-distill K=6 M=1 gamma=0.99  iters=500" | tee -a "$SUMMARY"

echo "--- TRAIN $TAG (search-distillation from v6_best) $(date) ---" | tee -a "$SUMMARY"
python3 -u train_eight_ball_distill.py \
  --tag "$TAG" --iters 500 --envs 8 --steps 24 \
  --search_k 6 --search_m 1 --search_mc 1 --gamma 0.99 \
  --log_std_min -3.0 --warm_start "$BASE" \
  > "train_${TAG}.log" 2>&1
echo "  train $TAG exit=$? ($(date))" | tee -a "$SUMMARY"

echo "=== EVAL phase $(date) ===" | tee -a "$SUMMARY"
# Distilled RAW policy vs v6_best raw policy (deterministic, no search).
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

echo "=== Round-2 complete $(date) ===" | tee -a "$SUMMARY"
echo "If v7_distill raw beats v6_best, a second distill round compounds — promote to v7_best." | tee -a "$SUMMARY"
