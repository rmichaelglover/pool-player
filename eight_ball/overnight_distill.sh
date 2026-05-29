#!/usr/bin/env bash
# Overnight 8-ball SEARCH-DISTILLATION (2026-05-28).
# Replaces the sharpening sweep. Bakes depth-1 search (which beat raw 16/16)
# into the policy weights via supervised distillation (train_eight_ball_distill).
# Warm-starts from v5_best. After training, evaluates the distilled RAW policy
# (no search, deterministic) vs v5_best — the real test of whether search was
# successfully baked into the net.
#
# Uses the sweep_*_summary.log naming so the existing watch loop keeps working.
set -u
cd "$(dirname "$0")"

BASE=checkpoints/eight_ball_8ball_v5_best.pt
TAG=8ball_v6_distill
STAMP=$(date +%Y%m%dT%H%M%S)
SUMMARY=sweep_${STAMP}_distill_summary.log

echo "=== Distillation session started $(date) ===" | tee "$SUMMARY"
echo "warm_start=$BASE  search-distill K=6 M=1 gamma=0.99  iters=500" | tee -a "$SUMMARY"

echo "--- TRAIN $TAG (search-distillation from v5_best) $(date) ---" | tee -a "$SUMMARY"
python3 -u train_eight_ball_distill.py \
  --tag "$TAG" --iters 500 --envs 8 --steps 24 \
  --search_k 6 --search_m 1 --search_mc 1 --gamma 0.99 \
  --log_std_min -3.0 --warm_start "$BASE" \
  > "train_${TAG}.log" 2>&1
echo "  train $TAG exit=$? ($(date))" | tee -a "$SUMMARY"

echo "=== EVAL phase $(date) ===" | tee -a "$SUMMARY"
# Distilled RAW policy vs v5_best raw policy (eval harness uses deterministic
# net play for both — no search at eval time). If the distilled net beats
# v5_best here, search strength is now in the weights.
for ck in best final; do
  CKPT=checkpoints/eight_ball_${TAG}_${ck}.pt
  if [ -f "$CKPT" ]; then
    echo "--- EVAL ${TAG}_${ck} vs v5_best (raw deterministic, no search) ---" | tee -a "$SUMMARY"
    python3 -u eval_eight_ball.py "$CKPT" \
      --model_b "$BASE" --games 100 2>&1 \
      | grep -E "Win rate|Elo|Avg game|Avg fouls" | tee -a "$SUMMARY"
  else
    echo "--- EVAL ${TAG}_${ck}: checkpoint missing, skipped ---" | tee -a "$SUMMARY"
  fi
done

echo "=== Sweep complete $(date) ===" | tee -a "$SUMMARY"
echo "If distilled raw beats v5_best, search is baked in — promote to v6_best." | tee -a "$SUMMARY"
