#!/usr/bin/env bash
# Overnight 8-ball sharpening sweep (2026-05-28).
# Baseline established: v5_best beats v4_best 55-45 (+35 Elo), det. game len 18 shots.
# Hypothesis: the policy mean is good but too noisy when sampled (Ent~2.2,
# log_std floor -2.5). Anneal entropy down + lower log_std_min to sharpen aim.
# All runs warm-start from v5_best. Sequential (CPU-bound; parallel would thrash).
# After training, each *_best is eval'd head-to-head vs v5_best (100 games).
set -u
cd "$(dirname "$0")"

BASE=checkpoints/eight_ball_8ball_v5_best.pt
ITERS=2000
LR=8e-5
STAMP=$(date +%Y%m%dT%H%M%S)
SUMMARY=sweep_${STAMP}_summary.log

echo "=== Overnight sweep started $(date) ===" | tee "$SUMMARY"
echo "warm_start=$BASE iters=$ITERS lr=$LR" | tee -a "$SUMMARY"

# config: tag  entropy_start  entropy_final  log_std_min  shape_weight
run_cfg () {
  local tag=$1 e0=$2 e1=$3 lsm=$4 shp=$5
  echo "--- TRAIN $tag (ent $e0->$e1, log_std_min $lsm, shape $shp) $(date) ---" | tee -a "$SUMMARY"
  python3 -u train_eight_ball.py \
    --tag "$tag" --iters "$ITERS" --lr "$LR" \
    --warm_start "$BASE" \
    --entropy_coef "$e0" --entropy_coef_final "$e1" \
    --log_std_min "$lsm" --shape_reward_weight "$shp" \
    > "train_${tag}.log" 2>&1
  echo "  train $tag exit=$? ($(date))" | tee -a "$SUMMARY"
}

# 1) sharpen: moderate anneal, tighter aim floor
run_cfg 8ball_v6_sharpen   0.01  0.003  -3.5  0.05
# 2) sharpen_hard: aggressive anneal, tightest aim floor
run_cfg 8ball_v6_sharphard 0.008 0.0015 -4.0  0.05
# 3) shape: moderate sharpen + double the position-play shaping
run_cfg 8ball_v6_shape     0.01  0.004  -3.5  0.10

echo "=== EVAL phase $(date) ===" | tee -a "$SUMMARY"
for tag in 8ball_v6_sharpen 8ball_v6_sharphard 8ball_v6_shape; do
  ckpt=checkpoints/eight_ball_${tag}_best.pt
  if [ -f "$ckpt" ]; then
    echo "--- EVAL $tag vs v5_best ---" | tee -a "$SUMMARY"
    python3 -u eval_eight_ball.py "$ckpt" \
      --model_b "$BASE" --games 100 2>&1 \
      | grep -E "Win rate|Elo|Avg game|Avg fouls" | tee -a "$SUMMARY"
  else
    echo "--- EVAL $tag: checkpoint missing, skipped ---" | tee -a "$SUMMARY"
  fi
done

echo "=== Sweep complete $(date) ===" | tee -a "$SUMMARY"
echo "Promote the highest +Elo run to v6 (copy its _best to eight_ball_8ball_v6_best.pt)." | tee -a "$SUMMARY"
