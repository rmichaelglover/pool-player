#!/usr/bin/env bash
# v10 = v6's recipe + DEPTH-2 opponent-reply search (Variant A), on the
# corrected enumerator (side-pocket front-jaw gate + kick foul-avoidance tier,
# merged to master in PR #1). 2026-05-29.
#
# Rationale: depth-1 search-distillation plateaued — v7/v8/v9 were all flat vs
# v6. Depth-2 is the actual step-change lever: on turn-switch states, instead
# of trusting the value head's 1 - V(opponent), search the opponent's best
# reply (opp_K=4, opp_M=1) and use that. This correctly devalues shots/safeties
# that leave the opponent a good position (validated: post-break safety Q drops
# ~0.15 -> ~0.02). ~2.4x the depth-1 per-decision cost.
#
# Top-level branching is held at v6's K=6 M=1 so this isolates the effect of
# depth alone vs v6_best. Warm-starts from v6_best (current champion).
set -u
cd "$(dirname "$0")"

BASE=checkpoints/eight_ball_8ball_v6_best.pt
TAG=8ball_v10_depth2
STAMP=$(date +%Y%m%dT%H%M%S)
SUMMARY=sweep_${STAMP}_distill_summary.log

echo "=== v10 depth-2 opponent-reply distillation started $(date) ===" | tee "$SUMMARY"
echo "warm_start=$BASE  K=6 M=1 MC=1 gamma=0.99  search_depth=2 K2=4 M2=1  place_n=8" | tee -a "$SUMMARY"

echo "--- TRAIN $TAG (depth-2 search-distill) $(date) ---" | tee -a "$SUMMARY"
python3 -u train_eight_ball_distill.py \
  --tag "$TAG" --iters 500 --envs 8 --steps 24 \
  --search_k 6 --search_m 1 --search_mc 1 --gamma 0.99 \
  --search_depth 2 --search_k2 4 --search_m2 1 \
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

echo "=== v10 complete $(date) ===" | tee -a "$SUMMARY"
echo "If v10 > v6 (beyond ~5% noise): depth-2 broke the plateau -> promote, then" | tee -a "$SUMMARY"
echo "consider depth-2 + full negamax (Variant B) and/or sparse-reward self-play." | tee -a "$SUMMARY"
