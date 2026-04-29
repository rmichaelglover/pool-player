# TODO

## In flight
- **Phase 7 deterministic retrain on new geometry** running in background.
  - Tag `p7_newgeom`, 500 iters, 16 envs, CPU. PoolGameNet ~567K params.
  - Output: `rl/checkpoints/phase7_p7_newgeom_{best,latest}.pt`.
  - Log: `/tmp/phase7_newgeom_train.log`.
  - When it finishes: compare rolling avg run length vs the prior
    `phase7_p7_first_best.pt` (which was trained on old geometry).
  - Demo with: `python3 demo_phase6b_server.py --policy phase7
    --p7_ckpt checkpoints/phase7_p7_newgeom_best.pt --port 8001`

## Next training step (after deterministic baseline lands)
- Warm-start a **noise-augmented Phase 7 run** from the deterministic best.
  - Suggested: `--aim_noise_deg 0.10 --force_noise_pct 0.03 --spin_noise 0.10`
    (matches good-amateur execution, same as demo defaults).
  - Tag `p7_newgeom_noisy`. Should learn robust shot selection without
    refighting the geometry-feasibility problem.

## Demo / visualization
- **Show the chosen aim point** in the demo (small marker on the
  cushion-back chord), not just the pocket center. The shot enumerator now
  sets per-shot `aim_point` — needs to flow through `LegalShot` → server
  response → `highlightCall()` in `demo_phase6b.html`.
- Optional: dim or hide the cushion-line "pocket center" marker since it's
  no longer where the AI actually aims.

## Geometry follow-ups
- **Decide if the strict cushion-corner clearance is too aggressive.** Right
  now `optimal_pocket_aim` requires ≥ BALL_R perpendicular distance from
  all 4 corridor corners. If too many real-pool-feasible shots are filtered,
  add a small fudge factor (e.g., `>= BALL_R * 0.9`). Symptom: heuristic
  policy runs short despite reasonable layouts.
- **Action-space expansion (Option 2).** If you ever want the network to
  *strategically* pick edge vs center aim for cue-position reasons, add K
  aim variants per pocket as separate shot tokens.

## Open questions
- Memory note `project_phase4_scaling_sweep` says bigger nets (gpu1M)
  underfit at 1500 iters and didn't beat the 438K CPU baseline. Phase 7 is
  ~567K — probably fine on CPU. If we ever try GPU scaling here, learn
  from that sweep: lr/batch tuning, not just more params.

## Won't-do (decided)
- ~~Hand-coded geometry filter for "this side shot is steep" — solved by
  per-shot `optimal_pocket_aim`.~~
- ~~Network learning the new pocket geometry without enumerator changes —
  rejected as sample-inefficient; geometric filter is free at inference.~~
