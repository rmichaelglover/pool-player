# TODO

## In flight
- **8-ball v1 training** running overnight (PID 1319292, started 2026-05-25).
  - 1500 iters, 16 envs, 32 steps, CPU. EightBallNet ~569K params.
  - Full 8-ball self-play (auto-break, group assignment, fouls, safety).
  - Log: `/tmp/eight_ball_v1_train.log`
  - Checkpoints: `rl/checkpoints/eight_ball_8ball_v1_{best,latest,final}.pt`
  - When done: run `eval_eight_ball.py` vs random to measure improvement.

## Current state — 14.1 (2026-05-25)
- **Best model: v20_hardcut_best + search (MC=8, prob_threshold=0.075).**
  - Demo server running this config on port 8001.
  - 297-ball run (21 reracks) achieved today — strongest run to date.
  - v20 beat v23 in paired eval (mean 49.5 vs 45.7, 16-12-2 wins).
- v24_valuefocus trained (1500 iters, warm-start from v20) but didn't
  improve over v20 — rolling avg settled ~8.0 vs v20's baseline.
  Log: `rl/phase7_p7_v24_valuefocus.log`.

## Iteration history (v19–v24)
| Tag | Idea | Outcome |
|-----|------|---------|
| v19_easyshape | Multi-criterion break-ball scoring | 56-ball run with search |
| v20_hardcut | Hard EOR cutoff shaping | Current best; 297 balls with search |
| v21_noise | Noise-augmented training | No lift over v20 |
| v22_railheavy | Heavy rail-shot weighting | Trained but not best |
| v23_railbreak | Rail-break drill curriculum | Narrowly lost to v20 in paired eval |
| v24_valuefocus | 1.5x value loss weight | No improvement; rolling avg ~8.0 |

## Next ideas
- **Policy is the bottleneck, not value head** (confirmed by v24 result
  and earlier observation that search unlocks strategies the policy
  can't find alone). Next experiments should focus on policy quality:
  - Search-augmented training with higher MC budget during data generation
  - Curriculum on specific failure modes (post-break cue position)
  - Larger search budget at inference (MC=16 or MC=32)
- **Noise-augmented fine-tune** of v20 with conservative noise
  (`--aim_noise_deg 0.10 --force_noise_pct 0.03 --spin_noise 0.10`)
  to improve robustness without regressing policy quality.

## Demo / visualization
- **Show the chosen aim point** in the demo (small marker on the
  cushion-back chord), not just the pocket center. The shot enumerator
  sets per-shot `aim_point` — needs to flow through `LegalShot` → server
  response → `highlightCall()` in `demo_phase6b.html`.
- Optional: dim or hide the cushion-line "pocket center" marker since
  it's no longer where the AI actually aims.

## Geometry follow-ups
- **Decide if the strict cushion-corner clearance is too aggressive.**
  `optimal_pocket_aim` requires ≥ BALL_R perpendicular distance from
  all 4 corridor corners. If too many real-pool-feasible shots are
  filtered, add a small fudge factor (e.g., `>= BALL_R * 0.9`).
  Symptom: heuristic policy runs short despite reasonable layouts.
- **Action-space expansion (Option 2).** If the network should
  *strategically* pick edge vs center aim for cue-position reasons,
  add K aim variants per pocket as separate shot tokens.

## Open questions
- Bigger nets (gpu1M) underfit at 1500 iters in Phase 4 sweep and
  didn't beat the 438K CPU baseline. Phase 7 is ~567K — fine on CPU.
  If GPU scaling is revisited, needs lr/batch tuning, not just more params.
- Post-break cue position is the main failure mode in long runs — worth
  a targeted curriculum?

## Won't-do (decided)
- ~~Hand-coded geometry filter for "this side shot is steep" — solved by
  per-shot `optimal_pocket_aim`.~~
- ~~Network learning the new pocket geometry without enumerator changes —
  rejected as sample-inefficient; geometric filter is free at inference.~~
