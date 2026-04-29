"""
Phase 8: Break-ball drill.

Env starts in the post-rerack configuration:
  - 14 object balls racked (apex empty, per 14.1 rules)
  - 1 "break ball" placed 4-10" from the rack, at varied positions
  - Cue ball in the kitchen (head quarter)

Episode flow (user's design):
  - Agent's first shot IS the break shot — no auto-execution.
  - If the break ball is pocketed AND the rack scatters such that there are
    legal shots → the episode continues as a normal run-out.
  - If the break fails (break ball not pocketed, or no legal shots remaining)
    → episode ends.

This produces a self-balancing training distribution:
  - Bad breaks → short episodes → negative gradient signal
  - Good breaks → long cascading run-outs → positive signal over many shots
  - No explicit mixing ratio needed; the env's outcome structure does it.

Action space, network, and training loop are identical to Phase 7 — only
the initial state sampler changes, and we reuse the Phase 7 training code.
"""
from __future__ import annotations

import math
import random
import sys, os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot_enumerator import R
from train_phase6 import RACK_POSITIONS, RACK_APEX, TABLE_LENGTH, TABLE_WIDTH
from train_phase7 import Phase7Env, train_phase7


class Phase8Env(Phase7Env):
    """Break-ball drill env. Inherits Phase 7 step/rerack logic; overrides only
    the initial state sampler so the agent chooses its own break shot."""

    def reset(self):
        self.cue, self.balls = self._sample_break_drill_setup()
        self.shot_idx = 0
        self.done = False
        self.rerack_count = 0
        self.total_pocketed = 0
        # No auto-break: agent picks its own break shot from the legal set.
        return self.get_obs()

    def _sample_break_drill_setup(self):
        """Place 14 racked balls (apex empty) + 1 break ball beside the rack,
        with cue paired to the break ball position for a natural break-shot
        geometry. Four scenario classes:

          - Break ball alongside rack (not near rail) on upper side → cue on
            same upper-kitchen side. Cue caroms off ball directly into rack.
          - Break ball alongside rack on lower side → cue on lower kitchen side.
          - Break ball near the upper rail (y < ~13) → cue on OPPOSITE
            (lower) kitchen side. Cue pockets ball, rebounds off rail back
            into rack.
          - Break ball near the lower rail → cue on upper kitchen side.
        """
        from shot_enumerator import generate_legal_shots
        balls = {}
        for i, pos in enumerate(RACK_POSITIONS[1:]):
            balls[i + 1] = list(pos)

        # Try scenarios until we get a valid setup. Six scenario classes for
        # variety (two distance variants × two sides × rail-close special):
        #   Upper alongside close   (by 11-16, x hugs rack):   cue upper kitchen
        #   Upper alongside far     (by 14-20, x 70-85):        cue upper kitchen
        #   Upper rail-close        (by 3-8):                   cue LOWER kitchen
        #   Lower alongside close   (by 34-39, x hugs rack):   cue lower kitchen
        #   Lower alongside far     (by 30-36, x 70-85):        cue lower kitchen
        #   Lower rail-close        (by 42-47):                 cue UPPER kitchen
        for _ in range(60):
            scenario = random.random()
            if scenario < 0.22:
                # Upper alongside, close to rack x-range
                bx = 75.0 + random.random() * 9.0       # 75-84 (tight to rack back)
                by = 11.0 + random.random() * 5.0       # 11-16
                cy_target = max(4.0, by - 2.0 - random.random() * 3.0)
            elif scenario < 0.40:
                # Upper alongside, farther from rack x-range
                bx = 70.0 + random.random() * 15.0      # 70-85 (wider x)
                by = 14.0 + random.random() * 6.0       # 14-20
                cy_target = max(4.0, by - 2.0 - random.random() * 4.0)
            elif scenario < 0.50:
                # Upper rail-close (classic around-the-corner)
                bx = 74.0 + random.random() * 11.0
                by = 3.0 + random.random() * 5.0        # 3-8
                cy_target = 30.0 + random.random() * 14.0
            elif scenario < 0.72:
                # Lower alongside, close to rack x-range
                bx = 75.0 + random.random() * 9.0
                by = 34.0 + random.random() * 5.0       # 34-39
                cy_target = min(46.0, by + 2.0 + random.random() * 3.0)
            elif scenario < 0.90:
                # Lower alongside, farther from rack x-range
                bx = 70.0 + random.random() * 15.0
                by = 30.0 + random.random() * 6.0       # 30-36
                cy_target = min(46.0, by + 2.0 + random.random() * 4.0)
            else:
                # Lower rail-close (classic around-the-corner)
                bx = 74.0 + random.random() * 11.0
                by = 42.0 + random.random() * 5.0
                cy_target = 6.0 + random.random() * 14.0

            # Bounds + overlap check on break ball
            if not (3 * R < bx < TABLE_LENGTH - 3 * R): continue
            if not (3 * R < by < TABLE_WIDTH - 3 * R): continue
            if any(math.hypot(bx - p[0], by - p[1]) < 3 * R
                   for p in RACK_POSITIONS[1:]):
                continue

            # Tentatively place break ball
            balls[15] = [bx, by]

            # Cue placement: kitchen x (8-28, widened for variety), target y
            # with jitter, clear of all balls, has legal shot.
            for _ in range(20):
                cx = 8.0 + random.random() * 20.0           # 8-28
                cy = cy_target + (random.random() - 0.5) * 7.0  # ±3.5 jitter
                cy = max(4.0, min(TABLE_WIDTH - 4.0, cy))
                if any(math.hypot(cx - p[0], cy - p[1]) < 4 * R
                       for p in balls.values()):
                    continue
                cue = [cx, cy]
                if generate_legal_shots(cue, balls, max_cut_deg=75.0):
                    return cue, balls
            # Cue placement failed; loop retries break-ball placement.
            del balls[15]

        # Fallback
        balls[15] = [80.0, 36.0]
        return [15.0, 12.0], balls


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--envs', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--tag', default='p8_baseline')
    p.add_argument('--lr', type=float, default=3e-5,  # lower than Phase 7 default
                   help='Default lowered vs Phase 7 (3e-5 vs 1e-4) to avoid '
                        'catastrophic forgetting of Phase 7 skills.')
    p.add_argument('--steps_per_update', type=int, default=32)
    p.add_argument('--entropy_coef', type=float, default=0.01)
    p.add_argument('--log_std_min', type=float, default=-2.5)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--warm', default=None,
                   help='Warm-start checkpoint (typically phase7 best)')
    args = p.parse_args()
    train_phase7(
        num_envs=args.envs, device_name=args.device, max_iters=args.iters,
        tag=args.tag, lr=args.lr, steps_per_update=args.steps_per_update,
        entropy_coef=args.entropy_coef, log_std_min=args.log_std_min,
        embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
        warm_start=args.warm,
        env_class=Phase8Env,
        label='Phase 8: break-ball drill', ckpt_prefix='phase8',
    )
