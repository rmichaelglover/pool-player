"""
Heuristic shot picker — zero learning. Uses legal-shot enumeration + simple
rules for ball choice and force. Serves as:
  (a) A non-learned baseline to verify the architecture works end-to-end,
  (b) A starting policy for later AlphaZero-style learning.

Policy:
  - Enumerate all legal shots.
  - If any exist, pick the easiest (lowest difficulty score).
  - Aim at the ghost position. Force scales with total travel distance. Spin 0.
  - If no legal shot: fall back to a safety (just aim softly at any ball — TODO).

For break shots (first of a rack), the env bypasses call-shot; we pick a
high-force central break shot.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from shot_enumerator import (generate_legal_shots, easiest_shot,
                              POCKETS, LegalShot, R, TABLE_LENGTH, TABLE_WIDTH)


@dataclass
class HeuristicAction:
    aim_angle: float
    force: float
    spin: float
    # Debug info — lets the demo show what was chosen
    chosen_shot: LegalShot | None
    reason: str


def break_shot_action(cue_pos) -> HeuristicAction:
    """Opening break or post-rerack break: aim at the rack apex with heavy force."""
    # Rack apex is at RACK_APEX from train_phase6.py — duplicating constant here
    # to keep this module import-light.
    RACK_APEX = (75.0, 25.0)
    dx = RACK_APEX[0] - cue_pos[0]
    dy = RACK_APEX[1] - cue_pos[1]
    aim = math.atan2(dy, dx)
    return HeuristicAction(aim_angle=aim, force=240.0, spin=0.0,
                           chosen_shot=None, reason='break shot — hard at rack apex')


def regular_shot_action(cue_pos, balls) -> HeuristicAction:
    """Post-break shot — find legal shots, pick easiest."""
    shots = generate_legal_shots(cue_pos, balls, max_cut_deg=70.0)
    if not shots:
        # No direct shot available → fall back to a soft shot aimed at the
        # nearest ball. Primitive safety; just avoids scratching-by-flailing.
        if not balls:
            return HeuristicAction(
                aim_angle=0.0, force=60.0, spin=0.0,
                chosen_shot=None, reason='no balls — stalling')
        nearest_id = min(balls, key=lambda b: math.hypot(balls[b][0] - cue_pos[0],
                                                         balls[b][1] - cue_pos[1]))
        nb = balls[nearest_id]
        return HeuristicAction(
            aim_angle=math.atan2(nb[1] - cue_pos[1], nb[0] - cue_pos[0]),
            force=60.0, spin=0.0,
            chosen_shot=None, reason='no legal shot — soft safety')

    shot = easiest_shot(shots)
    # Force: scales with ball-to-pocket distance primarily. Enough to reach the
    # pocket comfortably, not so much that the cue rockets after contact.
    force = max(55.0, min(110.0, 35.0 + 1.1 * shot.ball_to_pocket_dist))
    # Spin: slight draw on non-straight shots so cue doesn't follow the object
    # ball into the pocket on near-straight hits. The -0.5 spin_factor is
    # physics-bounded — fades with distance, so long shots get less.
    spin = -0.5 if shot.cut_angle_deg < 15.0 else 0.0
    return HeuristicAction(
        aim_angle=shot.aim_angle, force=force, spin=spin,
        chosen_shot=shot,
        reason=f'ball {shot.ball_id} → pocket {shot.pocket_idx} '
               f'(cut={shot.cut_angle_deg:.1f}°, diff={shot.difficulty:.2f})',
    )


def heuristic_action(env) -> HeuristicAction:
    """Pick a shot for the env's current state. Handles break vs regular."""
    is_break = getattr(env, 'is_break_shot', False)
    if is_break:
        return break_shot_action(env.cue)
    return regular_shot_action(env.cue, env.balls)


if __name__ == '__main__':
    # Quick demo: run the heuristic on a random Phase 6b state.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_phase6b import Phase6bEnv
    from train_phase4 import decode_action
    env = Phase6bEnv(max_shots=50)
    total = 0
    step = 0
    while not env.done and step < 30:
        a = heuristic_action(env)
        _, r, _, info = env.step(a.aim_angle, a.force, a.spin)
        total += int(r)
        step += 1
        flag = 'BRK' if info.get('is_break_shot') else ('OK' if info.get('called_shot_valid') else 'MISS')
        chosen = a.chosen_shot
        desc = (f"ball {chosen.ball_id}→{chosen.pocket_idx} (cut {chosen.cut_angle_deg:.0f}°)"
                if chosen else a.reason)
        print(f'shot {step}: {flag:4s} force={a.force:5.1f} {desc}  '
              f'pocketed_ids={info.get("pocketed_ids")}  total={total//10}')
    print(f'\nFinal: pocketed {env.total_pocketed} balls in {step} shots, '
          f'{env.rerack_count} reracks')
