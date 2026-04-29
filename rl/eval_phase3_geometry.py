"""
Geometry-based evaluation of a Phase 3 checkpoint. Two analyses on the same
set of shots:

  (3) Closest-approach-to-ghost histogram — for every shot, how far did the
      cue ball's straight-line trajectory pass from each pocket's ghost
      position?

  (4) Object-ball-trajectory analysis — for hit-but-not-pocketed shots,
      compute the object ball's exit direction analytically (along line of
      centers at contact) and how close its straight-line trajectory came
      to each pocket, pre-bounce.

Usage:
    python eval_phase3_geometry.py --ckpt checkpoints/phase3_v2_61pct.pt
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from train_phase3 import Phase3Env, POCKETS, ghost_ball, R


def line_point_distance(p0, v, q):
    """Closest distance from point q to the ray starting at p0 with unit dir v.
    (Infinite line; negative projections give the tangential distance too.)"""
    qx = q[0] - p0[0]
    qy = q[1] - p0[1]
    # Project q onto v (any scalar), subtract projection
    t = qx * v[0] + qy * v[1]
    perp_x = qx - t * v[0]
    perp_y = qy - t * v[1]
    return math.hypot(perp_x, perp_y), t  # distance + signed param along v


def simulate_ball_exit_direction(cue, ball, aim_angle):
    """
    Analytic object-ball exit direction after ideal ball-ball contact.

    The cue travels from `cue` in direction `aim_angle`. We find the first
    point on that line within 2R of the ball; at that point the cue's
    center-to-ball-center line defines the impulse direction. The object
    ball exits in that direction.

    Returns (exit_angle, d_closest) or (None, d_closest) if no contact.
    """
    v = (math.cos(aim_angle), math.sin(aim_angle))
    bx = ball[0] - cue[0]
    by = ball[1] - cue[1]
    # Parametrize cue position as cue + t*v. Distance^2 to ball center:
    #   (t*vx - bx)^2 + (t*vy - by)^2 = t^2 - 2t(bx*vx+by*vy) + bx^2+by^2
    proj = bx * v[0] + by * v[1]
    closest_sq = bx * bx + by * by - proj * proj
    closest = math.sqrt(max(0.0, closest_sq))
    if closest > 2 * R or proj <= 0:
        return None, closest
    # Contact happens at t such that |cue+t*v - ball|^2 = (2R)^2
    #   t = proj - sqrt((2R)^2 - closest^2)
    t_contact = proj - math.sqrt(max(0.0, (2 * R) ** 2 - closest_sq))
    cx = cue[0] + t_contact * v[0]
    cy = cue[1] + t_contact * v[1]
    # Exit direction = ball_center - cue_contact (unit vector)
    ex = ball[0] - cx
    ey = ball[1] - cy
    n = math.hypot(ex, ey)
    return math.atan2(ey / n, ex / n), closest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/phase3_v2_61pct.pt')
    parser.add_argument('--shots', type=int, default=1500)
    parser.add_argument('--cut', type=float, default=60.0)
    args = parser.parse_args()

    device = torch.device('cpu')
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2).to(device)
    net.log_std = torch.nn.Parameter(torch.full((2,), -0.5).to(device))
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    env = Phase3Env(max_cut_deg=args.cut)

    # Storage
    # (3) closest-approach to ghost
    d_ghost_target = []   # to ghost of env's target pocket
    d_ghost_best = []     # to ghost of whichever pocket minimizes distance
    # Shot classification
    n_hit = 0
    n_miss = 0
    n_pocket = 0
    # (4) For hit-not-pocket shots: object-ball trajectory closest approach
    obj_d_target = []     # obj trajectory closest to target pocket
    obj_d_best = []       # obj trajectory closest to any pocket
    obj_pocket_choice = [0] * 6  # which pocket the obj trajectory was closest to

    with torch.no_grad():
        for _ in range(args.shots):
            obs = env.reset()
            t = torch.FloatTensor(obs).unsqueeze(0)
            action, _, _ = net.get_action(t, deterministic=True)
            a = action[0].cpu().numpy()
            aim_angle = math.atan2(float(a[0]), float(a[1]))
            cue = tuple(env.cue)
            ball = tuple(env.ball_pos)
            target = env.target_pocket
            target_idx = POCKETS.index(target)

            # (3) closest-approach of cue-trajectory to each pocket's ghost
            v = (math.cos(aim_angle), math.sin(aim_angle))
            d_per_pocket = []
            for p in POCKETS:
                g = ghost_ball(ball, p)
                d, _ = line_point_distance(cue, v, g)
                d_per_pocket.append(d)
            d_ghost_target.append(d_per_pocket[target_idx])
            d_ghost_best.append(min(d_per_pocket))

            # Actually simulate to classify
            _, _, info = env.step(aim_angle)
            if info['pocketed']:
                n_pocket += 1
                continue
            if not info['hit']:
                n_miss += 1
                continue
            n_hit += 1

            # (4) analytic exit direction of object ball
            exit_angle, _ = simulate_ball_exit_direction(cue, ball, aim_angle)
            if exit_angle is None:
                continue
            ev = (math.cos(exit_angle), math.sin(exit_angle))
            d_per = []
            for p in POCKETS:
                d, t_proj = line_point_distance(ball, ev, p)
                # Only count pockets in the forward direction
                if t_proj > 0:
                    d_per.append(d)
                else:
                    d_per.append(float('inf'))
            obj_d_target.append(d_per[target_idx])
            obj_d_best.append(min(d_per))
            obj_pocket_choice[int(np.argmin(d_per))] += 1

    # ── Report ────────────────────────────────────────────────────────────
    print(f'Checkpoint: {args.ckpt}')
    print(f'Shots: {args.shots}   Env: Phase3(max_cut={args.cut})')
    total = args.shots
    print(f'\nOutcome breakdown:')
    print(f'  Pocketed:    {n_pocket:4d}  ({n_pocket/total:.1%})')
    print(f'  Hit no pkt:  {n_hit:4d}  ({n_hit/total:.1%})')
    print(f'  Complete miss: {n_miss:4d}  ({n_miss/total:.1%})')

    def summarize(name, arr):
        if not arr:
            print(f'  {name}: (no samples)')
            return
        a = np.array(arr)
        pct = lambda q: np.percentile(a, q)
        print(f'  {name}:  p10={pct(10):.3f}  p25={pct(25):.3f}  '
              f'p50={pct(50):.3f}  p75={pct(75):.3f}  p90={pct(90):.3f}  '
              f'mean={a.mean():.3f}')

    print(f'\n(3) Closest-approach of cue trajectory to ghost-ball (inches)')
    print(f'    [2R = {2*R:.2f} inch  is the contact threshold]')
    summarize('d_to_ghost(target pocket)', d_ghost_target)
    summarize('d_to_ghost(best pocket) ', d_ghost_best)

    print(f'\n    Histogram buckets (target-pocket ghost):')
    a = np.array(d_ghost_target)
    for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0),
                   (1.0, 2.0), (2.0, 4.0), (4.0, 10.0)]:
        n = ((a >= lo) & (a < hi)).sum()
        print(f'      {lo:4.2f} - {hi:4.2f} in: {n:4d}  ({n/len(a):.1%})')

    print(f'\n(4) Object ball trajectory analysis (hit-no-pocket shots)')
    print(f'    [pocket radius = 2.5 corner, 2.75 side; trajectory passing')
    print(f'     within that distance would have pocketed straight-line]')
    summarize('obj_d(target pocket)', obj_d_target)
    summarize('obj_d(best pocket) ', obj_d_best)

    print(f'\n    Of hit-no-pocket shots, how close was the theoretical')
    print(f'    straight-line ball trajectory to the NEAREST pocket?')
    ob = np.array(obj_d_best) if obj_d_best else np.array([0.0])
    pocket_thresh = 2.5
    would_pocket = (ob < pocket_thresh).sum()
    print(f'    Would have pocketed (traj < 2.5 in) straight-line:  '
          f'{would_pocket}/{len(ob)}  ({would_pocket/max(1,len(ob)):.1%})')
    print(f'    Implication: these shots hit a rail-bounce or lost signal')
    print(f'    that prevented the straight-line outcome.')

    print(f'\n    Pocket choice distribution (which pocket did the obj ball')
    print(f'    actually target, by closest-approach?):')
    names = ['TL', 'TS', 'TR', 'BL', 'BS', 'BR']
    for i, n in enumerate(obj_pocket_choice):
        print(f'      {names[i]}: {n}')


if __name__ == '__main__':
    main()
