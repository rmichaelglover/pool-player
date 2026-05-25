"""Visualize the geometry of a specific shot setup, plus the simulator's
empirical pocket rate so we can see why the formula and simulator disagree.

Default: the close, hard-cut shot (Dc=10, Do=10, cut=70°) where the formula
predicts ~8% but the simulator hits ~40%.
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from pool_sim import simulate_shot

R = 1.125
D_DIAG = math.hypot(100.0, 50.0)
W = 0.25

POCKET_AIM = (3.0, 3.0)
OB_DIR = ((100 - 6) / math.hypot(94, 44),
          (50 - 6) / math.hypot(94, 44))


def cut_func(theta_rad):
    return 0.5 + 0.5 * math.cos(math.pi * math.sin(2 * theta_rad * theta_rad / math.pi))


def dist_scale(x, y):
    return 1.0 - W * min(math.hypot(x, y) / D_DIAG, 1.0)


def setup_shot(cut_deg, d_cb, d_op):
    theta = math.radians(cut_deg)
    ob = (POCKET_AIM[0] + d_op * OB_DIR[0],
          POCKET_AIM[1] + d_op * OB_DIR[1])
    gb = (ob[0] + 2 * R * OB_DIR[0], ob[1] + 2 * R * OB_DIR[1])
    op_dir = (-OB_DIR[0], -OB_DIR[1])
    best, best_m = None, -1e9
    for sign in (+1, -1):
        cs, sn = math.cos(sign * theta), math.sin(sign * theta)
        cb_to_gb = (op_dir[0] * cs - op_dir[1] * sn,
                    op_dir[0] * sn + op_dir[1] * cs)
        cue = (gb[0] - d_cb * cb_to_gb[0], gb[1] - d_cb * cb_to_gb[1])
        m = min(cue[0] - R, 100 - R - cue[0], cue[1] - R, 50 - R - cue[1])
        if m > best_m:
            best_m, best = m, (cue, ob, gb, cb_to_gb)
    return best


def empirical_rate(cue, ob, aim_dir, n=400, base_spin=-0.5):
    """Run noisy trials, return (made_rate, scratch_rate)."""
    aim_dx, aim_dy = aim_dir
    aim_ang0 = math.atan2(aim_dy, aim_dx)
    rng = np.random.default_rng(42)
    n_made = n_scr = 0
    for _ in range(n):
        ang = aim_ang0 + rng.normal() * 0.2 * math.pi / 180.0
        f = 100.0 * (1.0 + rng.normal() * 0.02)
        spin = base_spin + rng.normal() * 0.02
        ax, ay = math.cos(ang), math.sin(ang)
        r = simulate_shot(cue, {1: ob}, ax * f, ay * f, spin, ax, ay)
        if r.cue_scratched: n_scr += 1
        if 1 in r.pocketed_ids and not r.cue_scratched: n_made += 1
    return n_made / n, n_scr / n


def draw_table(ax):
    # Slate (with overhang) — light gray
    ax.add_patch(Rectangle((-3.5, -3.5), 107, 57,
                            facecolor='#bdbdbd', edgecolor='black', lw=1))
    # Playing surface — green
    ax.add_patch(Rectangle((0, 0), 100, 50,
                            facecolor='#1b6b3a', edgecolor='black', lw=2))
    # Pocket capture circles
    for (cx, cy, rad) in [
        (-3.5, -3.5, 6.21), (50, -12.625, 12.375), (103.5, -3.5, 6.21),
        (-3.5, 53.5, 6.21), (50, 62.625, 12.375), (103.5, 53.5, 6.21),
    ]:
        ax.add_patch(Circle((cx, cy), rad, facecolor='black', alpha=0.3))


def draw_shot(ax, cue, ob, gb, aim_dir, cut_deg, d_cb, d_op, p_formula, p_sim):
    # Aim line: cue → ghost ball → extended (where cue *would* go absent OB).
    ext = (cue[0] + (gb[0] - cue[0]) * 1.4, cue[1] + (gb[1] - cue[1]) * 1.4)
    ax.plot([cue[0], ext[0]], [cue[1], ext[1]],
            color='gold', linestyle='--', linewidth=2, label='aim (CB→GB)')
    # Pocket line: OB → pocket aim.
    ax.plot([ob[0], POCKET_AIM[0]], [ob[1], POCKET_AIM[1]],
            color='red', linestyle=':', linewidth=2, label='OB→pocket')
    # Distances along the lines.
    mid_cb = ((cue[0] + gb[0]) / 2, (cue[1] + gb[1]) / 2)
    mid_op = ((ob[0] + POCKET_AIM[0]) / 2, (ob[1] + POCKET_AIM[1]) / 2)
    ax.annotate(f'Dc = {d_cb}″', mid_cb, fontsize=10, color='gold',
                weight='bold', ha='center')
    ax.annotate(f'Do = {d_op}″', mid_op, fontsize=10, color='red',
                weight='bold', ha='center')
    # Balls.
    ax.add_patch(Circle(cue, R, facecolor='white', edgecolor='black', lw=1.5))
    ax.add_patch(Circle(ob, R, facecolor='#ffaa44', edgecolor='black', lw=1.5))
    ax.add_patch(Circle(gb, R, facecolor='none', edgecolor='gold',
                         linestyle='--', lw=1.5))
    ax.annotate('cue', cue, fontsize=9, ha='center', va='center')
    ax.annotate('OB', ob, fontsize=9, ha='center', va='center')
    ax.annotate('ghost', (gb[0], gb[1] + 2.2), fontsize=8, color='gold',
                ha='center')
    ax.set_title(
        f'cut = {cut_deg}°,  Dc = {d_cb}″,  Do = {d_op}″\n'
        f'Formula P = {p_formula:.2f}    Simulator P = {p_sim:.2f}',
        fontsize=11)


def main():
    cases = [
        (0,  10, 10),
        (70, 10, 10),
        (45, 10, 10),
        (70, 30, 30),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (cut, dcb, dop) in zip(axes.flat, cases):
        cue, ob, gb, aim = setup_shot(cut, dcb, dop)
        p_formula = cut_func(math.radians(cut)) * dist_scale(dcb, dop)
        p_sim, p_scr = empirical_rate(cue, ob, aim)
        draw_table(ax)
        draw_shot(ax, cue, ob, gb, aim, cut, dcb, dop, p_formula, p_sim)
        ax.set_xlim(-6, 110)
        ax.set_ylim(-6, 56)
        ax.set_aspect('equal')
        ax.set_xlabel('x (inches)')
        ax.legend(loc='upper right', fontsize=9)
        # Print to stdout too.
        print(f'cut={cut}°  Dc={dcb}  Do={dop}  '
              f'cue=({cue[0]:.1f},{cue[1]:.1f})  '
              f'ob=({ob[0]:.1f},{ob[1]:.1f})  '
              f'P_formula={p_formula:.3f}  P_sim={p_sim:.3f}  '
              f'scratch={p_scr:.3f}')

    fig.suptitle('Shot geometry comparison: formula vs simulator',
                 fontsize=13)
    fig.tight_layout()
    out = 'shot_geometry.png'
    fig.savefig(out, dpi=140)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
