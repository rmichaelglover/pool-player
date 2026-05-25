"""Empirical study: P(pocket) as a function of cut angle, cue→ghost distance,
and OB→pocket distance, under the same noise the agent trains under.

Layout: TL corner pocket (aim point ≈ (3, 3)). OB placed along the TL→BR
diagonal of the playing surface so larger d_op fits without going off-table.
For each cut angle, try both rotation directions for the cue and pick the one
that keeps the cue ball furthest inside the table (so steep cuts at large
distances still work).

Generates a 2x2 grid of plots — one per d_op ∈ {10, 20, 30, 40} — each
showing P(pocket) vs cut angle for d_cb ∈ {10, 20, 30, 40, 50}.
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pool_sim import simulate_shot

R = 1.125  # ball radius

# Noise levels representative of a SKILLED player.
AIM_NOISE_DEG    = 0.2
FORCE_NOISE_PCT  = 0.02
SPIN_NOISE       = 0.02
# Light draw spin to kill cue follow-through and prevent scratches on
# straight shots into corner pockets — what any real player would do.
BASE_SPIN        = -0.5

CUT_ANGLES   = list(range(0, 71, 5))
D_CB_VALUES  = [10, 20, 30, 40, 50]
D_OP_VALUES  = [10, 20, 30, 40]
FORCE        = 100.0
N_TRIALS     = 300

# TL pocket aim point (well inside the corner throat).
POCKET_AIM = (3.0, 3.0)
# OB direction along TL→BR diagonal of the 100×50 playing surface.
_DIAG_LEN = math.hypot(100 - 6, 50 - 6)
OB_DIR    = ((100 - 6) / _DIAG_LEN, (50 - 6) / _DIAG_LEN)


def _rotate(v, theta_rad):
    cs, sn = math.cos(theta_rad), math.sin(theta_rad)
    return (v[0] * cs - v[1] * sn, v[0] * sn + v[1] * cs)


def setup_shot(cut_deg, d_cb, d_op):
    """Returns (cue_pos, ob_pos, aim_dir) or None if neither rotation keeps
    the cue ball on the table. Picks the rotation that puts cue furthest
    from any rail."""
    theta = math.radians(cut_deg)
    ob_pos = (POCKET_AIM[0] + d_op * OB_DIR[0],
              POCKET_AIM[1] + d_op * OB_DIR[1])
    # Ghost ball: 2R from OB on the side AWAY from the pocket.
    gb = (ob_pos[0] + 2 * R * OB_DIR[0],
          ob_pos[1] + 2 * R * OB_DIR[1])
    # OB→pocket direction (negative of OB_DIR).
    op_dir = (-OB_DIR[0], -OB_DIR[1])

    best = None
    best_margin = -1.0
    for sign in (+1, -1):
        cb_to_gb = _rotate(op_dir, sign * theta)
        cue = (gb[0] - d_cb * cb_to_gb[0], gb[1] - d_cb * cb_to_gb[1])
        # Distance from nearest table edge.
        m = min(cue[0] - R, 100 - R - cue[0],
                cue[1] - R, 50 - R - cue[1])
        if m > best_margin:
            best_margin = m
            best = (cue, ob_pos, cb_to_gb)
    if best_margin < 0:
        return None
    return best


def trial(cue_pos, ob_pos, aim_dir, force, rng):
    aim_dx, aim_dy = aim_dir
    aim_ang = math.atan2(aim_dy, aim_dx)
    aim_ang += rng.normal() * AIM_NOISE_DEG * (math.pi / 180.0)
    f = force * (1.0 + rng.normal() * FORCE_NOISE_PCT)
    f = max(20.0, min(280.0, f))
    spin = BASE_SPIN + rng.normal() * SPIN_NOISE
    spin = max(-2.5, min(2.5, spin))
    ax = math.cos(aim_ang); ay = math.sin(aim_ang)
    r = simulate_shot(cue_pos, {1: ob_pos}, ax * f, ay * f, spin, ax, ay)
    return (1 in r.pocketed_ids) and not r.cue_scratched


def pocket_rate(cut_deg, d_cb, d_op, n=N_TRIALS):
    setup = setup_shot(cut_deg, d_cb, d_op)
    if setup is None:
        return None
    cue_pos, ob_pos, aim_dir = setup
    rng = np.random.default_rng(seed=hash((cut_deg, d_cb, d_op)) & 0xFFFFFFFF)
    hits = sum(trial(cue_pos, ob_pos, aim_dir, FORCE, rng) for _ in range(n))
    return hits / n


def main():
    n_cells = len(CUT_ANGLES) * len(D_CB_VALUES) * len(D_OP_VALUES)
    print(f'Computing {n_cells} cells, {N_TRIALS} trials each…')
    results = {}
    for d_op in D_OP_VALUES:
        for d_cb in D_CB_VALUES:
            curve = [pocket_rate(c, d_cb, d_op) for c in CUT_ANGLES]
            results[(d_cb, d_op)] = curve
        print(f'  done d_op={d_op}')

    # 2×2 grid of subplots, one per d_op.
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    cmap = plt.colormaps['viridis']
    colors = [cmap(i / max(1, len(D_CB_VALUES) - 1)) for i in range(len(D_CB_VALUES))]

    for ax, d_op in zip(axes.flat, D_OP_VALUES):
        for d_cb, color in zip(D_CB_VALUES, colors):
            ys_full = results[(d_cb, d_op)]
            xs = [c for c, y in zip(CUT_ANGLES, ys_full) if y is not None]
            ys = [y for y in ys_full if y is not None]
            if xs:
                ax.plot(xs, ys, marker='o', color=color,
                        label=f'cue→ghost = {d_cb}″')
        ax.set_title(f'OB → pocket = {d_op}″', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(CUT_ANGLES))
        ax.grid(True, alpha=0.3)
        ax.axvline(22.5, color='gray', linestyle=':', alpha=0.5)
        ax.legend(loc='lower left', fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel('cut angle (°)')
    for ax in axes[:, 0]:
        ax.set_ylabel('P(pocket)')

    fig.suptitle(
        f'P(pocket) vs cut angle  ·  TL corner pocket  ·  {N_TRIALS} trials/cell\n'
        f'noise: aim={AIM_NOISE_DEG}° force={FORCE_NOISE_PCT*100:.0f}% '
        f'spin±{SPIN_NOISE}  ·  base spin={BASE_SPIN} (light draw)  ·  force={FORCE:.0f}',
        fontsize=13)
    fig.tight_layout()
    out = 'pocket_prob_grid.png'
    fig.savefig(out, dpi=150)
    print(f'Saved → {out}')

    # Print a clean table per d_op.
    for d_op in D_OP_VALUES:
        print(f'\n=== d_op = {d_op}″ ===')
        header = '  cut° | ' + ' | '.join(f'cb={cb:>2}' for cb in D_CB_VALUES)
        print(header)
        print('-' * len(header))
        for i, cut in enumerate(CUT_ANGLES):
            row = []
            for d_cb in D_CB_VALUES:
                v = results[(d_cb, d_op)][i]
                row.append('  --  ' if v is None else f' {v:>5.2f}')
            print(f'  {cut:>3}° | ' + ' | '.join(row))


if __name__ == '__main__':
    main()
