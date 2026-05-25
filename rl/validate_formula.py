"""Overlay the proposed P(pocket) formula on the empirical curves to see
where it fits and where it diverges.

Formula:
    P = cut_func(theta) * dist_scale(x, y)
    cut_func(theta)  = 0.5 + 0.5 * cos(pi * sin(2 theta^2 / pi))
    dist_scale(x, y) = 1 - w * min(sqrt(x^2 + y^2) / D, 1)
    D = sqrt(100^2 + 50^2)        (table diagonal)
    w = 0.75                       (distance penalty weight)
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_pocket_prob import (
    pocket_rate, CUT_ANGLES, D_CB_VALUES, D_OP_VALUES,
    AIM_NOISE_DEG, FORCE_NOISE_PCT, SPIN_NOISE, BASE_SPIN, FORCE, N_TRIALS,
)

D_DIAG = math.hypot(100.0, 50.0)
W = 0.25  # so min scale = 1 - w = 0.75 at full diagonal


def cut_func(theta_rad):
    return 0.5 + 0.5 * math.cos(math.pi * math.sin(2 * theta_rad * theta_rad / math.pi))


def dist_scale(x, y, w=W):
    return 1.0 - w * min(math.hypot(x, y) / D_DIAG, 1.0)


def predict(cut_deg, d_cb, d_op):
    return cut_func(math.radians(cut_deg)) * dist_scale(d_cb, d_op)


def main():
    print(f'Computing empirical grid '
          f'({len(CUT_ANGLES)}×{len(D_CB_VALUES)}×{len(D_OP_VALUES)} cells, '
          f'{N_TRIALS} trials each)…')
    empirical = {}
    for d_op in D_OP_VALUES:
        for d_cb in D_CB_VALUES:
            empirical[(d_cb, d_op)] = [pocket_rate(c, d_cb, d_op) for c in CUT_ANGLES]
        print(f'  done d_op={d_op}')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    cmap = plt.colormaps['viridis']
    colors = [cmap(i / max(1, len(D_CB_VALUES) - 1)) for i in range(len(D_CB_VALUES))]

    for ax, d_op in zip(axes.flat, D_OP_VALUES):
        for d_cb, color in zip(D_CB_VALUES, colors):
            # Empirical: dots
            ys_emp = empirical[(d_cb, d_op)]
            xs_e = [c for c, y in zip(CUT_ANGLES, ys_emp) if y is not None]
            ys_e = [y for y in ys_emp if y is not None]
            ax.plot(xs_e, ys_e, marker='o', linestyle='', color=color,
                    markersize=6, label=f'cb={d_cb}″ empirical')
            # Formula: solid line on the same color
            xs_f = list(CUT_ANGLES)
            ys_f = [predict(c, d_cb, d_op) for c in xs_f]
            ax.plot(xs_f, ys_f, linestyle='-', color=color, alpha=0.8,
                    linewidth=2, label=f'cb={d_cb}″ formula')
        ax.set_title(f'OB → pocket = {d_op}″', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(CUT_ANGLES))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=8, ncol=2)

    for ax in axes[-1]:
        ax.set_xlabel('cut angle (°)')
    for ax in axes[:, 0]:
        ax.set_ylabel('P(pocket)')

    fig.suptitle(
        f'P(pocket): empirical (dots) vs formula (lines)  ·  w={W}\n'
        f'noise: aim={AIM_NOISE_DEG}° force={FORCE_NOISE_PCT*100:.0f}% '
        f'spin±{SPIN_NOISE}  ·  base spin={BASE_SPIN}  ·  force={FORCE:.0f}',
        fontsize=13)
    fig.tight_layout()
    out = 'pocket_prob_validate.png'
    fig.savefig(out, dpi=150)
    print(f'Saved → {out}')

    # Per-cell residual report.
    print('\n=== residuals (formula − empirical), positive = formula too high ===')
    abs_errs = []
    print(f'\n{"cut°":>4} ' + ' '.join(
        f'(cb={cb:2d},op={op:2d})' for op in D_OP_VALUES for cb in D_CB_VALUES))
    for i, cut in enumerate(CUT_ANGLES):
        row = []
        for op in D_OP_VALUES:
            for cb in D_CB_VALUES:
                emp = empirical[(cb, op)][i]
                if emp is None:
                    row.append('   ---  ')
                else:
                    pred = predict(cut, cb, op)
                    err = pred - emp
                    abs_errs.append(abs(err))
                    row.append(f' {err:+5.2f}  ')
        print(f'{cut:>4} ' + ''.join(row))
    if abs_errs:
        print(f'\nMean |error|: {np.mean(abs_errs):.3f}  '
              f'Max |error|: {max(abs_errs):.3f}  '
              f'95th %ile: {np.percentile(abs_errs, 95):.3f}')


if __name__ == '__main__':
    main()
