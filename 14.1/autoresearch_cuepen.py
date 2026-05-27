"""Autoresearch sweep: cue-ricochet penalty × OB-scatter penalty.

Each run is a fine-tune from phase7_p7_eor_x2_best.pt (rolling 5.50 baseline)
with three new penalty terms wired into Phase7Env._shape_bonus:

  - cue_movement_penalty_weight = 0.1 (FIXED): cue-ball post-contact path /100".
  - cue_ricochet_penalty_weight  ∈ {0.0, 0.5, 1.0}: cost per extra cue→OB contact.
  - movement_penalty_weight       ∈ {1.0, 1.5, 2.0}: OB-scatter sum-displacement /50".

3 × 3 = 9 runs, 800 iters each. Pool: 4 parallel (4 torch threads/run on 16-core box).

Tracks per-run:
  - rolling reward (final + best)
  - mean cue path length (last ~100 logged iters)
  - mean cue contacts per shot (last ~100 logged iters)

Usage:  python3 autoresearch_cuepen.py
Logs:   logs/autoresearch_cuepen/<tag>.log
Result: autoresearch_cuepen_summary.txt
"""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / 'logs' / 'autoresearch_cuepen'
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = HERE / 'autoresearch_cuepen_summary.txt'

WARM_START = 'checkpoints/phase7_p7_eor_x2_best.pt'
ITERS = 800
THREADS_PER_RUN = 4
MAX_PARALLEL = 4

# Hparam grid.
RICOCHETS = [0.0, 0.5, 1.0]
MOV_PENS  = [1.0, 1.5, 2.0]

# Held-fixed hparams (matching the eor_x2 SOTA recipe + new cue_movement at 0.1).
FIXED = dict(
    lr=5e-5,
    cue_movement_penalty_weight=0.1,
    envs=16, steps_per_update=32, entropy_coef=0.01,
    shape_bonus_max=2.0,
    eor_bonus_max=4.0,
    aim_noise_deg=0.5, force_noise_pct=0.05, spin_noise=0.05,
)


def tag_for(ricochet, mov_pen):
    return f'sweep_ric{ricochet:g}_mov{mov_pen:g}'


def run_one(ricochet, mov_pen):
    """Launch one training run, block until done, return summary dict."""
    tag = tag_for(ricochet, mov_pen)
    log_path = LOG_DIR / f'{tag}.log'
    cmd = [
        'python3', '-u', str(HERE / 'train_phase7.py'),
        '--warm', WARM_START,
        '--tag', tag,
        '--iters', str(ITERS),
        '--lr', f'{FIXED["lr"]:g}',
        '--cue_movement_penalty_weight', f'{FIXED["cue_movement_penalty_weight"]:g}',
        '--cue_ricochet_penalty_weight', f'{ricochet:g}',
        '--movement_penalty_weight', f'{mov_pen:g}',
        '--envs', str(FIXED['envs']),
        '--steps_per_update', str(FIXED['steps_per_update']),
        '--entropy_coef', f'{FIXED["entropy_coef"]:g}',
        '--shape_bonus_max', f'{FIXED["shape_bonus_max"]:g}',
        '--eor_bonus_max', f'{FIXED["eor_bonus_max"]:g}',
        '--aim_noise_deg', f'{FIXED["aim_noise_deg"]:g}',
        '--force_noise_pct', f'{FIXED["force_noise_pct"]:g}',
        '--spin_noise', f'{FIXED["spin_noise"]:g}',
    ]
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(THREADS_PER_RUN)
    env['MKL_NUM_THREADS'] = str(THREADS_PER_RUN)
    t0 = time.time()
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env=env, cwd=str(HERE))
    elapsed = time.time() - t0
    return dict(
        tag=tag, ricochet=ricochet, mov_pen=mov_pen,
        rc=proc.returncode, elapsed_s=elapsed,
        final_rolling=parse_metric(log_path, r'Rolling=\s*(-?\d+\.\d+)', 'last'),
        best_rolling=parse_metric(log_path, r'Rolling=\s*(-?\d+\.\d+)', 'max') or
                     parse_best_rolling(log_path),
        last_cue_path=parse_metric(log_path, r'CuePath=\s*(-?\d+\.\d+)', 'last10mean'),
        last_contacts=parse_metric(log_path, r'Contacts=\s*(-?\d+\.\d+)', 'last10mean'),
        log=str(log_path),
    )


def parse_metric(log_path, pattern, mode):
    """Extract a metric series from log. mode ∈ {last, max, last10mean}."""
    pat = re.compile(pattern)
    vals = []
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                vals.append(float(m.group(1)))
    if not vals:
        return None
    if mode == 'last':
        return vals[-1]
    if mode == 'max':
        return max(vals)
    if mode == 'last10mean':
        last = vals[-10:]
        return sum(last) / len(last)
    raise ValueError(mode)


def parse_best_rolling(log_path):
    pat = re.compile(r'Best rolling avg run:\s*(-?\d+\.\d+)')
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                return float(m.group(1))
    return None


def main():
    grid = [(r, m) for r in RICOCHETS for m in MOV_PENS]
    print(f'[autoresearch] {len(grid)} runs, max_parallel={MAX_PARALLEL}, '
          f'iters={ITERS} per run, threads/run={THREADS_PER_RUN}', flush=True)
    print(f'[autoresearch] grid: ricochet={RICOCHETS}  mov_pen={MOV_PENS}', flush=True)
    print(f'[autoresearch] fixed: cue_move=0.1 lr=5e-5 shape=2.0 eor=4.0', flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = {ex.submit(run_one, r, m): (r, m) for r, m in grid}
        for fut in as_completed(futures):
            r_, m_ = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f'[done] {res["tag"]:<26} '
                      f'final={res["final_rolling"]:.2f} '
                      f'best={res["best_rolling"]:.2f} '
                      f'cue={res["last_cue_path"] or 0:.1f} '
                      f'cnt={res["last_contacts"] or 0:.2f} '
                      f't={res["elapsed_s"]:.0f}s rc={res["rc"]}',
                      flush=True)
            except Exception as e:
                print(f'[FAIL] ricochet={r_} mov_pen={m_}: {e}', flush=True)

    # Sort by best rolling, descending.
    results.sort(key=lambda r: (r['best_rolling'] or -1), reverse=True)
    total = time.time() - t0

    with open(SUMMARY, 'w') as f:
        f.write(f'Autoresearch sweep: cue-ricochet × OB-scatter penalty\n')
        f.write(f'{ITERS} iters/run, {len(grid)} runs, total {total:.0f}s\n')
        f.write(f'Baseline: phase7_p7_eor_x2_best.pt rolling 5.50\n')
        f.write(f'Fixed: cue_move_pen=0.1, lr=5e-5, shape=2.0, eor=4.0\n')
        f.write(f'\n{"tag":<26} {"ric":>5} {"mov":>5} '
                f'{"final":>7} {"best":>7} {"cue″":>7} {"cnt":>5} '
                f'{"t(s)":>7} {"rc":>3}\n')
        f.write('-' * 90 + '\n')
        for r in results:
            f.write(f'{r["tag"]:<26} {r["ricochet"]:>5.2f} {r["mov_pen"]:>5.2f} '
                    f'{(r["final_rolling"] or -1):>7.2f} '
                    f'{(r["best_rolling"] or -1):>7.2f} '
                    f'{(r["last_cue_path"] or -1):>7.1f} '
                    f'{(r["last_contacts"] or -1):>5.2f} '
                    f'{r["elapsed_s"]:>7.0f} {r["rc"]:>3d}\n')
    print(f'\n[autoresearch] complete in {total:.0f}s. Summary → {SUMMARY}',
          flush=True)
    with open(SUMMARY) as f:
        sys.stdout.write(f.read())


if __name__ == '__main__':
    main()
