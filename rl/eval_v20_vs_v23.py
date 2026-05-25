"""Headless side-by-side eval of two PoolGameNet checkpoints under demo
settings (search_mc=8, prob_threshold=0.075, deterministic execution).
Uses the same random seed for each pair of games so the initial positions
are identical for both models.

Usage:
    python eval_v20_vs_v23.py [--n 30] [--max-shots 400]
"""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
import torch

from pool_game_net import PoolGameNet
from train_phase7 import Phase7Env
from shot_search_phase7 import shot_search_phase7


def play_one_game(ckpt_path, seed, max_shots=400,
                  search_k=8, search_m=8, search_mc=8,
                  prob_threshold=0.075,
                  embed_dim=128, num_heads=8, num_layers=4,
                  device='cpu', net_cache=None):
    """Play one full game from a fixed random seed. Returns dict with
    balls pocketed, rerack count, ending reason, # shots."""
    # Net cache lets us reuse a loaded checkpoint across many games.
    if net_cache is not None and ckpt_path in net_cache:
        net = net_cache[ckpt_path]
    else:
        net = PoolGameNet(embed_dim=embed_dim, num_heads=num_heads,
                          num_layers=num_layers).to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device,
                                        weights_only=True))
        net.eval()
        if net_cache is not None:
            net_cache[ckpt_path] = net

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = Phase7Env(max_shots=max_shots)
    n_shots = 0
    while not env.done:
        obs = env.get_obs()
        if not obs.shot_meta:
            break
        action = shot_search_phase7(
            net, env, obs,
            K_shots=search_k, M_per_shot=search_m,
            noise_samples=search_mc,
            prob_threshold=prob_threshold,
            device=device,
        )
        if action is None:
            break
        shot_idx, f_raw, s_raw = action
        env.step(int(shot_idx), float(f_raw), float(s_raw), obs)
        n_shots += 1

    return {
        'balls': env.total_pocketed,
        'reracks': env.rerack_count,
        'n_shots': n_shots,
        'reached_cap': n_shots >= max_shots,
    }


def stats(arr):
    arr = np.asarray(arr)
    return {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'max': int(arr.max()),
        'min': int(arr.min()),
        'std': float(arr.std()),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
    }


def hist_buckets(arr, edges=(0, 10, 30, 60, 100, 150, 250, 500)):
    arr = np.asarray(arr)
    counts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        c = int(((arr >= lo) & (arr < hi)).sum())
        counts.append((f'{lo}-{hi - 1}', c))
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=30,
                   help='games per model')
    p.add_argument('--max-shots', type=int, default=400)
    p.add_argument('--v20', default='checkpoints/phase7_p7_v20_hardcut_best.pt')
    p.add_argument('--v23', default='checkpoints/phase7_p7_v23_railbreak_best.pt')
    p.add_argument('--seed-base', type=int, default=1000)
    args = p.parse_args()

    print(f'eval: {args.n} games per model, max_shots={args.max_shots}, '
          f'seeds {args.seed_base}..{args.seed_base + args.n - 1}')
    print(f'  v20: {args.v20}')
    print(f'  v23: {args.v23}')
    net_cache = {}
    t0 = time.time()
    v20_runs, v23_runs = [], []
    paired = []
    for i in range(args.n):
        seed = args.seed_base + i
        r20 = play_one_game(args.v20, seed, args.max_shots, net_cache=net_cache)
        r23 = play_one_game(args.v23, seed, args.max_shots, net_cache=net_cache)
        v20_runs.append(r20)
        v23_runs.append(r23)
        paired.append((seed, r20['balls'], r23['balls']))
        elapsed = time.time() - t0
        print(f'  game {i+1:2d}/{args.n} seed={seed}: '
              f'v20={r20["balls"]:3d}b/{r20["reracks"]:2d}r  '
              f'v23={r23["balls"]:3d}b/{r23["reracks"]:2d}r  '
              f'({elapsed:.0f}s elapsed)', flush=True)

    v20_balls = [r['balls'] for r in v20_runs]
    v23_balls = [r['balls'] for r in v23_runs]
    v20_rer = [r['reracks'] for r in v20_runs]
    v23_rer = [r['reracks'] for r in v23_runs]

    print('\n=== Balls pocketed ===')
    for label, s in [('v20', stats(v20_balls)), ('v23', stats(v23_balls))]:
        print(f'  {label}: mean={s["mean"]:6.2f}  median={s["median"]:5.1f}  '
              f'p25={s["p25"]:5.1f}  p75={s["p75"]:5.1f}  '
              f'min={s["min"]:3d}  max={s["max"]:3d}  std={s["std"]:5.2f}')

    print('\n=== Reracks completed ===')
    for label, s in [('v20', stats(v20_rer)), ('v23', stats(v23_rer))]:
        print(f'  {label}: mean={s["mean"]:5.2f}  median={s["median"]:4.1f}  '
              f'max={s["max"]:2d}')

    print('\n=== Histogram of balls pocketed ===')
    h20 = hist_buckets(v20_balls)
    h23 = hist_buckets(v23_balls)
    print(f'  {"range":>8s}  {"v20":>4s}  {"v23":>4s}')
    for (lab20, c20), (lab23, c23) in zip(h20, h23):
        print(f'  {lab20:>8s}  {c20:>4d}  {c23:>4d}')

    # Paired comparison.
    diffs = [(b23 - b20) for _, b20, b23 in paired]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    tie = sum(1 for d in diffs if d == 0)
    print('\n=== Paired comparison (same starting position) ===')
    print(f'  v23 > v20 in {pos}/{args.n} games')
    print(f'  v23 < v20 in {neg}/{args.n} games')
    print(f'  v23 = v20 in {tie}/{args.n} games')
    print(f'  mean diff (v23 - v20): {float(np.mean(diffs)):+.2f} balls')
    print(f'  median diff (v23 - v20): {float(np.median(diffs)):+.2f} balls')

    print(f'\ntotal wall-clock: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
