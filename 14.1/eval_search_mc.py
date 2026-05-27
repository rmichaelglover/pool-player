"""Compare raw policy vs depth-1 search vs MC-noise-search on noisy env.

Question being tested: under noisy execution, does averaging each search
candidate's Q over independent noise samples (so search prefers robust shots,
not deterministic-frontier shots) recover the gains we usually see on the
deterministic env?

Prior eval (search vs raw on noisy env, noise_samples=1) showed search HURT:
  longrun  raw 5.12 → search 4.80, AvgCut 24° → 40°
  noise_v2 raw 6.25 → search 5.05, AvgCut 24° → 40°
i.e. search was promoting harder shots whose single deterministic rollout
looked good. MC averaging should fix that.
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
from pool_game_net import PoolGameNet, decode_force, decode_spin
from train_phase7 import Phase7Env
from shot_search_phase7 import shot_search_phase7, _obs_to_batch


def run_episode(net, env_factory, mode, seed, K=8, M=8, noise_samples=1, device='cpu'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = env_factory()
    cuts = []
    while not env.done:
        obs = env.get_obs()
        if mode == 'raw':
            batch = _obs_to_batch(obs, device)
            with torch.no_grad():
                idx, f, s, _, _ = net.get_action(batch, deterministic=True)
            shot_idx, f_raw, s_raw = int(idx.item()), float(f.item()), float(s.item())
        else:
            action = shot_search_phase7(net, env, obs, K_shots=K, M_per_shot=M,
                                         device=device, noise_samples=noise_samples)
            if action is None:
                env.done = True
                break
            shot_idx, f_raw, s_raw = action
        if shot_idx < len(obs.shot_meta):
            cuts.append(obs.shot_meta[shot_idx].cut_angle_deg)
        env.step(shot_idx, f_raw, s_raw, obs)
    return env.total_pocketed, (float(np.mean(cuts)) if cuts else 0.0)


def evaluate(label, ckpt_path, env_factory, N=50, K=8, M=8, device='cpu',
             modes=None):
    """modes: list of (label, mode_name, noise_samples_kwarg)."""
    if modes is None:
        modes = [('raw', 'raw', 1), ('search MC=1', 'search', 1),
                 ('search MC=4', 'search', 4), ('search MC=8', 'search', 8)]
    net = PoolGameNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()
    print(f'\n── {label} ── ckpt={os.path.basename(ckpt_path)} N={N} K={K} M={M}')
    print(f'{"mode":15s} {"mean":>6s} {"med":>4s} {"max":>4s} {"≥14":>4s} {"avgCut":>7s} {"sec":>6s}')
    for mode_label, mode, ns in modes:
        t0 = time.time()
        runs = []; cuts_all = []
        for seed in range(N):
            r, c = run_episode(net, env_factory, mode=mode, seed=seed,
                               K=K, M=M, noise_samples=ns, device=device)
            runs.append(r); cuts_all.append(c)
        runs = np.array(runs); cuts = np.array(cuts_all)
        elapsed = time.time() - t0
        print(f'{mode_label:15s} {runs.mean():>6.2f} {int(np.median(runs)):>4d} '
              f'{int(runs.max()):>4d} {int((runs>=14).sum()):>4d} '
              f'{cuts.mean():>6.1f}° {elapsed:>5.0f}s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=50)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--M', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--aim_noise_deg', type=float, default=0.10)
    p.add_argument('--force_noise_pct', type=float, default=0.03)
    p.add_argument('--spin_noise', type=float, default=0.05)
    p.add_argument('--ckpts', nargs='+', default=[
        'checkpoints/phase7_p7_longrun_best.pt',
        'checkpoints/phase7_p7_noise_v2_best.pt',
    ])
    p.add_argument('--max_shots', type=int, default=60)
    p.add_argument('--include_det', action='store_true',
                   help='Also run det-env raw + search MC=1 (sanity)')
    args = p.parse_args()

    rl_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(rl_dir)

    def noisy_env():
        return Phase7Env(max_shots=args.max_shots,
                          aim_noise_deg=args.aim_noise_deg,
                          force_noise_pct=args.force_noise_pct,
                          spin_noise=args.spin_noise)
    def det_env():
        return Phase7Env(max_shots=args.max_shots)

    print(f'Noisy env: aim={args.aim_noise_deg}° force={args.force_noise_pct*100:.1f}% '
          f'spin={args.spin_noise}')

    for ckpt in args.ckpts:
        if not os.path.exists(ckpt):
            print(f'SKIP missing: {ckpt}')
            continue
        if args.include_det:
            evaluate(f'{os.path.basename(ckpt)} on DET env', ckpt, det_env,
                     N=args.N, K=args.K, M=args.M, device=args.device,
                     modes=[('raw', 'raw', 1), ('search', 'search', 1)])
        evaluate(f'{os.path.basename(ckpt)} on NOISY env', ckpt, noisy_env,
                 N=args.N, K=args.K, M=args.M, device=args.device)


if __name__ == '__main__':
    main()
