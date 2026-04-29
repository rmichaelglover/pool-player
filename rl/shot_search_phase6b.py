"""
Shot-level search for Phase 6b (14.1 continuous with rerack).

Same depth-limited search as Phase 6, but uses Phase6bEnv. The rerack
mechanic is handled transparently inside env.step(), so search doesn't
need to know about it.
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from train_phase4 import decode_action, ACT_DIM
from train_phase6b import Phase6bEnv


def clone_phase6b_env(env: Phase6bEnv) -> Phase6bEnv:
    new_env = Phase6bEnv.__new__(Phase6bEnv)
    new_env.pocket_reward = env.pocket_reward
    new_env.max_shots = env.max_shots
    new_env.call_shot = getattr(env, 'call_shot', True)
    new_env.cue = list(env.cue)
    new_env.balls = {bid: list(pos) for bid, pos in env.balls.items()}
    new_env.shot_idx = env.shot_idx
    new_env.done = env.done
    new_env.rerack_count = env.rerack_count
    new_env.total_pocketed = env.total_pocketed
    new_env.is_break_shot = getattr(env, 'is_break_shot', False)
    return new_env


def _policy_samples(net, obs_t, K, device, std_floor=0.1):
    with torch.no_grad():
        action_mean, _ = net(obs_t)
        std = torch.exp(net.log_std).clamp(min=std_floor)
        noise = Normal(torch.zeros_like(std), std).sample((K - 1,))
        samples = torch.cat([action_mean, action_mean + noise], dim=0)
    return samples


def _best_q(net, env, K_list, gamma, device, std_floor):
    if env.done:
        return None, 0.0
    obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
    if not K_list:
        with torch.no_grad():
            _, value = net(obs_t)
        return None, float(value.item())
    K = K_list[0]; rest = K_list[1:]
    samples = _policy_samples(net, obs_t, K, device, std_floor)
    best_q = -float('inf'); best_a = None
    for a in samples:
        aim, force, spin = decode_action(a.cpu().numpy())
        env_c = clone_phase6b_env(env)
        _, r, done, _ = env_c.step(aim, force, spin)
        if done:
            q = r
        else:
            _, future = _best_q(net, env_c, rest, gamma, device, std_floor)
            q = r + gamma * future
        if q > best_q:
            best_q = q; best_a = a
    return best_a, best_q


def shot_search_phase6b(net, env, K_per_depth=(24, 16), gamma=0.99,
                        device='cpu', std_floor=0.1):
    assert not env.done
    best_a, _ = _best_q(net, env, list(K_per_depth), gamma, device, std_floor)
    return best_a.cpu().numpy()


def eval_phase6b(ckpt_path, embed_dim=96, num_heads=6, num_layers=4, ff_dim=192,
                 N=50, K_per_depth=(24, 16), device='cpu', max_shots=50):
    device = torch.device(device)
    net = PoolAttentionNet(embed_dim=embed_dim, num_heads=num_heads,
                           num_layers=num_layers, ff_dim=ff_dim,
                           act_dim=ACT_DIM).to(device)
    net.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5).to(device))
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()

    def run_episode(env, use_search):
        while not env.done:
            if use_search:
                a = shot_search_phase6b(net, env, K_per_depth=K_per_depth, device=device)
            else:
                obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
                with torch.no_grad():
                    act, _, _ = net.get_action(obs_t, deterministic=True)
                a = act[0].cpu().numpy()
            aim, force, spin = decode_action(a)
            env.step(aim, force, spin)
        return env.total_pocketed, env.rerack_count

    import random
    raw_runs, srch_runs = [], []
    raw_reracks, srch_reracks = [], []
    t0 = time.time()
    for i in range(N):
        seed = i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase6bEnv(max_shots=max_shots)
        r, k = run_episode(env, False); raw_runs.append(r); raw_reracks.append(k)

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase6bEnv(max_shots=max_shots)
        r, k = run_episode(env, True); srch_runs.append(r); srch_reracks.append(k)

    elapsed = time.time() - t0
    print(f'Checkpoint: {ckpt_path}')
    print(f'N={N} episodes  Time: {elapsed:.1f}s  max_shots={max_shots}')
    raw = np.array(raw_runs); srch = np.array(srch_runs)
    print(f'{"Strategy":24s} {"Mean":>6s} {"Median":>7s} {"Max":>4s} {"≥14":>4s} {"≥28":>4s} {"≥42":>4s} {"reracks_max":>12s}')
    print(f'{"raw policy":24s} {raw.mean():>6.2f} {int(np.median(raw)):>7d} {int(raw.max()):>4d} '
          f'{(raw >= 14).sum():>4d} {(raw >= 28).sum():>4d} {(raw >= 42).sum():>4d} '
          f'{int(max(raw_reracks)):>12d}')
    lbl = f'search K={K_per_depth}'
    print(f'{lbl:24s} {srch.mean():>6.2f} {int(np.median(srch)):>7d} {int(srch.max()):>4d} '
          f'{(srch >= 14).sum():>4d} {(srch >= 28).sum():>4d} {(srch >= 42).sum():>4d} '
          f'{int(max(srch_reracks)):>12d}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--N', type=int, default=50)
    p.add_argument('--K1', type=int, default=24)
    p.add_argument('--K2', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max_shots', type=int, default=50)
    p.add_argument('--embed_dim', type=int, default=96)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--ff_dim', type=int, default=192)
    args = p.parse_args()
    eval_phase6b(args.ckpt, embed_dim=args.embed_dim, num_heads=args.num_heads,
                 num_layers=args.num_layers, ff_dim=args.ff_dim,
                 N=args.N, K_per_depth=(args.K1, args.K2), device=args.device,
                 max_shots=args.max_shots)
