"""
Shot-level search for Phase 6 (14.1 run-out, up to 15 shots).

Variable-depth episodes so we can't fully enumerate. Use a depth-limited
search with the value network as leaf estimate:

    Q(a) = r(a) + γ · V(s')            if depth == limit or s' terminal
    Q(a) = r(a) + γ · max_{a'} Q(a')   otherwise

Default params: depth=2, K=(24, 16). ~400 sims per decision (~10ms on our
0.06ms/shot sim). Up to 15 decisions per episode = ~6K sims per episode.
"""
from __future__ import annotations
import math
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
from train_phase6 import Phase6Env


def clone_phase6_env(env: Phase6Env) -> Phase6Env:
    new_env = Phase6Env.__new__(Phase6Env)
    new_env.pocket_reward = env.pocket_reward
    new_env.max_shots = env.max_shots
    new_env.cue = list(env.cue)
    new_env.balls = {bid: list(pos) for bid, pos in env.balls.items()}
    new_env.shot_idx = env.shot_idx
    new_env.done = env.done
    return new_env


def _policy_samples_with_value(net, obs_t, K, device, std_floor=0.1):
    """Sample K candidate actions (first = mean). Also returns V(s)."""
    with torch.no_grad():
        action_mean, value = net(obs_t)
        std = torch.exp(net.log_std).clamp(min=std_floor)
        noise = Normal(torch.zeros_like(std), std).sample((K - 1,))
        samples = torch.cat([action_mean, action_mean + noise], dim=0)
    return samples, float(value.item())


def _best_q_limited(net, env: Phase6Env, K_list, gamma, device, std_floor):
    """Recursive depth-limited search. K_list[i] is candidates at depth i.
    At K_list empty (depth limit) OR env.done, fall back to V(s)."""
    if env.done:
        return None, 0.0
    obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
    if not K_list:
        with torch.no_grad():
            _, value = net(obs_t)
        return None, float(value.item())

    K = K_list[0]
    rest = K_list[1:]
    samples, _ = _policy_samples_with_value(net, obs_t, K, device, std_floor)

    best_q = -float('inf')
    best_a = None
    for a in samples:
        aim, force, spin = decode_action(a.cpu().numpy())
        env_c = clone_phase6_env(env)
        _, r, done, _ = env_c.step(aim, force, spin)
        if done:
            q = r
        else:
            _, future = _best_q_limited(net, env_c, rest, gamma, device, std_floor)
            q = r + gamma * future
        if q > best_q:
            best_q = q
            best_a = a
    return best_a, best_q


def shot_search_phase6(net, env, K_per_depth=(24, 16), gamma=0.99,
                       device='cpu', std_floor=0.1):
    """Pick best action for env's current shot via depth-limited search."""
    assert not env.done
    best_a, _ = _best_q_limited(net, env, list(K_per_depth), gamma, device, std_floor)
    return best_a.cpu().numpy()


def eval_phase6(ckpt_path: str, embed_dim=96, num_heads=6, num_layers=4,
                ff_dim=192, N=50, K_per_depth=(24, 16), device='cpu'):
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
                a = shot_search_phase6(net, env, K_per_depth=K_per_depth, device=device)
            else:
                obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
                with torch.no_grad():
                    act, _, _ = net.get_action(obs_t, deterministic=True)
                a = act[0].cpu().numpy()
            aim, force, spin = decode_action(a)
            _, _, _, info = env.step(aim, force, spin)
        return 15 - len(env.balls)

    import random
    raw_runs, srch_runs = [], []
    t0 = time.time()
    for i in range(N):
        seed = i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase6Env()
        raw_runs.append(run_episode(env, use_search=False))

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase6Env()
        srch_runs.append(run_episode(env, use_search=True))

    elapsed = time.time() - t0
    print(f'Checkpoint: {ckpt_path}')
    print(f'N={N} episodes  Time: {elapsed:.1f}s')
    raw = np.array(raw_runs); srch = np.array(srch_runs)
    print(f'{"Strategy":24s} {"Mean":>6s} {"Median":>7s} {"Max":>4s} {"≥5":>4s} {"≥10":>4s} {"=15":>4s}')
    print(f'{"raw policy":24s} {raw.mean():>6.2f} {int(np.median(raw)):>7d} {raw.max():>4d} '
          f'{(raw >= 5).sum():>4d} {(raw >= 10).sum():>4d} {(raw == 15).sum():>4d}')
    lbl = f'search K={K_per_depth}'
    print(f'{lbl:24s} {srch.mean():>6.2f} {int(np.median(srch)):>7d} {srch.max():>4d} '
          f'{(srch >= 5).sum():>4d} {(srch >= 10).sum():>4d} {(srch == 15).sum():>4d}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--N', type=int, default=50)
    p.add_argument('--K1', type=int, default=24)
    p.add_argument('--K2', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--embed_dim', type=int, default=96)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--ff_dim', type=int, default=192)
    args = p.parse_args()
    eval_phase6(args.ckpt, embed_dim=args.embed_dim, num_heads=args.num_heads,
                num_layers=args.num_layers, ff_dim=args.ff_dim,
                N=args.N, K_per_depth=(args.K1, args.K2), device=args.device)
