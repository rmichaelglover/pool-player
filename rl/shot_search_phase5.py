"""
Shot-level search for Phase 5 (3-shot break curriculum).

Depth-3 search with max over candidates at each shot. For terminal shot
(shot 3), we take max reward directly. For earlier shots, we recurse:

    Q(a_t) = r(a_t) + γ · max_{a_{t+1}} Q(a_{t+1})    if not terminal
           = r(a_t)                                    if terminal/episode-ends

This is expensive if we fully expand — O(K^3) simulations per decision.
We keep K modest (e.g. K1=24, K2=16, K3=16 → ~6K sims per decision ≈ 400 ms
with a 0.06ms physics sim). Cheap enough for eval.

For training-time search (AlphaZero loop), we'd reduce K or use a value net
to prune. Not doing that in this first pass.
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
from train_phase5 import (Phase5Env, OPEN_ID, BREAK_ID, RACK_IDS,
                           RACK_POSITIONS)


def clone_phase5_env(env: Phase5Env) -> Phase5Env:
    new_env = Phase5Env.__new__(Phase5Env)
    new_env.pocket_reward = env.pocket_reward
    new_env.max_shots = env.max_shots
    new_env.cue = list(env.cue)
    new_env.balls = {bid: list(pos) for bid, pos in env.balls.items()}
    new_env.shot_idx = env.shot_idx
    new_env.done = env.done
    new_env.last_shot_info = dict(env.last_shot_info) if hasattr(env, 'last_shot_info') else {}
    return new_env


def _policy_samples(net, obs_t, K, device, std_floor=0.1):
    """Sample K candidate actions (first = deterministic mean)."""
    with torch.no_grad():
        action_mean, _ = net(obs_t)
        std = torch.exp(net.log_std).clamp(min=std_floor)
        noise = Normal(torch.zeros_like(std), std).sample((K - 1,))
        samples = torch.cat([action_mean, action_mean + noise], dim=0)
    return samples


def _best_q(net, env, K_list, gamma, device, std_floor):
    """Recursive: evaluate env's state, pick best Q over K_list[0] candidates.
    Returns (best_action, best_q). At terminal depth, Q = r."""
    if env.done or not K_list:
        return None, 0.0

    K = K_list[0]
    rest = K_list[1:]
    obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
    candidates = _policy_samples(net, obs_t, K, device, std_floor)

    best_q = -float('inf')
    best_a = None
    for a in candidates:
        aim, force, spin = decode_action(a.cpu().numpy())
        env_c = clone_phase5_env(env)
        _, r, done, info = env_c.step(aim, force, spin)
        if done or not rest:
            q = r
        else:
            _, future = _best_q(net, env_c, rest, gamma, device, std_floor)
            q = r + gamma * future
        if q > best_q:
            best_q = q
            best_a = a
    return best_a, best_q


def shot_search_phase5(
    net, env: Phase5Env,
    K_per_shot=(24, 16, 16),  # (shot1, shot2, shot3)
    gamma=0.99, device='cpu', std_floor=0.1,
):
    """Pick best action for env's current shot via depth-(max_shots - shot_idx) search."""
    assert not env.done
    remaining = env.max_shots - env.shot_idx
    K_list = list(K_per_shot[:remaining])
    best_a, _ = _best_q(net, env, K_list, gamma, device, std_floor)
    return best_a.cpu().numpy()


def eval_phase5(ckpt_path: str, embed_dim=96, num_heads=6, num_layers=4,
                ff_dim=192, N=200, K_per_shot=(24, 16, 16), device='cpu'):
    """Deterministic eval: raw policy vs search."""
    device = torch.device(device)
    net = PoolAttentionNet(embed_dim=embed_dim, num_heads=num_heads,
                           num_layers=num_layers, ff_dim=ff_dim,
                           act_dim=ACT_DIM).to(device)
    net.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5).to(device))
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()

    def run_episode(env, use_search):
        total_r = 0.0
        shots_taken = 0
        while not env.done:
            if use_search:
                a = shot_search_phase5(net, env, K_per_shot=K_per_shot, device=device)
            else:
                obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
                with torch.no_grad():
                    act, _, _ = net.get_action(obs_t, deterministic=True)
                a = act[0].cpu().numpy()
            aim, force, spin = decode_action(a)
            _, r, _, info = env.step(aim, force, spin)
            total_r += r
            shots_taken += 1
        return total_r / env.pocket_reward, shots_taken  # pockets, shots

    raw_pockets = raw_shots = 0
    srch_pockets = srch_shots = 0
    raw_ran_rack = srch_ran_rack = 0  # episodes with ≥3 pockets

    import random
    t0 = time.time()
    for i in range(N):
        seed = i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase5Env()
        p, s = run_episode(env, use_search=False)
        raw_pockets += p; raw_shots += s
        if p >= 3: raw_ran_rack += 1

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase5Env()
        p, s = run_episode(env, use_search=True)
        srch_pockets += p; srch_shots += s
        if p >= 3: srch_ran_rack += 1

    elapsed = time.time() - t0
    print(f'Checkpoint: {ckpt_path}')
    print(f'Episodes: {N}  Time: {elapsed:.1f}s')
    print(f'{"Strategy":20s} {"Avg pockets/ep":>16s} {"Avg shots/ep":>14s} {"3-ball runs":>13s}')
    print(f'{"raw policy":20s} {raw_pockets/N:>16.2f} {raw_shots/N:>14.2f} {raw_ran_rack}/{N}')
    lbl = f'search K={K_per_shot}'
    print(f'{lbl:20s} {srch_pockets/N:>16.2f} {srch_shots/N:>14.2f} {srch_ran_rack}/{N}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--N', type=int, default=200)
    p.add_argument('--K1', type=int, default=24)
    p.add_argument('--K2', type=int, default=16)
    p.add_argument('--K3', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--embed_dim', type=int, default=96)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--ff_dim', type=int, default=192)
    args = p.parse_args()
    eval_phase5(args.ckpt, embed_dim=args.embed_dim, num_heads=args.num_heads,
                num_layers=args.num_layers, ff_dim=args.ff_dim,
                N=args.N, K_per_shot=(args.K1, args.K2, args.K3), device=args.device)
