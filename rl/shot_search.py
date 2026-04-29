"""
Shot-level search on top of a trained policy/value network.

Idea (AlphaZero-lite): at decision time, sample K candidate shots from the
policy, simulate each in pool_sim, and pick the one with the best estimated
return. Cheap because the physics sim is fast (~0.06 ms/shot).

For Phase 4 (2-ball, episode ends after shot 2), we can do full depth=2
search with max over shot-2 candidates — no value net needed for a terminal
evaluation because V(terminal) = 0 after shot 2 regardless of outcome:

    Q(a1) = r(a1) + γ · max_{a2} r(a2)

For longer-horizon phases, replace the inner max with r(a2) + γ · V(s'),
and recurse for deeper search.

Usage:
    action = shot_search_phase4(net, env, K1=32, K2=16, device='cuda')
    aim, force, spin = decode_action(action)
    env.step(aim, force, spin)
"""
from __future__ import annotations
import copy
import math
import os
import sys

import numpy as np
import torch
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from train_phase4 import Phase4Env, decode_action, ACT_DIM


def clone_phase4_env(env: Phase4Env) -> Phase4Env:
    """Shallow-deep copy of Phase4Env state. We stepped-simulate on the clone
    so the original env state stays untouched."""
    new_env = Phase4Env.__new__(Phase4Env)
    new_env.pocket_reward = env.pocket_reward
    new_env.position_bonus_weight = env.position_bonus_weight
    new_env.cue = list(env.cue)
    new_env.balls = {bid: list(pos) for bid, pos in env.balls.items()}
    new_env.shot_order = list(env.shot_order)
    new_env.shot_idx = env.shot_idx
    new_env.done = env.done
    return new_env


def _policy_samples(net, obs_t, K, device, std_floor=0.1):
    """Sample K candidate actions from the policy. Includes the deterministic
    mean as the first candidate so search never degrades to worse than the
    raw policy. std_floor prevents pathological collapse-era checkpoints from
    returning K identical actions."""
    with torch.no_grad():
        action_mean, _ = net(obs_t)             # (1, act_dim)
        std = torch.exp(net.log_std).clamp(min=std_floor)  # (act_dim,)
        # First sample = deterministic mean; remaining K-1 = stochastic.
        noise = Normal(torch.zeros_like(std), std).sample((K - 1,))  # (K-1, act_dim)
        det = action_mean                                             # (1, act_dim)
        stoch = action_mean + noise                                   # (K-1, act_dim)
        samples = torch.cat([det, stoch], dim=0)                      # (K, act_dim)
    return samples


def shot_search_phase4(
    net,
    env: Phase4Env,
    K1: int = 32,
    K2: int = 16,
    gamma: float = 0.99,
    device: str = 'cpu',
    std_floor: float = 0.1,
    return_diagnostics: bool = False,
):
    """
    Pick the best shot-1 action for the current env state via depth=2 search.

    Returns (action_np, info) where:
        action_np: (act_dim,) numpy array — raw network output to feed into decode_action
        info: dict with 'best_q', 'n_shot1_hit', etc. (only if return_diagnostics)
    """
    assert env.shot_idx == 0 and not env.done, "search designed for start-of-episode"
    obs = env.get_obs()
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

    shot1_candidates = _policy_samples(net, obs_t, K1, device, std_floor)  # (K1, act_dim)

    best_q = -float('inf')
    best_action = None
    n_shot1_hit = 0

    for a1 in shot1_candidates:
        aim, force, spin = decode_action(a1.cpu().numpy())
        env_c = clone_phase4_env(env)
        _, r1, done1, info1 = env_c.step(aim, force, spin)

        if done1:
            # Shot 1 missed/scratched → episode over; Q = r1 (typically 0)
            q = r1
        else:
            n_shot1_hit += 1
            # Shot 1 pocketed → evaluate shot 2 via K2 candidates, take best
            obs2_t = torch.FloatTensor(env_c.get_obs()).unsqueeze(0).to(device)
            shot2_candidates = _policy_samples(net, obs2_t, K2, device, std_floor)
            best_r2 = -float('inf')
            for a2 in shot2_candidates:
                aim2, force2, spin2 = decode_action(a2.cpu().numpy())
                env_c2 = clone_phase4_env(env_c)
                _, r2, _, _ = env_c2.step(aim2, force2, spin2)
                if r2 > best_r2:
                    best_r2 = r2
            q = r1 + gamma * best_r2

        if q > best_q:
            best_q = q
            best_action = a1

    result_action = best_action.cpu().numpy()
    if return_diagnostics:
        return result_action, {'best_q': float(best_q), 'n_shot1_hit': n_shot1_hit,
                               'K1': K1, 'K2': K2}
    return result_action


# ─── Evaluation ────────────────────────────────────────────────────────────

def eval_with_search(ckpt_path: str, embed_dim=160, num_heads=8, num_layers=4,
                     ff_dim=320, N=1500, K1=32, K2=16, device='cpu'):
    """Deterministic eval with shot search vs. raw policy (for comparison)."""
    device = torch.device(device)
    net = PoolAttentionNet(embed_dim=embed_dim, num_heads=num_heads,
                           num_layers=num_layers, ff_dim=ff_dim,
                           act_dim=ACT_DIM).to(device)
    import torch.nn as nn
    net.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5).to(device))
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()

    raw_s1 = raw_s2 = raw_g = raw_both = 0
    srch_s1 = srch_s2 = srch_g = srch_both = 0

    for i in range(N):
        # Same env seed for fair comparison (deterministic sampling).
        import random
        seed = i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # --- Raw policy ---
        env = Phase4Env()
        obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
        with torch.no_grad():
            act, _, _ = net.get_action(obs_t, deterministic=True)
        a = act[0].cpu().numpy()
        aim, force, spin = decode_action(a)
        _, _, done, info = env.step(aim, force, spin)
        if info.get('pocketed_target'): raw_s1 += 1
        if not done:
            raw_g += 1
            obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
            with torch.no_grad():
                act, _, _ = net.get_action(obs_t, deterministic=True)
            aim, force, spin = decode_action(act[0].cpu().numpy())
            _, _, _, info2 = env.step(aim, force, spin)
            if info2.get('pocketed_target'): raw_s2 += 1; raw_both += 1

        # --- Shot search (same seed → same env scenario) ---
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase4Env()
        a1 = shot_search_phase4(net, env, K1=K1, K2=K2, device=device)
        aim, force, spin = decode_action(a1)
        _, _, done, info = env.step(aim, force, spin)
        if info.get('pocketed_target'): srch_s1 += 1
        if not done:
            srch_g += 1
            # Shot 2: also use search but with K2 candidates at depth=1 (no
            # further lookahead since episode ends).
            obs_t = torch.FloatTensor(env.get_obs()).unsqueeze(0).to(device)
            cands = _policy_samples(net, obs_t, K2, device, std_floor=0.1)
            best_r = -float('inf'); best_a = None
            for a in cands:
                aim, force, spin = decode_action(a.cpu().numpy())
                env_c = clone_phase4_env(env)
                _, r, _, _ = env_c.step(aim, force, spin)
                if r > best_r: best_r = r; best_a = a
            aim, force, spin = decode_action(best_a.cpu().numpy())
            _, _, _, info2 = env.step(aim, force, spin)
            if info2.get('pocketed_target'): srch_s2 += 1; srch_both += 1

    print(f'Checkpoint: {ckpt_path}')
    print(f'{"Strategy":20s} {"Shot1":>7s} {"Shot2|got":>10s} {"Perfect":>8s}')
    print(f'{"raw policy":20s} {raw_s1/N:>7.1%} {raw_s2/max(1,raw_g):>10.1%} {raw_both/N:>8.1%}')
    search_label = f'search K1={K1},K2={K2}'
    print(f'{search_label:20s} {srch_s1/N:>7.1%} {srch_s2/max(1,srch_g):>10.1%} {srch_both/N:>8.1%}')
    print(f'Δ perfect: {(srch_both-raw_both)/N*100:+.1f} pp')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--N', type=int, default=500)
    p.add_argument('--K1', type=int, default=32)
    p.add_argument('--K2', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--embed_dim', type=int, default=160)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--ff_dim', type=int, default=320)
    args = p.parse_args()
    eval_with_search(args.ckpt, embed_dim=args.embed_dim, num_heads=args.num_heads,
                     num_layers=args.num_layers, ff_dim=args.ff_dim,
                     N=args.N, K1=args.K1, K2=args.K2, device=args.device)
