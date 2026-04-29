"""
Depth-1 shot search for the Phase 7/8 PoolGameNet architecture.

Pattern: at decision time, the network proposes a distribution over (shot,
force, spin). We then EVALUATE the top candidates by simulating each and
scoring the resulting state with the value function. Pick the candidate
with highest Q = r + γ · V(s').

This is the AlphaZero "policy as prior, value as critic, search as planner"
pattern — much higher quality than the raw deterministic policy because
search uses the actual physics to verify good outcomes rather than relying
on the network to commit to one action via training.

Key knobs:
  - K_shots: how many top-scored legal shots to consider (4-12)
  - M_per_shot: how many (force, spin) samples per shot (3-8)
  - gamma: discount on future value (0.99 default)

Total simulations per decision = K_shots × M_per_shot. Each sim ~0.1ms,
plus one batched value-net forward over all candidate next states.
Typical: K=8, M=4 → 32 sims + 1 batched forward, ≈30-50 ms per decision.
"""
from __future__ import annotations
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_game_net import (PoolGameNet, MAX_BALLS, MAX_POCKETS, MAX_SHOTS,
                            decode_force, decode_spin)
from train_phase7 import Phase7Env


def _obs_to_batch(obs, device):
    return {
        'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(device),
        'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(device),
        'ball_is_cue': torch.from_numpy(obs.ball_is_cue).unsqueeze(0).to(device),
        'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(device),
        'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(device),
        'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(device),
    }


def _stack_obs(obs_list, device):
    """Stack a list of Phase7Obs into a batched tensor dict for net.forward."""
    return {
        'balls': torch.from_numpy(np.stack([o.balls for o in obs_list])).to(device),
        'ball_mask': torch.from_numpy(np.stack([o.ball_mask for o in obs_list])).to(device),
        'ball_is_cue': torch.from_numpy(np.stack([o.ball_is_cue for o in obs_list])).to(device),
        'pockets': torch.from_numpy(np.stack([o.pockets for o in obs_list])).to(device),
        'shots': torch.from_numpy(np.stack([o.shots for o in obs_list])).to(device),
        'shot_mask': torch.from_numpy(np.stack([o.shot_mask for o in obs_list])).to(device),
    }


def clone_phase7_env(env: Phase7Env) -> Phase7Env:
    """Cheap state copy of a Phase7Env. Used to simulate candidate shots
    without affecting the live env state."""
    new_env = Phase7Env.__new__(Phase7Env)
    new_env.pocket_reward = env.pocket_reward
    new_env.max_shots = env.max_shots
    new_env.opening_break_force = env.opening_break_force
    new_env.scratch_penalty = env.scratch_penalty
    # Noise attributes (added later — clone must copy them or env.step crashes).
    new_env.aim_noise_deg = getattr(env, 'aim_noise_deg', 0.0)
    new_env.force_noise_pct = getattr(env, 'force_noise_pct', 0.0)
    new_env.spin_noise = getattr(env, 'spin_noise', 0.0)
    new_env.cue = list(env.cue)
    new_env.balls = {bid: list(pos) for bid, pos in env.balls.items()}
    new_env.shot_idx = env.shot_idx
    new_env.done = env.done
    new_env.rerack_count = env.rerack_count
    new_env.total_pocketed = env.total_pocketed
    new_env.pending_rerack = getattr(env, 'pending_rerack', False)
    new_env._last_rerack_count = getattr(env, '_last_rerack_count', env.rerack_count)
    return new_env


def shot_search_phase7(
    net: PoolGameNet,
    env: Phase7Env,
    obs,                    # Phase7Obs from env.get_obs()
    K_shots: int = 8,
    M_per_shot: int = 4,
    gamma: float = 0.99,
    device: str = 'cpu',
    noise_samples: int = 1,
):
    """Depth-1 search. Returns (shot_idx, force_raw, spin_raw) of the best
    candidate, or None if there are no legal shots.

    Monte-Carlo over execution noise: when the env has noise > 0 and
    noise_samples > 1, each candidate (shot, force, spin) is rolled out
    `noise_samples` times via cloned envs. Each clone inherits the live
    env's noise attributes, so its step() applies fresh independent noise.
    The candidate's score is the mean Q across these MC rollouts — i.e.
    expected Q under noise — so search prefers shots that succeed *robustly*
    rather than shots whose deterministic Q happens to be high.
    With deterministic envs (noise == 0) we collapse back to one rollout
    per candidate."""
    if not obs.shot_meta:
        return None

    env_noisy = (getattr(env, 'aim_noise_deg', 0.0) > 0 or
                 getattr(env, 'force_noise_pct', 0.0) > 0 or
                 getattr(env, 'spin_noise', 0.0) > 0)
    n_mc = max(1, noise_samples) if env_noisy else 1

    # 1) Network forward on current state → scores, force/spin means, value
    batch = _obs_to_batch(obs, device)
    with torch.no_grad():
        scores, f_means, s_means, _ = net.forward(**batch)
    n_legal = len(obs.shot_meta)
    score_arr = scores[0, :n_legal].cpu().numpy()
    K = min(K_shots, n_legal)
    top_k_idx = np.argsort(-score_arr)[:K]

    force_std = max(torch.exp(net.log_std[0]).item(), 0.3)
    spin_std = max(torch.exp(net.log_std[1]).item(), 0.3)

    # 2) Generate (K × M) candidate (shot_idx, f_raw, s_raw) tuples
    actions = []
    for shot_idx in top_k_idx:
        f_mu = float(f_means[0, shot_idx].item())
        s_mu = float(s_means[0, shot_idx].item())
        for j in range(M_per_shot):
            if j == 0:
                actions.append((int(shot_idx), f_mu, s_mu))
            else:
                f_raw = f_mu + np.random.randn() * force_std
                s_raw = s_mu + np.random.randn() * spin_std
                actions.append((int(shot_idx), f_raw, s_raw))

    # 3) Roll out each candidate n_mc times. Env clones inherit the live
    # env's noise attrs, so each clone.step() applies fresh independent
    # execution noise via the env's own machinery.
    rollouts = []
    for shot_idx, f_raw, s_raw in actions:
        samples = []
        for _ in range(n_mc):
            env_c = clone_phase7_env(env)
            _, r, done, _ = env_c.step(int(shot_idx), float(f_raw),
                                        float(s_raw), obs)
            samples.append((r, env_c, done))
        rollouts.append(samples)

    # 4) Batched value-net forward on all non-terminal next states (across
    # all rollouts of all candidates), then compute average Q per candidate.
    flat = []   # (cand_idx, sample_idx, reward, env_c_or_None_if_done)
    for ci, samples in enumerate(rollouts):
        for si, (r, env_c, done) in enumerate(samples):
            flat.append((ci, si, r, None if done else env_c))
    nonterm_obs_list = [env_c.get_obs() for _, _, _, env_c in flat if env_c is not None]
    if nonterm_obs_list:
        next_batch = _stack_obs(nonterm_obs_list, device)
        with torch.no_grad():
            _, _, _, next_values = net.forward(**next_batch)
        next_values = next_values.cpu().numpy()
    else:
        next_values = np.array([])

    candidate_qs = [[] for _ in actions]
    nt_pos = 0
    for ci, si, r, env_c in flat:
        if env_c is None:
            q = r
        else:
            q = r + gamma * next_values[nt_pos]
            nt_pos += 1
        candidate_qs[ci].append(q)

    best_q = -float('inf')
    best_action = None
    for ci, qs in enumerate(candidate_qs):
        mean_q = float(np.mean(qs))
        if mean_q > best_q:
            best_q = mean_q
            best_action = actions[ci]
    return best_action


# ── Eval utility ──────────────────────────────────────────────────────────

def eval_search_vs_raw(ckpt_path, N=100, K_shots=8, M_per_shot=4,
                        embed_dim=128, num_heads=8, num_layers=4,
                        device='cpu', max_shots=60):
    """Compare deterministic raw policy vs depth-1 search on the same seeds."""
    import random
    device = torch.device(device)
    net = PoolGameNet(embed_dim=embed_dim, num_heads=num_heads,
                      num_layers=num_layers).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()

    raw_runs = []
    srch_runs = []
    t0 = time.time()
    for seed in range(N):
        # Raw policy (deterministic)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase7Env(max_shots=max_shots)
        while not env.done:
            obs = env.get_obs()
            batch = _obs_to_batch(obs, device)
            with torch.no_grad():
                idx, f, s, _, _ = net.get_action(batch, deterministic=True)
            env.step(int(idx.item()), float(f.item()), float(s.item()), obs)
        raw_runs.append(env.total_pocketed)

        # Search
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        env = Phase7Env(max_shots=max_shots)
        while not env.done:
            obs = env.get_obs()
            action = shot_search_phase7(net, env, obs,
                                         K_shots=K_shots, M_per_shot=M_per_shot,
                                         device=device)
            if action is None:
                env.done = True
                break
            shot_idx, f_raw, s_raw = action
            env.step(shot_idx, f_raw, s_raw, obs)
        srch_runs.append(env.total_pocketed)

    elapsed = time.time() - t0
    raw = np.array(raw_runs); srch = np.array(srch_runs)
    print(f'Checkpoint: {ckpt_path}')
    print(f'N={N} episodes, max_shots={max_shots}, K={K_shots} M={M_per_shot}, elapsed {elapsed:.1f}s')
    print(f'{"Strategy":24s} {"Mean":>6s} {"Med":>4s} {"Max":>4s} {"≥14":>5s} {"≥28":>5s} {"≥42":>5s}')
    print(f'{"raw policy":24s} {raw.mean():>6.2f} {int(np.median(raw)):>4d} '
          f'{int(raw.max()):>4d} {(raw>=14).sum():>5d} {(raw>=28).sum():>5d} {(raw>=42).sum():>5d}')
    lbl = f'search K={K_shots},M={M_per_shot}'
    print(f'{lbl:24s} {srch.mean():>6.2f} {int(np.median(srch)):>4d} '
          f'{int(srch.max()):>4d} {(srch>=14).sum():>5d} {(srch>=28).sum():>5d} {(srch>=42).sum():>5d}')
    print(f'Δ mean: {srch.mean()-raw.mean():+.2f} balls')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--N', type=int, default=100)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--M', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max_shots', type=int, default=60)
    args = p.parse_args()
    eval_search_vs_raw(args.ckpt, N=args.N, K_shots=args.K, M_per_shot=args.M,
                        device=args.device, max_shots=args.max_shots)
