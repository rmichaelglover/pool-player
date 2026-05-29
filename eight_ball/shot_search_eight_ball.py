"""
Depth-1 shot search for the 8-ball EightBallNet architecture.

Ported from 14.1's shot_search_phase7.py. Same AlphaZero-style pattern —
"policy as prior, value as critic, search as planner": the network proposes
a distribution over (shot, force, spin); we EVALUATE the top candidates by
cloning the env, simulating each with real physics, and scoring the resulting
state with the value head. Pick the candidate with the highest

    Q = r + γ · V_me(s')

The two things that make 8-ball different from 14.1's single-player run:

  1. ADVERSARIAL VALUE. EightBallNet's value head is a *win probability from
     the current player's perspective*, and a shot can pass the turn to the
     opponent. After a cloned step we look at env_c.current_player:
       - turn stayed with me     → V_me(s') = V(s')
       - turn switched to opponent→ V_me(s') = 1 - V(s')   (V is now the
                                     opponent's win prob from their POV)
       - game over               → Q = r   (terminal reward is already ±1
                                     from my perspective: win +1, loss -1)
     This is what lets search value a good safety: it leaves the opponent a
     low win prob, so 1 - V(s') is high.

  2. SAFETY AS A CANDIDATE. The env treats action_idx == len(legal) as a
     safety (soft aim at the easiest legal shot). We always append it to the
     candidate set so search can *choose* to play safe — core 8-ball strategy.

Ball-in-hand placement is NOT searched here (it's a continuous 2-D action
whose value only resolves after a follow-up shot). The caller should keep
using the net's deterministic placement for awaiting_placement states.
Searching placement (sample N placements → nested shot search each) is a
natural follow-up; see TODO at the bottom.

Total sims per decision ≈ (K_shots × M_per_shot + 1) × n_mc, each ~0.1-0.3ms,
plus one batched value forward over all candidate next states.
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
from eight_ball_net import EightBallNet, EightBallObs, MAX_SHOTS
from eight_ball_env import EightBallEnv

_SEARCH_VERBOSE = False


def set_verbose(on: bool):
    global _SEARCH_VERBOSE
    _SEARCH_VERBOSE = bool(on)


def _obs_to_batch(obs: EightBallObs, device):
    return {
        'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(device),
        'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(device),
        'ball_group': torch.from_numpy(obs.ball_group).unsqueeze(0).to(device),
        'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(device),
        'game_state': torch.from_numpy(obs.game_state).unsqueeze(0).to(device),
        'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(device),
        'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(device),
    }


def _stack_obs(obs_list, device):
    """Stack a list of EightBallObs into a batched tensor dict for forward."""
    return {
        'balls': torch.from_numpy(np.stack([o.balls for o in obs_list])).to(device),
        'ball_mask': torch.from_numpy(np.stack([o.ball_mask for o in obs_list])).to(device),
        'ball_group': torch.from_numpy(np.stack([o.ball_group for o in obs_list])).to(device),
        'pockets': torch.from_numpy(np.stack([o.pockets for o in obs_list])).to(device),
        'game_state': torch.from_numpy(np.stack([o.game_state for o in obs_list])).to(device),
        'shots': torch.from_numpy(np.stack([o.shots for o in obs_list])).to(device),
        'shot_mask': torch.from_numpy(np.stack([o.shot_mask for o in obs_list])).to(device),
    }


def clone_eight_ball_env(env: EightBallEnv) -> EightBallEnv:
    """Cheap deep-enough copy of an EightBallEnv for simulating candidate
    shots without touching the live env. Copies every instance attribute that
    step()/get_obs()/_switch_player()/the helpers read or write."""
    new = EightBallEnv.__new__(EightBallEnv)
    # --- config ---
    new.max_shots_per_game = env.max_shots_per_game
    new.opening_break_force = env.opening_break_force
    new.aim_noise_deg = env.aim_noise_deg
    new.force_noise_pct = env.force_noise_pct
    new.spin_noise = env.spin_noise
    new.shape_reward_weight = env.shape_reward_weight
    # --- mutable game state ---
    new.cue = list(env.cue)
    new.balls = {bid: list(pos) for bid, pos in env.balls.items()}
    new.phase = env.phase
    new.current_player = env.current_player
    new.groups = dict(env.groups)
    new.ball_in_hand = env.ball_in_hand
    new.ball_in_hand_behind_head = env.ball_in_hand_behind_head
    new.awaiting_placement = env.awaiting_placement
    new.winner = env.winner
    new.total_shots = env.total_shots
    new.consecutive_fouls = list(env.consecutive_fouls)
    new.is_safety = env.is_safety
    return new


def shot_search_eight_ball(
    net: EightBallNet,
    env: EightBallEnv,
    obs: EightBallObs,
    K_shots: int = 8,
    M_per_shot: int = 4,
    gamma: float = 0.99,
    device: str = 'cpu',
    noise_samples: int = 1,
    include_safety: bool = True,
):
    """Depth-1 adversarial search. Returns (action_idx, force_raw, spin_raw)
    of the best candidate, or None if there are no legal shots / it's a
    placement decision (caller handles those).

    action_idx == len(obs.shot_meta) denotes the safety action (the env aims
    softly at the easiest legal shot and overrides force/spin internally).
    """
    if env.awaiting_placement or not obs.shot_meta:
        return None

    me = env.current_player
    n_legal = len(obs.shot_meta)

    env_noisy = (env.aim_noise_deg > 0 or env.force_noise_pct > 0 or
                 env.spin_noise > 0)
    n_mc = max(1, noise_samples) if env_noisy else 1

    # 1) Net forward on current state → per-shot scores + force/spin means.
    batch = _obs_to_batch(obs, device)
    with torch.no_grad():
        scores, f_means, s_means, _, _, _ = net.forward(**batch)
    score_arr = scores[0, :n_legal].cpu().numpy()
    K = min(K_shots, n_legal)
    top_k_idx = np.argsort(-score_arr)[:K]

    force_std = max(torch.exp(net.log_std[0]).item(), 0.3)
    spin_std = max(torch.exp(net.log_std[1]).item(), 0.3)

    # 2) Build (K × M) candidate (action_idx, f_raw, s_raw) tuples, plus a
    #    soft-force position variant per shot, plus the safety action.
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
        # A soft-force variant helps find position/safe-speed alternatives.
        if abs(f_mu) > 1.0:
            actions.append((int(shot_idx), 0.0, s_mu))
    if include_safety:
        # Env reads action_idx == n_legal as "safety"; it overrides f/s itself.
        actions.append((n_legal, -2.0, 0.0))

    # 3) Roll out each candidate n_mc times on cloned envs (clones inherit the
    #    live env's noise attrs → fresh independent execution noise per clone).
    rollouts = []
    for action_idx, f_raw, s_raw in actions:
        samples = []
        for _ in range(n_mc):
            env_c = clone_eight_ball_env(env)
            _, r, done, _ = env_c.step(int(action_idx), float(f_raw),
                                       float(s_raw), obs)
            # switched: did the turn pass to the opponent? (only meaningful
            # when not done; on done we score with r directly)
            switched = (env_c.current_player != me)
            samples.append((r, done, switched, None if done else env_c))
        rollouts.append(samples)

    # 4) Batched value forward over all non-terminal next states.
    flat = []  # (cand_idx, r, done, switched, env_c_or_None)
    for ci, samples in enumerate(rollouts):
        for r, done, switched, env_c in samples:
            flat.append((ci, r, done, switched, env_c))
    nonterm = [(i, f) for i, f in enumerate(flat) if f[4] is not None]
    if nonterm:
        next_obs_list = [f[4].get_obs() for _, f in nonterm]
        next_batch = _stack_obs(next_obs_list, device)
        with torch.no_grad():
            _, _, _, _, next_values, _ = net.forward(**next_batch)
        next_values = next_values.cpu().numpy()
    else:
        next_values = np.array([])

    # 5) Q per sample (with adversarial value flip), averaged per candidate.
    candidate_qs = [[] for _ in actions]
    nt_pos = 0
    for ci, r, done, switched, env_c in flat:
        if done:
            q = r  # terminal reward already in my perspective (+1 win / -1 loss)
        else:
            v = float(next_values[nt_pos])
            nt_pos += 1
            v_me = (1.0 - v) if switched else v
            q = r + gamma * v_me
        candidate_qs[ci].append(q)

    best_q = -float('inf')
    best_action = None
    for ci, qs in enumerate(candidate_qs):
        mean_q = float(np.mean(qs))
        if mean_q > best_q:
            best_q = mean_q
            best_action = actions[ci]

    if _SEARCH_VERBOSE:
        _print_search_breakdown(obs, actions, candidate_qs, best_action,
                                n_legal, K, M_per_shot, n_mc, gamma)

    return best_action


def _print_search_breakdown(obs, actions, candidate_qs, best_action,
                            n_legal, K, M, n_mc, gamma):
    from shot_enumerator import POCKET_NAMES as _PN
    # Best Q per action_idx (shots + safety).
    per_act = {}
    for ci, qs in enumerate(candidate_qs):
        a_idx = actions[ci][0]
        mean_q = float(np.mean(qs))
        if a_idx not in per_act or mean_q > per_act[a_idx]:
            per_act[a_idx] = mean_q
    rows = sorted(per_act.items(), key=lambda kv: -kv[1])
    best_idx = best_action[0] if best_action else -2
    print(f'[8ball-search] K={K} M={M} n_mc={n_mc} γ={gamma} n_legal={n_legal}')
    print(f'           {"sel":>3} {"target":>14} {"Q":>8}')
    for a_idx, q in rows:
        sel = '*' if a_idx == best_idx else ' '
        if a_idx >= n_legal:
            label = 'SAFETY'
        else:
            sh = obs.shot_meta[a_idx]
            label = f'ball {sh.ball_id}->{_PN[sh.pocket_idx]}'
        print(f'           {sel:>3} {label:>14} {q:>+8.3f}')
    sys.stdout.flush()


def shot_search_distill(
    net: EightBallNet,
    env: EightBallEnv,
    obs: EightBallObs,
    K_shots: int = 6,
    M_per_shot: int = 2,
    gamma: float = 0.99,
    device: str = 'cpu',
    noise_samples: int = 1,
    include_safety: bool = True,
):
    """Like shot_search_eight_ball but returns enriched info for search-
    augmented (distillation) training:

        best_action: (action_idx, force_raw, spin_raw) or None
        action_qs:   {action_idx: best_mean_Q} over evaluated candidates
                     (key == n_legal is the safety action) — soft policy target
        best_q:      mean Q of the best candidate — value target (already in
                     my perspective, ∈ roughly [-1, 1+])
    """
    if env.awaiting_placement or not obs.shot_meta:
        return None, {}, 0.0

    me = env.current_player
    n_legal = len(obs.shot_meta)
    env_noisy = (env.aim_noise_deg > 0 or env.force_noise_pct > 0 or
                 env.spin_noise > 0)
    n_mc = max(1, noise_samples) if env_noisy else 1

    batch = _obs_to_batch(obs, device)
    with torch.no_grad():
        scores, f_means, s_means, _, _, _ = net.forward(**batch)
    score_arr = scores[0, :n_legal].cpu().numpy()
    K = min(K_shots, n_legal)
    top_k_idx = np.argsort(-score_arr)[:K]

    force_std = max(torch.exp(net.log_std[0]).item(), 0.3)
    spin_std = max(torch.exp(net.log_std[1]).item(), 0.3)

    actions = []
    for shot_idx in top_k_idx:
        f_mu = float(f_means[0, shot_idx].item())
        s_mu = float(s_means[0, shot_idx].item())
        for j in range(M_per_shot):
            if j == 0:
                actions.append((int(shot_idx), f_mu, s_mu))
            else:
                actions.append((int(shot_idx),
                                f_mu + np.random.randn() * force_std,
                                s_mu + np.random.randn() * spin_std))
        if abs(f_mu) > 1.0:
            actions.append((int(shot_idx), 0.0, s_mu))
    if include_safety:
        actions.append((n_legal, -2.0, 0.0))

    rollouts = []
    for action_idx, f_raw, s_raw in actions:
        samples = []
        for _ in range(n_mc):
            env_c = clone_eight_ball_env(env)
            _, r, done, _ = env_c.step(int(action_idx), float(f_raw),
                                       float(s_raw), obs)
            switched = (env_c.current_player != me)
            samples.append((r, done, switched, None if done else env_c))
        rollouts.append(samples)

    flat = []
    for ci, samples in enumerate(rollouts):
        for r, done, switched, env_c in samples:
            flat.append((ci, r, done, switched, env_c))
    nonterm = [(i, f) for i, f in enumerate(flat) if f[4] is not None]
    if nonterm:
        next_obs_list = [f[4].get_obs() for _, f in nonterm]
        next_batch = _stack_obs(next_obs_list, device)
        with torch.no_grad():
            _, _, _, _, next_values, _ = net.forward(**next_batch)
        next_values = next_values.cpu().numpy()
    else:
        next_values = np.array([])

    candidate_qs = [[] for _ in actions]
    nt_pos = 0
    for ci, r, done, switched, env_c in flat:
        if done:
            q = r
        else:
            v = float(next_values[nt_pos])
            nt_pos += 1
            q = r + gamma * ((1.0 - v) if switched else v)
        candidate_qs[ci].append(q)

    action_qs = {}
    best_q = -float('inf')
    best_action = None
    for ci, qs in enumerate(candidate_qs):
        mean_q = float(np.mean(qs))
        a_idx, f_raw, s_raw = actions[ci]
        if a_idx not in action_qs or mean_q > action_qs[a_idx]:
            action_qs[a_idx] = mean_q
        if mean_q > best_q:
            best_q = mean_q
            best_action = (a_idx, f_raw, s_raw)
    return best_action, action_qs, best_q


def placement_search_distill(
    net: EightBallNet,
    env: EightBallEnv,
    obs: EightBallObs,
    n_place: int = 8,
    K_shots: int = 6,
    M_per_shot: int = 2,
    gamma: float = 0.99,
    device: str = 'cpu',
    noise_samples: int = 1,
):
    """Value-based ball-in-hand placement search — the heuristic-free analogue
    of shot_search_distill for awaiting_placement states.

    Sample n_place candidate cue positions from the net's OWN placement head
    (its learned prior, mean + Gaussian samples around it), apply each via
    step_placement on a clone, run depth-1 shot search on the resulting state,
    and score the placement by the follow-up best_q (already in my
    perspective). The candidate with the highest follow-up Q wins — the network
    decides what a good placement is from its own value, with no hand-coded
    placement heuristic, so novel placements can surface and be distilled.

    Returns (best_xy, place_q):
        best_xy:  (x_norm, y_norm) in [0,1]^2, or None if not a placement state
        place_q:  follow-up best_q of the chosen placement (value target,
                  roughly [-1, 1+]); -1.0 if a placement leaves no legal shot.
    """
    if not env.awaiting_placement:
        return None, 0.0

    batch = _obs_to_batch(obs, device)
    with torch.no_grad():
        *_, pooled = net.forward(**batch)
        mu = torch.sigmoid(net.placement_head(pooled))[0].cpu().numpy()
        std = torch.exp(net.placement_log_std).cpu().numpy()

    # Candidate placements: the deterministic mean + samples around it.
    cands = [(float(mu[0]), float(mu[1]))]
    for _ in range(max(0, n_place - 1)):
        x = float(np.clip(mu[0] + np.random.randn() * std[0], 0.0, 1.0))
        y = float(np.clip(mu[1] + np.random.randn() * std[1], 0.0, 1.0))
        cands.append((x, y))

    best_xy, best_score = None, -float('inf')
    for xn, yn in cands:
        env_c = clone_eight_ball_env(env)
        obs_c, _, _, _ = env_c.step_placement(xn, yn)
        if not obs_c.shot_meta:
            score = -1.0  # placed yourself with no legal shot — clearly bad
        else:
            _, _, fq = shot_search_distill(
                net, env_c, obs_c, K_shots=K_shots, M_per_shot=M_per_shot,
                gamma=gamma, device=device, noise_samples=noise_samples)
            score = fq
        if score > best_score:
            best_score, best_xy = score, (xn, yn)
    return best_xy, float(best_score)
