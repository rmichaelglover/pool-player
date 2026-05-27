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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
from pool_game_net import (PoolGameNet, MAX_BALLS, MAX_POCKETS, MAX_SHOTS,
                            decode_force, decode_spin)
from train_phase7 import Phase7Env
from rack_geometry import RACK_APEX


# ── Break-ball suppression heuristic (Tier 1) ──────────────────────────────
# Identifies "candidate break balls" geometrically (close enough to the rack
# apex to be useful as a break ball, but not inside the rack interior). When
# applied during search, suppresses these balls' shot probabilities so the
# agent saves them for end-of-rack instead of pocketing them mid-rack.
# Suppression scales with remaining ball count — heavy when many balls left,
# none with few balls left.

# Diagnostic toggle: when True, shot_search_phase7 prints a per-candidate
# breakdown (ball → pocket | net_prob | imm_r | next_V | Q) after each call.
# Off by default; flip on via set_verbose(True) from the demo server when
# investigating why search disagrees with the network's argmax.
_SEARCH_VERBOSE = False


def set_verbose(on: bool):
    global _SEARCH_VERBOSE
    _SEARCH_VERBOSE = bool(on)


_RACK_INTERIOR_X_MIN = 73.5
_RACK_INTERIOR_X_MAX = 84.5
_RACK_INTERIOR_Y_MIN = 21.0
_RACK_INTERIOR_Y_MAX = 29.0
_BREAK_BALL_MIN_DIST = 4.0     # too close to apex = inside or touching rack
_BREAK_BALL_MAX_DIST = 14.0    # too far = not useful as a break ball
_BREAK_BALL_DIVISOR = 10.0     # suppression mul = n_candidates / divisor

# ── Break-shot force/spin exploration ─────────────────────────────────────
# When env._post_rerack_break_pending is True, the search injects these
# high-force candidates (with both follow and draw spin) for each top-K shot.
# The simulation evaluates them honestly — if high force scatters the rack
# for +17 reward, it wins; otherwise the network's normal candidates prevail.
# The pocketing window is narrow at high cut angles, so we sample densely.
# Force raws: 0.5→147, 1.0→196, 1.25→205, 1.5→214, 1.75→220, 2.0→226, 2.5→235, 3.0→241
_BREAK_FORCE_RAWS = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
# Spin raws: follow (+1.0→+1.14), center (0), draw (-1.0→-1.14)
_BREAK_SPIN_RAWS = [1.0, 0.0, -1.0]


def is_candidate_break_ball(ball_pos):
    """True if a ball at this position is in the geometric region where it
    would make a good break-ball candidate: close enough to the rack apex
    to be reachable by a break shot, but not inside the future rack area."""
    bx, by = ball_pos
    if (_RACK_INTERIOR_X_MIN <= bx <= _RACK_INTERIOR_X_MAX
            and _RACK_INTERIOR_Y_MIN <= by <= _RACK_INTERIOR_Y_MAX):
        return False
    d = math.hypot(bx - RACK_APEX[0], by - RACK_APEX[1])
    return _BREAK_BALL_MIN_DIST < d <= _BREAK_BALL_MAX_DIST


def apply_break_ball_suppression(scores, shot_meta, balls):
    """Return modified scores (numpy array) with candidate break ball shots
    suppressed by adding log(mul) to their logits. Equivalent to multiplying
    their post-softmax probability by `mul` and renormalizing.

    Suppression strength scales by *number of candidate break balls* on the
    table, not by total ball count. Intuition: with only one candidate we
    must protect it strongly (mul=0.1 if 1 candidate); with many candidates
    we can spare one (mul=1.0 if ≥10). Matches what a real player does —
    "ball A is a great break ball but we have B and C as backup, so we can
    afford to use A here."

    mul = min(1.0, n_candidates / _BREAK_BALL_DIVISOR)
    """
    # Count and identify candidate break balls.
    candidate_ids = set()
    for bid, pos in balls.items():
        if is_candidate_break_ball(pos):
            candidate_ids.add(bid)
    n_candidates = len(candidate_ids)
    if n_candidates == 0:
        return scores
    mul = min(1.0, n_candidates / _BREAK_BALL_DIVISOR)
    if mul >= 1.0:
        return scores
    log_factor = math.log(mul)
    new_scores = scores.copy()
    for i, s in enumerate(shot_meta):
        if s.ball_id in candidate_ids:
            new_scores[i] += log_factor
    return new_scores


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
    new_env.shape_bonus_max = getattr(env, 'shape_bonus_max', 0.0)
    new_env.movement_penalty_weight = getattr(env, 'movement_penalty_weight', 1.0)
    new_env.cue_movement_penalty_weight = getattr(env, 'cue_movement_penalty_weight', 0.0)
    new_env.cue_ricochet_penalty_weight = getattr(env, 'cue_ricochet_penalty_weight', 0.0)
    new_env.next_shape_bonus_max = getattr(env, 'next_shape_bonus_max', 0.0)
    new_env.force_efficiency_penalty_weight = getattr(env, 'force_efficiency_penalty_weight', 0.0)
    new_env.rail_shot_bonus_weight = getattr(env, 'rail_shot_bonus_weight', 0.0)
    # End-of-rack reward attributes
    new_env.eor_bonus_max = getattr(env, 'eor_bonus_max', 0.0)
    new_env._post_rerack_break_pending = getattr(env, '_post_rerack_break_pending', False)
    new_env._break_ball_id_after_rerack = getattr(env, '_break_ball_id_after_rerack', None)
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
    prob_threshold: float = 0.0,
    break_ball_suppression: bool = False,
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
    # Optional break-ball suppression: lower the logit of shots whose target
    # ball is a candidate break ball, scaled by remaining ball count. Applied
    # BEFORE top-K so suppressed shots fall out of the search candidate set.
    candidate_ball_ids = set()
    if break_ball_suppression:
        # Identify candidates so we can hard-exclude them from search if
        # any non-candidate alternatives are available (Tier-1B logic).
        candidate_ball_ids = {bid for bid, pos in env.balls.items()
                               if is_candidate_break_ball(pos)}
        score_arr = apply_break_ball_suppression(
            score_arr, obs.shot_meta, env.balls)
    K = min(K_shots, n_legal)
    top_k_idx = np.argsort(-score_arr)[:K]

    # Optional probability-threshold filter: discard shots whose softmax
    # probability is below `prob_threshold`. Prevents search from overriding
    # the network on shots the network has already strongly rejected — turns
    # search into a tiebreaker rather than an overrider. Always keep at least
    # the argmax so we never end up with zero candidates.
    #
    # When break-ball suppression is active and at least one candidate exists,
    # apply a two-stage filter: prefer non-candidate shots that pass the
    # threshold; only fall back to candidates if no non-candidate qualifies.
    # This implements "skip break balls unless no other choice."
    if prob_threshold > 0.0:
        # Numerically-stable softmax over legal shots only.
        s = score_arr - score_arr.max()
        ps = np.exp(s); ps = ps / ps.sum()
        if break_ball_suppression and candidate_ball_ids:
            # Stage 1: scan ALL legal shots (not just top-K) for non-candidate
            # threshold-passers. Without this, a non-candidate with a decent
            # post-suppression prob but a lower raw logit can fall below the
            # top-K cut and be invisible to search.
            non_cand_all = [i for i in range(n_legal)
                            if obs.shot_meta[i].ball_id not in candidate_ball_ids
                            and ps[i] >= prob_threshold]
            if non_cand_all:
                # Limit to K best non-candidates by score (post-suppression).
                non_cand_all.sort(key=lambda i: -score_arr[i])
                kept = non_cand_all[:K]
            else:
                # Stage 2 (fallback): no non-candidate qualifies; allow
                # candidates back so search still has something to pick.
                kept = [i for i in top_k_idx if ps[i] >= prob_threshold]
                if not kept:
                    kept = [int(top_k_idx[0])]
        else:
            kept = [i for i in top_k_idx if ps[i] >= prob_threshold]
            if not kept:
                kept = [int(top_k_idx[0])]
        top_k_idx = np.asarray(kept, dtype=np.int64)

    force_std = max(torch.exp(net.log_std[0]).item(), 0.3)
    spin_std = max(torch.exp(net.log_std[1]).item(), 0.3)

    # 2) Generate (K × M) candidate (shot_idx, f_raw, s_raw) tuples
    actions = []
    is_break_shot = getattr(env, '_post_rerack_break_pending', False)
    if is_break_shot:
        print(f'[search] BREAK SHOT: injecting {len(top_k_idx)*len(_BREAK_FORCE_RAWS)*len(_BREAK_SPIN_RAWS)} high-force candidates', flush=True)
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
        if abs(f_mu) > 1.0:
            actions.append((int(shot_idx), 0.0, s_mu))
        if is_break_shot:
            for f_raw_hi in _BREAK_FORCE_RAWS:
                for s_raw_hi in _BREAK_SPIN_RAWS:
                    actions.append((int(shot_idx), f_raw_hi, s_raw_hi))

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

    if _SEARCH_VERBOSE:
        from shot_enumerator import POCKET_NAMES as _PN
        # Softmax of the (possibly suppressed) scores over all legal shots —
        # matches what the demo HUD shows as the per-shot probability.
        sc = score_arr - score_arr.max()
        ps = np.exp(sc); ps = ps / ps.sum()
        # Per-candidate imm reward + next-V means (re-iterate rollouts in the
        # same order used to fill next_values so nt_pos lines up).
        cand_imm_r = []
        cand_next_v = []
        nt_pos = 0
        for samples in rollouts:
            rs, vs = [], []
            for r, env_c, done in samples:
                rs.append(r)
                if env_c is None:
                    vs.append(0.0)
                else:
                    vs.append(float(next_values[nt_pos]))
                    nt_pos += 1
            cand_imm_r.append(float(np.mean(rs)))
            cand_next_v.append(float(np.mean(vs)))
        # Aggregate per shot_idx: keep the best-Q variant for each shot.
        per_shot = {}
        for ci, qs in enumerate(candidate_qs):
            sidx = actions[ci][0]
            mean_q = float(np.mean(qs))
            if sidx not in per_shot or mean_q > per_shot[sidx]['q']:
                per_shot[sidx] = {'q': mean_q,
                                  'imm_r': cand_imm_r[ci],
                                  'next_v': cand_next_v[ci]}
        rows = []
        for sidx, info in per_shot.items():
            sh = obs.shot_meta[sidx]
            rows.append((info['q'], sidx, sh.ball_id, sh.pocket_idx,
                         float(ps[sidx]), info['imm_r'], info['next_v']))
        rows.sort(key=lambda r: -r[0])
        best_sidx = best_action[0] if best_action is not None else -1
        print(f'[search] K={K} M={M_per_shot} n_mc={n_mc} γ={gamma} '
              f'n_legal={n_legal}')
        print(f'         {"sel":>3} {"ball":>4} {"pkt":>3} '
              f'{"net_p":>6} {"imm_r":>7} {"next_V":>7} {"Q":>8}')
        for q, sidx, ball_id, pkt_idx, netp, imm_r, nv in rows:
            sel = '*' if sidx == best_sidx else ' '
            print(f'         {sel:>3} {ball_id:>4} {_PN[pkt_idx]:>3} '
                  f'{netp:>6.3f} {imm_r:>+7.3f} {nv:>+7.3f} {q:>+8.3f}')
        sys.stdout.flush()

    if is_break_shot and best_action is not None:
        _, bf, bs = best_action
        from pool_game_net import FORCE_LO, FORCE_HI
        decoded_f = FORCE_LO + (FORCE_HI - FORCE_LO) / (1.0 + math.exp(-bf))
        decoded_s = 1.5 * math.tanh(bs)
        print(f'[search] BREAK result: force_raw={bf:.2f} → {decoded_f:.0f}, '
              f'spin_raw={bs:.2f} → {decoded_s:+.2f} '
              f'({"follow" if decoded_s > 0.3 else "draw" if decoded_s < -0.3 else "center"})',
              flush=True)

    return best_action


def shot_search_distill(
    net: PoolGameNet,
    env: Phase7Env,
    obs,
    K_shots: int = 4,
    M_per_shot: int = 1,
    gamma: float = 0.99,
    device: str = 'cpu',
    noise_samples: int = 1,
):
    """Like shot_search_phase7 but returns enriched info needed for
    distillation training.

    Returns:
        best_action: (shot_idx, force_raw, spin_raw) — chosen action; None if
                      no legal shots
        shot_qs: dict {shot_idx: best_mean_q_for_that_shot} over the K
                  candidate shot indices that were evaluated. Used to build
                  a soft policy target for cross-entropy distillation.
        best_q: float — mean Q of the best candidate; used as value target.
    """
    if not obs.shot_meta:
        return None, {}, 0.0

    env_noisy = (getattr(env, 'aim_noise_deg', 0.0) > 0 or
                 getattr(env, 'force_noise_pct', 0.0) > 0 or
                 getattr(env, 'spin_noise', 0.0) > 0)
    n_mc = max(1, noise_samples) if env_noisy else 1

    batch = _obs_to_batch(obs, device)
    with torch.no_grad():
        scores, f_means, s_means, _ = net.forward(**batch)
    n_legal = len(obs.shot_meta)
    score_arr = scores[0, :n_legal].cpu().numpy()
    K = min(K_shots, n_legal)
    top_k_idx = np.argsort(-score_arr)[:K]

    force_std = max(torch.exp(net.log_std[0]).item(), 0.3)
    spin_std = max(torch.exp(net.log_std[1]).item(), 0.3)

    is_break_shot = getattr(env, '_post_rerack_break_pending', False)
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
        if abs(f_mu) > 1.0:
            actions.append((int(shot_idx), 0.0, s_mu))
        if is_break_shot:
            for f_raw_hi in _BREAK_FORCE_RAWS:
                for s_raw_hi in _BREAK_SPIN_RAWS:
                    actions.append((int(shot_idx), f_raw_hi, s_raw_hi))

    rollouts = []
    for shot_idx, f_raw, s_raw in actions:
        samples = []
        for _ in range(n_mc):
            env_c = clone_phase7_env(env)
            _, r, done, _ = env_c.step(int(shot_idx), float(f_raw),
                                        float(s_raw), obs)
            samples.append((r, env_c, done))
        rollouts.append(samples)

    flat = []
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

    # Aggregate per shot_idx (best mean Q across M variants); track best overall.
    shot_qs = {}
    best_q = -float('inf')
    best_action = None
    for ci, qs in enumerate(candidate_qs):
        mean_q = float(np.mean(qs))
        shot_idx, f_raw, s_raw = actions[ci]
        if shot_idx not in shot_qs or mean_q > shot_qs[shot_idx]:
            shot_qs[shot_idx] = mean_q
        if mean_q > best_q:
            best_q = mean_q
            best_action = (shot_idx, f_raw, s_raw)
    return best_action, shot_qs, best_q


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
