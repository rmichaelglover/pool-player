"""
Search-augmented (distillation) training for 8-ball.

Ported from 14.1's train_phase7_distill. At each rollout step we run depth-1
search (shot_search_distill) and use its results as SUPERVISED targets — no
PPO importance ratio. 14.1 found PPO+search blows up; distillation is stable
and bakes search's strength permanently into the policy.

    Loss = ce_weight   · CE(policy logits, softmax(search Q / T))
         + mse_force_w  · MSE(force_mean[chosen], search force)   (shots only)
         + mse_spin_w   · MSE(spin_mean[chosen],  search spin)    (shots only)
         + value_weight · BCE(value, clamp(best_q, 0, 1))
         − entropy_w    · entropy(policy logits)

8-ball-specific vs 14.1:
  * Policy distribution is [shot_scores (MAX_SHOTS), safety_logit] — the
    search safety action (key == n_legal) maps onto target index MAX_SHOTS.
  * Value head is sigmoid win-prob, so the value target is clamp(best_q,0,1)
    with BCE (matching how PPO trained it), not raw MSE on run length.
  * Ball-in-hand placement steps are advanced by the net's deterministic
    placement and carry NO training target (placement search is future work).
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
from eight_ball_net import EightBallNet, MAX_SHOTS
from eight_ball_env import EightBallEnv, GAME_OVER
from shot_search_eight_ball import shot_search_distill, placement_search_distill

_OBS_KEYS = ('balls', 'ball_mask', 'ball_group', 'pockets',
             'game_state', 'shots', 'shot_mask')


def _obs_to_batch(obs, device):
    return {k: torch.from_numpy(getattr(obs, k)).unsqueeze(0).to(device)
            for k in _OBS_KEYS}


def _load_warm(net, warm_start, device):
    state = torch.load(warm_start, map_location=device, weights_only=True)
    key = 'shot_encoder.0.weight'
    if key in state and state[key].shape[1] < net.shot_encoder[0].in_features:
        pad = net.shot_encoder[0].in_features - state[key].shape[1]
        state[key] = torch.cat(
            [state[key], torch.zeros(state[key].shape[0], pad,
                                     device=state[key].device)], dim=1)
    net.load_state_dict(state, strict=False)


def train_distill(num_envs=6, device_name='cpu', max_iters=200,
                  tag='8ball_v6_distill', lr=1e-4, steps_per_iter=24,
                  search_k=4, search_m=1, search_mc=1, gamma=0.99,
                  softmax_temp=1.0, ce_weight=1.0, mse_force_weight=0.1,
                  mse_spin_weight=0.1, value_weight=0.5, entropy_weight=0.005,
                  log_std_min=-3.0, warm_start=None, env_kwargs=None,
                  epochs_per_iter=2, batch_size=256,
                  place_n=8, place_weight=1.0):
    device = torch.device(device_name)
    net = EightBallNet().to(device)
    if warm_start and os.path.exists(warm_start):
        _load_warm(net, warm_start, device)
        print(f'Warm-started from {warm_start}', flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'8-ball distillation. EightBallNet {n_params:,} params on {device}',
          flush=True)
    print(f'config: tag={tag} lr={lr} steps={steps_per_iter} envs={num_envs} '
          f'K={search_k} M={search_m} MC={search_mc} gamma={gamma} '
          f'(ce={ce_weight} mse_f={mse_force_weight} mse_s={mse_spin_weight} '
          f'v={value_weight} ent={entropy_weight})', flush=True)

    envs = [EightBallEnv(**(env_kwargs or {})) for _ in range(num_envs)]
    last_obs = [e.reset() for e in envs]

    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    recent_lengths = deque(maxlen=500)
    recent_safety = deque(maxlen=2000)
    best_metric = -1.0

    for iteration in range(max_iters):
        # ── Rollout: search at each step, record supervised targets ──────
        roll_obs = {k: [] for k in _OBS_KEYS}
        roll_gather = []      # shot index to gather force/spin (0 for safety)
        roll_force = []
        roll_spin = []
        roll_tgt_dist = []    # (MAX_SHOTS+1,) soft policy target
        roll_tgt_value = []   # scalar in [0,1]
        roll_force_valid = [] # 1.0 for real shots, 0.0 for safety
        # Placement (ball-in-hand) distillation targets — value-searched, not
        # heuristic. Parallel buffer keyed by the same obs schema.
        roll_place_obs = {k: [] for k in _OBS_KEYS}
        roll_place_x = []
        roll_place_y = []
        roll_place_value = []
        iter_decisions = 0
        iter_placements = 0

        for _step in range(steps_per_iter):
            for i, env in enumerate(envs):
                obs = last_obs[i]

                # Placement: value-search candidate cue positions and distill
                # toward the best (heuristic-free — see placement_search_distill).
                if env.awaiting_placement:
                    best_xy, place_q = placement_search_distill(
                        net, env, obs, n_place=place_n, K_shots=search_k,
                        M_per_shot=search_m, noise_samples=search_mc,
                        gamma=gamma, device=device)
                    if best_xy is None:
                        with torch.no_grad():
                            _, xn, yn, _, _ = net.get_action(
                                _obs_to_batch(obs, device), deterministic=True)
                        best_xy = (xn.item(), yn.item())
                    else:
                        for k in _OBS_KEYS:
                            roll_place_obs[k].append(getattr(obs, k))
                        roll_place_x.append(best_xy[0])
                        roll_place_y.append(best_xy[1])
                        roll_place_value.append(
                            float(np.clip(place_q, 0.0, 1.0)))
                        iter_placements += 1
                    nobs, _, done, _ = env.step_placement(best_xy[0], best_xy[1])
                    last_obs[i] = env.reset() if done else nobs
                    continue

                ba, action_qs, best_q = shot_search_distill(
                    net, env, obs, K_shots=search_k, M_per_shot=search_m,
                    noise_samples=search_mc, gamma=gamma, device=device)

                if ba is None or not action_qs:
                    # No legal shots — forced step, no target.
                    nobs, _, done, _ = env.step(0, 0.0, 0.0, obs)
                    last_obs[i] = env.reset() if done else nobs
                    continue

                a_idx, f_raw, s_raw = ba
                n_legal = len(obs.shot_meta)
                is_safety = a_idx >= n_legal

                # Soft policy target over [shots..., safety] = MAX_SHOTS+1 dims.
                tgt = np.zeros(MAX_SHOTS + 1, dtype=np.float32)
                keys = list(action_qs.keys())
                qv = np.array([action_qs[k] for k in keys], dtype=np.float32)
                soft = np.exp((qv - qv.max()) / max(softmax_temp, 1e-6))
                soft = soft / soft.sum()
                for k, sv in zip(keys, soft):
                    net_idx = MAX_SHOTS if k >= n_legal else k
                    tgt[net_idx] = sv

                for k in _OBS_KEYS:
                    roll_obs[k].append(getattr(obs, k))
                roll_gather.append(0 if is_safety else a_idx)
                roll_force.append(f_raw)
                roll_spin.append(s_raw)
                roll_tgt_dist.append(tgt)
                roll_tgt_value.append(float(np.clip(best_q, 0.0, 1.0)))
                roll_force_valid.append(0.0 if is_safety else 1.0)
                iter_decisions += 1
                recent_safety.append(1.0 if is_safety else 0.0)

                nobs, _, done, info = env.step(a_idx, f_raw, s_raw, obs)
                if done:
                    recent_lengths.append(env.total_shots)
                    last_obs[i] = env.reset()
                else:
                    last_obs[i] = nobs

        if iter_decisions == 0:
            continue

        # ── Update: multi-epoch SGD on the distillation buffer ───────────
        all_obs = {k: np.stack(roll_obs[k]) for k in _OBS_KEYS}
        all_gather = np.asarray(roll_gather, dtype=np.int64)
        all_force = np.asarray(roll_force, dtype=np.float32)
        all_spin = np.asarray(roll_spin, dtype=np.float32)
        all_tgt_dist = np.stack(roll_tgt_dist)
        all_tgt_value = np.asarray(roll_tgt_value, dtype=np.float32)
        all_fvalid = np.asarray(roll_force_valid, dtype=np.float32)
        N = iter_decisions

        # Placement buffer (few samples per iter; updated as one batch).
        Np = iter_placements
        if Np > 0:
            place_obs = {k: torch.from_numpy(np.stack(roll_place_obs[k])).to(device)
                         for k in _OBS_KEYS}
            place_x = torch.tensor(roll_place_x, dtype=torch.float32, device=device)
            place_y = torch.tensor(roll_place_y, dtype=torch.float32, device=device)
            place_v = torch.tensor(roll_place_value, dtype=torch.float32, device=device)

        tot_ce = tot_mf = tot_ms = tot_v = tot_ent = 0.0
        tot_place = 0.0
        n_upd = 0
        for _epoch in range(epochs_per_iter):
            perm = np.random.permutation(N)
            for start in range(0, N, batch_size):
                bi = perm[start:start + batch_size]
                obs_b = {k: torch.from_numpy(all_obs[k][bi]).to(device)
                         for k in _OBS_KEYS}
                gather_b = torch.from_numpy(all_gather[bi]).to(device)
                force_b = torch.from_numpy(all_force[bi]).to(device)
                spin_b = torch.from_numpy(all_spin[bi]).to(device)
                tgt_dist_b = torch.from_numpy(all_tgt_dist[bi]).to(device)
                tgt_value_b = torch.from_numpy(all_tgt_value[bi]).to(device)
                fvalid_b = torch.from_numpy(all_fvalid[bi]).to(device)

                scores, f_means, s_means, safety_logit, value, _ = net.forward(**obs_b)
                combined = torch.cat([scores, safety_logit], dim=-1)  # (B, S+1)
                log_probs = F.log_softmax(combined, dim=-1)
                ce_loss = -(tgt_dist_b * log_probs).sum(-1).mean()

                # Force/spin MSE only on real-shot (non-safety) samples.
                f_chosen = f_means.gather(1, gather_b.unsqueeze(-1)).squeeze(-1)
                s_chosen = s_means.gather(1, gather_b.unsqueeze(-1)).squeeze(-1)
                denom = fvalid_b.sum().clamp(min=1.0)
                mse_f = (((f_chosen - force_b) ** 2) * fvalid_b).sum() / denom
                mse_s = (((s_chosen - spin_b) ** 2) * fvalid_b).sum() / denom

                value_loss = F.binary_cross_entropy(value, tgt_value_b)

                probs = log_probs.exp()
                ent = -(probs * log_probs).sum(-1).mean()

                loss = (ce_weight * ce_loss
                        + mse_force_weight * mse_f
                        + mse_spin_weight * mse_s
                        + value_weight * value_loss
                        - entropy_weight * ent)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(min=log_std_min)

                tot_ce += ce_loss.item(); tot_mf += mse_f.item()
                tot_ms += mse_s.item(); tot_v += value_loss.item()
                tot_ent += ent.item(); n_upd += 1

            # Placement update: distill the searched (x,y) into the placement
            # head (MSE on its mean) + value target at the placed state. One
            # batch per epoch since placement samples are sparse.
            if Np > 0:
                _, _, _, _, p_value, p_pooled = net.forward(**place_obs)
                p_mu = torch.sigmoid(net.placement_head(p_pooled))  # (Np, 2)
                place_mse = (((p_mu[:, 0] - place_x) ** 2
                              + (p_mu[:, 1] - place_y) ** 2)).mean()
                place_value_loss = F.binary_cross_entropy(p_value, place_v)
                ploss = place_weight * place_mse + value_weight * place_value_loss
                opt.zero_grad()
                ploss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                tot_place += place_mse.item()

        if (iteration + 1) % 10 == 0:
            el = time.time() - t0
            avg_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            safety_rate = float(np.mean(recent_safety)) if recent_safety else 0.0
            print(f'Iter {iteration+1:5d} | dec={iter_decisions} '
                  f'AvgLen={avg_len:5.1f} SafetyRate={safety_rate:.2f} | '
                  f'CE={tot_ce/n_upd:.4f} MSEf={tot_mf/n_upd:.3f} '
                  f'MSEs={tot_ms/n_upd:.3f} V={tot_v/n_upd:.4f} '
                  f'Ent={tot_ent/n_upd:.3f} '
                  f'Plc={iter_placements}(mse={tot_place/max(1,epochs_per_iter):.3f}) '
                  f'| {el:.0f}s', flush=True)
            # "best" by lowest CE proxy → save latest; keep a best by CE.
            metric = -tot_ce / n_upd
            if metric > best_metric:
                best_metric = metric
                torch.save(net.state_dict(),
                           f'checkpoints/eight_ball_{tag}_best.pt')

        if (iteration + 1) % 50 == 0:
            torch.save(net.state_dict(),
                       f'checkpoints/eight_ball_{tag}_latest.pt')

    torch.save(net.state_dict(), f'checkpoints/eight_ball_{tag}_final.pt')
    print(f'Done in {time.time()-t0:.0f}s', flush=True)


def main():
    p = argparse.ArgumentParser(description='8-ball search-distillation training')
    p.add_argument('--tag', default='8ball_v6_distill')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--envs', type=int, default=6)
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--search_k', type=int, default=4)
    p.add_argument('--search_m', type=int, default=1)
    p.add_argument('--search_mc', type=int, default=1)
    p.add_argument('--softmax_temp', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--mse_force_weight', type=float, default=0.1)
    p.add_argument('--mse_spin_weight', type=float, default=0.1)
    p.add_argument('--value_weight', type=float, default=0.5)
    p.add_argument('--entropy_weight', type=float, default=0.005)
    p.add_argument('--log_std_min', type=float, default=-3.0)
    p.add_argument('--warm_start', type=str, default=None)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--aim_noise_deg', type=float, default=0.0)
    p.add_argument('--force_noise_pct', type=float, default=0.0)
    p.add_argument('--spin_noise', type=float, default=0.0)
    p.add_argument('--shape_reward_weight', type=float, default=0.05)
    p.add_argument('--place_n', type=int, default=8,
                   help='Candidate cue positions sampled per BIH placement search')
    p.add_argument('--place_weight', type=float, default=1.0,
                   help='Weight on placement-mean MSE in the distill loss')
    args = p.parse_args()

    env_kwargs = dict(aim_noise_deg=args.aim_noise_deg,
                      force_noise_pct=args.force_noise_pct,
                      spin_noise=args.spin_noise,
                      shape_reward_weight=args.shape_reward_weight)
    train_distill(
        num_envs=args.envs, device_name=args.device, max_iters=args.iters,
        tag=args.tag, lr=args.lr, steps_per_iter=args.steps,
        search_k=args.search_k, search_m=args.search_m, search_mc=args.search_mc,
        gamma=args.gamma, softmax_temp=args.softmax_temp,
        ce_weight=args.ce_weight, mse_force_weight=args.mse_force_weight,
        mse_spin_weight=args.mse_spin_weight, value_weight=args.value_weight,
        entropy_weight=args.entropy_weight, log_std_min=args.log_std_min,
        warm_start=args.warm_start, env_kwargs=env_kwargs,
        place_n=args.place_n, place_weight=args.place_weight)


if __name__ == '__main__':
    main()
