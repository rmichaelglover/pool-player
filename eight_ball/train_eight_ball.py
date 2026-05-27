"""
8-ball self-play training with PPO.

Both players share the same network. The env alternates current_player and
flips the observation perspective. Transitions from both sides feed into the
same PPO buffer. GAE is computed per-player within each game.

Curriculum phases:
  A: Single-player pocketing drill (7 balls, no opponent, no 8-ball rules)
  B: Full 8-ball single-player runout (all "mine", 8-ball-last)
  C: Two-player self-play (full rules)
  D: + historical opponent sampling and Elo tracking
"""
from __future__ import annotations

import argparse
import math
import os
import random
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
from eight_ball_net import (EightBallNet, EightBallObs, MAX_BALLS, MAX_POCKETS,
                            MAX_SHOTS, GAME_STATE_DIM, GROUP_CUE, GROUP_MINE,
                            GROUP_8BALL)
from eight_ball_env import EightBallEnv


# ── Rollout buffer ─────────────────────────────────────────────────────────

class EightBallBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.balls = []
        self.ball_mask = []
        self.ball_group = []
        self.pockets = []
        self.game_state = []
        self.shots = []
        self.shot_mask = []
        self.action_idx = []
        self.force_raw = []
        self.spin_raw = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.is_placement = []

    def add(self, obs: EightBallObs, action_idx, force_raw, spin_raw,
            reward, done, log_prob, value, is_placement=False):
        self.balls.append(obs.balls)
        self.ball_mask.append(obs.ball_mask)
        self.ball_group.append(obs.ball_group)
        self.pockets.append(obs.pockets)
        self.game_state.append(obs.game_state)
        self.shots.append(obs.shots)
        self.shot_mask.append(obs.shot_mask)
        self.action_idx.append(action_idx)
        self.force_raw.append(force_raw)
        self.spin_raw.append(spin_raw)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.is_placement.append(is_placement)

    def compute_returns(self, gamma=0.999, gae_lambda=0.95):
        n = len(self.rewards)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]
            not_terminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * not_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * not_terminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + self.values[t]

    def get_batches(self, batch_size, device):
        n = len(self.rewards)
        if n == 0:
            return
        idx = np.random.permutation(n)
        adv = np.array(self.advantages)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = idx[start:end]
            obs = {
                'balls': torch.from_numpy(np.stack([self.balls[i] for i in b])).to(device),
                'ball_mask': torch.from_numpy(np.stack([self.ball_mask[i] for i in b])).to(device),
                'ball_group': torch.from_numpy(np.stack([self.ball_group[i] for i in b])).to(device),
                'pockets': torch.from_numpy(np.stack([self.pockets[i] for i in b])).to(device),
                'game_state': torch.from_numpy(np.stack([self.game_state[i] for i in b])).to(device),
                'shots': torch.from_numpy(np.stack([self.shots[i] for i in b])).to(device),
                'shot_mask': torch.from_numpy(np.stack([self.shot_mask[i] for i in b])).to(device),
            }
            act_idx = torch.tensor([self.action_idx[i] for i in b], dtype=torch.long, device=device)
            f_raw = torch.tensor([self.force_raw[i] for i in b], dtype=torch.float32, device=device)
            s_raw = torch.tensor([self.spin_raw[i] for i in b], dtype=torch.float32, device=device)
            is_plc = torch.tensor([self.is_placement[i] for i in b], dtype=torch.bool, device=device)
            old_lp = torch.tensor([self.log_probs[i] for i in b], dtype=torch.float32, device=device)
            ret = torch.tensor([self.returns[i] for i in b], dtype=torch.float32, device=device)
            adv_b = torch.tensor(adv[b], dtype=torch.float32, device=device)
            yield obs, (act_idx, f_raw, s_raw, is_plc), (old_lp, ret, adv_b)


# ── Vectorized env ─────────────────────────────────────────────────────────

class VecEightBall:
    def __init__(self, num_envs, env_kwargs=None):
        kw = env_kwargs or {}
        self.num_envs = num_envs
        self.envs = [EightBallEnv(**kw) for _ in range(num_envs)]
        self.last_obs = [e.reset() for e in self.envs]

    def reset(self):
        self.last_obs = [e.reset() for e in self.envs]
        return self.last_obs

    def step_single(self, env_idx, action_idx, force_raw, spin_raw):
        env = self.envs[env_idx]
        obs = self.last_obs[env_idx]
        next_obs, reward, done, info = env.step(
            int(action_idx), float(force_raw), float(spin_raw), obs)
        if done:
            info['final_winner'] = env.winner
            info['final_shots'] = env.total_shots
            next_obs = env.reset()
        self.last_obs[env_idx] = next_obs
        return next_obs, reward, done, info

    def step_placement(self, env_idx, x_norm, y_norm):
        env = self.envs[env_idx]
        next_obs, reward, done, info = env.step_placement(
            float(x_norm), float(y_norm))
        if done:
            info['final_winner'] = env.winner
            info['final_shots'] = env.total_shots
            next_obs = env.reset()
        self.last_obs[env_idx] = next_obs
        return next_obs, reward, done, info


# ── Training loop ──────────────────────────────────────────────────────────

def train_eight_ball(num_envs=8, device_name='cpu', max_iters=1000,
                     tag='8ball_baseline', lr=1e-4,
                     steps_per_iter=64, entropy_coef=0.01,
                     log_std_min=-2.5, embed_dim=128, num_heads=8,
                     num_layers=4, warm_start=None, env_kwargs=None,
                     gamma=0.999, gae_lambda=0.95):
    device = torch.device(device_name)
    net = EightBallNet(embed_dim=embed_dim, num_heads=num_heads,
                       num_layers=num_layers).to(device)
    if warm_start and os.path.exists(warm_start):
        state = torch.load(warm_start, map_location=device, weights_only=True)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f'Warm-started from {warm_start}', flush=True)
        if missing:
            print(f'  New parameters (default init): {missing}', flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'8-ball self-play. EightBallNet {n_params:,} params on {device}', flush=True)
    print(f'config: tag={tag} lr={lr} steps={steps_per_iter} envs={num_envs} '
          f'gamma={gamma} ent={entropy_coef}', flush=True)

    vec_env = VecEightBall(num_envs, env_kwargs=env_kwargs)

    batch_size = 256
    ppo_epochs = 4
    clip_eps = 0.2
    value_coef = 0.5

    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    best_win_rate = 0.0
    recent_wins = deque(maxlen=500)   # 1 if p0 wins, 0 if p1 wins, 0.5 if draw
    recent_lengths = deque(maxlen=500)
    recent_fouls = deque(maxlen=2000)

    for iteration in range(max_iters):
        buffer = EightBallBuffer(steps_per_iter * num_envs)
        iter_games = 0
        iter_wins = {0: 0, 1: 0, None: 0}
        iter_lengths = []

        # Collect transitions
        total_steps = 0
        iter_placements = 0
        iter_placement_reward = 0.0
        while total_steps < steps_per_iter:
            for env_idx in range(num_envs):
                obs = vec_env.last_obs[env_idx]
                env = vec_env.envs[env_idx]

                obs_batch = {
                    'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(device),
                    'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(device),
                    'ball_group': torch.from_numpy(obs.ball_group).unsqueeze(0).to(device),
                    'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(device),
                    'game_state': torch.from_numpy(obs.game_state).unsqueeze(0).to(device),
                    'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(device),
                    'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(device),
                }

                if env.awaiting_placement:
                    with torch.no_grad():
                        action_idx, xn, yn, log_prob, value = net.get_action(obs_batch)
                    lp = log_prob.item()
                    val = value.item()
                    next_obs, reward, done, info = vec_env.step_placement(
                        env_idx, xn.item(), yn.item())
                    buffer.add(obs, 0, xn.item(), yn.item(),
                               reward, done, lp, val, is_placement=True)
                    iter_placements += 1
                    iter_placement_reward += reward
                else:
                    with torch.no_grad():
                        action_idx, force_raw, spin_raw, log_prob, value = net.get_action(obs_batch)
                    ai = action_idx.item()
                    fr = force_raw.item()
                    sr = spin_raw.item()
                    lp = log_prob.item()
                    val = value.item()
                    next_obs, reward, done, info = vec_env.step_single(env_idx, ai, fr, sr)
                    buffer.add(obs, ai, fr, sr, reward, done, lp, val, is_placement=False)

                if 'foul' in info:
                    recent_fouls.append(1)
                elif not info.get('is_placement_step'):
                    recent_fouls.append(0)

                if done:
                    winner = info.get('final_winner')
                    iter_games += 1
                    iter_wins[winner] += 1
                    iter_lengths.append(info.get('final_shots', 0))
                    recent_lengths.append(info.get('final_shots', 0))
                    if winner == 0:
                        recent_wins.append(1.0)
                    elif winner == 1:
                        recent_wins.append(0.0)
                    else:
                        recent_wins.append(0.5)

            total_steps += 1

        # Compute GAE
        buffer.compute_returns(gamma=gamma, gae_lambda=gae_lambda)

        # PPO update
        total_pg = total_vl = total_ent = 0.0
        n_updates = 0
        for epoch in range(ppo_epochs):
            for b_obs, b_act, b_trg in buffer.get_batches(batch_size, device):
                act_idx, f_raw, s_raw, is_plc = b_act
                old_lp, b_ret, b_adv = b_trg

                new_lp, entropy, values = net.evaluate_actions(
                    b_obs, act_idx, f_raw, s_raw, is_placement=is_plc)

                # Value loss: BCE since value is win probability in [0,1]
                # and returns are shaped rewards (may be outside [0,1]),
                # so clamp returns for BCE target
                v_target = b_ret.clamp(0.0, 1.0)
                v_loss = F.binary_cross_entropy(values, v_target)

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                loss = pg_loss + value_coef * v_loss - entropy_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(min=log_std_min)
                    net.placement_log_std.clamp_(min=log_std_min)

                total_pg += pg_loss.item()
                total_vl += v_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        # Logging
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_len = float(np.mean(list(recent_lengths))) if recent_lengths else 0.0
            win_rate = float(np.mean(list(recent_wins))) if recent_wins else 0.5
            foul_rate = float(np.mean(list(recent_fouls))) if recent_fouls else 0.0
            pg = total_pg / max(1, n_updates)
            vl = total_vl / max(1, n_updates)
            ent = total_ent / max(1, n_updates)

            avg_plc_r = iter_placement_reward / max(1, iter_placements)
            print(f'Iter {iteration+1:5d} | Games={iter_games} '
                  f'W0={iter_wins[0]} W1={iter_wins[1]} D={iter_wins.get(None,0)} | '
                  f'AvgLen={avg_len:5.1f} FoulRate={foul_rate:.2f} '
                  f'Plc={iter_placements}({avg_plc_r:+.3f}) | '
                  f'PG={pg:.4f} VL={vl:.4f} Ent={ent:.3f} | {elapsed:.0f}s',
                  flush=True)

            if n_updates > 0 and win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(net.state_dict(),
                           f'checkpoints/eight_ball_{tag}_best.pt')

        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(),
                       f'checkpoints/eight_ball_{tag}_latest.pt')

    print(f'Done in {time.time()-t0:.0f}s', flush=True)
    torch.save(net.state_dict(), f'checkpoints/eight_ball_{tag}_final.pt')


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='8-ball self-play training')
    p.add_argument('--tag', default='8ball_v1')
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.999)
    p.add_argument('--entropy_coef', type=float, default=0.01)
    p.add_argument('--warm_start', type=str, default=None)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--aim_noise_deg', type=float, default=0.0)
    p.add_argument('--force_noise_pct', type=float, default=0.0)
    p.add_argument('--spin_noise', type=float, default=0.0)
    args = p.parse_args()

    env_kwargs = dict(
        aim_noise_deg=args.aim_noise_deg,
        force_noise_pct=args.force_noise_pct,
        spin_noise=args.spin_noise,
    )
    train_eight_ball(
        num_envs=args.envs, device_name=args.device, max_iters=args.iters,
        tag=args.tag, lr=args.lr, steps_per_iter=args.steps,
        entropy_coef=args.entropy_coef, warm_start=args.warm_start,
        env_kwargs=env_kwargs, gamma=args.gamma,
    )


if __name__ == '__main__':
    main()
