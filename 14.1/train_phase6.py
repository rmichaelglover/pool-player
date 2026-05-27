"""
Phase 6a: Break a full 14.1 rack and run it out (no rerack).

Env:
  - 15 balls placed in a standard 14.1 triangular rack (5 rows: 1+2+3+4+5)
  - Head ball at the foot spot, rack extends toward the foot rail
  - Cue ball in the head "kitchen" (back quarter of the table)

Episode:
  - Shots continue as long as each shot pockets at least one ball
  - Episode ends on: scratch, or no ball pocketed in a shot, or all 15 pocketed
  - Reward = +10 per ball pocketed per shot
  - Max shots = 15 (one per ball)

Observation (38-dim, 16-ball capacity — fits 1 cue + 15 object perfectly):
  [0:2]    cue / (TL, TW)
  [2:32]   15 object balls (bid=1..15) or -1 if pocketed
  [32]     balls_remaining / 15
  [33:38]  reserved

Action: 4-dim (aim_sin, aim_cos, force_raw, spin_raw) — same as Phase 4/5.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
from pool_attention_net import PoolAttentionNet
from pool_sim import simulate_shot
from train_curriculum import RolloutBuffer
from train_phase4 import decode_action, ACT_DIM
from rack_geometry import (TABLE_LENGTH, TABLE_WIDTH, R,
                           RACK_APEX, RACK_POSITIONS, sample_phase6_setup)


class Phase6Env:
    """14.1 run-out env (no rerack)."""

    def __init__(self, pocket_reward=10.0, max_shots=15):
        self.pocket_reward = pocket_reward
        self.max_shots = max_shots
        self.reset()

    def reset(self):
        self.cue, self.balls = sample_phase6_setup()
        self.shot_idx = 0
        self.done = False
        return self.get_obs()

    def get_obs(self):
        obs = np.full(38, -1.0, dtype=np.float32)
        obs[0] = self.cue[0] / TABLE_LENGTH
        obs[1] = self.cue[1] / TABLE_WIDTH
        for bid in range(1, 16):
            idx = 2 + (bid - 1) * 2
            if bid in self.balls:
                obs[idx] = self.balls[bid][0] / TABLE_LENGTH
                obs[idx + 1] = self.balls[bid][1] / TABLE_WIDTH
        obs[32] = len(self.balls) / 15.0
        return obs

    def step(self, aim_angle, force, spin_factor):
        if self.done:
            return self.get_obs(), 0.0, True, {'pocketed_count': 0}

        aim_dx = math.cos(aim_angle)
        aim_dy = math.sin(aim_angle)
        balls_in_sim = {bid: tuple(pos) for bid, pos in self.balls.items()}
        result = simulate_shot(
            tuple(self.cue), balls_in_sim,
            aim_dx * force, aim_dy * force,
            spin_factor, aim_dx, aim_dy,
        )

        scratch = result.cue_scratched
        pocketed_ids = set(result.pocketed_ids)
        pocketed_obj_count = len(pocketed_ids)

        info = {
            'pocketed_count': pocketed_obj_count,
            'scratch': scratch,
            'hit_ball': result.hit_ball,
            'balls_remaining': len(self.balls) - pocketed_obj_count,
        }

        if scratch:
            self.done = True
            return self.get_obs(), 0.0, True, info

        if pocketed_obj_count == 0:
            self.done = True
            return self.get_obs(), 0.0, True, info

        # Apply reward, update positions.
        reward = self.pocket_reward * pocketed_obj_count
        for bid in pocketed_ids:
            if bid in self.balls:
                del self.balls[bid]
        if 0 in result.final_positions:
            self.cue = list(result.final_positions[0])
        for bid, pos in result.final_positions.items():
            if bid in self.balls:
                self.balls[bid] = list(pos)

        self.shot_idx += 1
        if self.shot_idx >= self.max_shots or not self.balls:
            self.done = True
        return self.get_obs(), reward, self.done, info


class VecPhase6:
    def __init__(self, num_envs, pocket_reward=10.0, max_shots=15):
        self.num_envs = num_envs
        self.envs = [Phase6Env(pocket_reward=pocket_reward, max_shots=max_shots)
                     for _ in range(num_envs)]

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, raw_actions):
        obs = np.zeros((self.num_envs, 38), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        stats = {
            'episodes_finished': 0,
            'total_pockets': 0,
            'run_lengths': [],    # balls pocketed per finished episode
            'break_pockets': 0,   # extra balls pocketed on shots that broke
        }
        for i, (env, raw) in enumerate(zip(self.envs, raw_actions)):
            aim, force, spin = decode_action(raw)
            next_obs, r, d, info = env.step(aim, force, spin)
            rewards[i] = r
            dones[i] = d
            stats['total_pockets'] += info['pocketed_count']
            if info['pocketed_count'] > 1:
                stats['break_pockets'] += info['pocketed_count'] - 1
            if d:
                stats['episodes_finished'] += 1
                run_len = 15 - info['balls_remaining']
                stats['run_lengths'].append(run_len)
                next_obs = env.reset()
            obs[i] = next_obs
        return obs, rewards, dones, stats


def train_phase6(num_envs=32, device_name='cpu', max_iters=500,
                 tag='p6_baseline', lr=1e-4, steps_per_update=64,
                 pocket_reward=10.0, log_std_min=-3.0,
                 entropy_coef=0.01, warm_start=None,
                 embed_dim=96, num_heads=6, num_layers=4, ff_dim=None):
    device = torch.device(device_name)
    if ff_dim is None:
        ff_dim = embed_dim * 2
    net = PoolAttentionNet(embed_dim=embed_dim, num_heads=num_heads,
                           num_layers=num_layers, ff_dim=ff_dim,
                           act_dim=ACT_DIM).to(device)
    net.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5).to(device))

    if warm_start and os.path.exists(warm_start):
        src = torch.load(warm_start, map_location=device, weights_only=True)
        dst = net.state_dict()
        loaded = 0
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k] = v
                loaded += 1
        net.load_state_dict(dst)
        print(f'Warm-started {loaded}/{len(src)} tensors from {warm_start}', flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Phase 6a: 14.1 run-out (no rerack). PoolAttentionNet {n_params:,} params on {device}', flush=True)
    print(f'config: tag={tag} lr={lr} steps={steps_per_update} envs={num_envs} '
          f'ent={entropy_coef} pocket_r={pocket_reward}', flush=True)

    env = VecPhase6(num_envs, pocket_reward=pocket_reward)
    obs = env.reset()

    batch_size = min(512, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    value_coef = 0.5
    buffer = RolloutBuffer(num_envs, steps_per_update, obs_dim=38, act_dim=ACT_DIM)

    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    best_avg_runlen = 0.0
    recent_runlens = []

    for iteration in range(max_iters):
        buffer.ptr = 0
        iter_run_lengths = []
        iter_episodes = 0
        iter_total_pockets = 0
        iter_break_pockets = 0

        for step in range(steps_per_update):
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                actions, log_probs, values = net.get_action(obs_t)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, stats = env.step(actions_np)
            buffer.add(obs, actions_np, rewards, dones.astype(np.float32),
                       log_probs.cpu().numpy(), values.cpu().numpy())
            obs = next_obs
            iter_run_lengths.extend(stats['run_lengths'])
            iter_episodes += stats['episodes_finished']
            iter_total_pockets += stats['total_pockets']
            iter_break_pockets += stats['break_pockets']

        with torch.no_grad():
            _, last_values = net(torch.FloatTensor(obs).to(device))
            last_values = last_values.cpu().numpy()
        buffer.compute_returns(last_values)

        total_pg = total_vl = total_ent = 0.0
        n_updates = 0
        for epoch in range(ppo_epochs):
            for batch in buffer.get_batches(batch_size):
                b_obs, b_act, b_old_lp, b_ret, b_adv = [x.to(device) for x in batch]
                new_lp, entropy, values = net.evaluate_actions(b_obs, b_act)
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values, b_ret)
                loss = pg_loss + value_coef * v_loss - entropy_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(min=log_std_min)
                total_pg += pg_loss.item()
                total_vl += v_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        avg_runlen = float(np.mean(iter_run_lengths)) if iter_run_lengths else 0.0
        max_runlen = int(np.max(iter_run_lengths)) if iter_run_lengths else 0
        recent_runlens.extend(iter_run_lengths)
        recent_runlens = recent_runlens[-500:]  # keep rolling window

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            rolling_avg = float(np.mean(recent_runlens)) if recent_runlens else 0.0
            print(f'Iter {iteration+1:5d} | '
                  f'AvgRun={avg_runlen:4.2f} Rolling={rolling_avg:4.2f} MaxRun={max_runlen:2d} | '
                  f'Eps={iter_episodes} Pkts={iter_total_pockets} BrkExtras={iter_break_pockets} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)
            if rolling_avg > best_avg_runlen:
                best_avg_runlen = rolling_avg
                torch.save(net.state_dict(), f'checkpoints/phase6_{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f'checkpoints/phase6_{tag}_latest.pt')

    print(f'Done. Best rolling avg run length: {best_avg_runlen:.2f} in {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--tag', default='p6_baseline')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps_per_update', type=int, default=64)
    parser.add_argument('--pocket_reward', type=float, default=10.0)
    parser.add_argument('--log_std_min', type=float, default=-3.0)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--warm', default=None)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=None)
    args = parser.parse_args()
    train_phase6(
        num_envs=args.envs, device_name=args.device,
        max_iters=args.iters, tag=args.tag,
        lr=args.lr, steps_per_update=args.steps_per_update,
        pocket_reward=args.pocket_reward,
        log_std_min=args.log_std_min,
        entropy_coef=args.entropy_coef,
        warm_start=args.warm,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, ff_dim=args.ff_dim,
    )
