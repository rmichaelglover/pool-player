"""
Phase 5: 3-shot break-shot curriculum.

Env has:
  - 1 open ball (random position on table, away from rack)
  - 1 break ball (5-10 inches in front of the rack's apex)
  - 5-ball tight rack cluster (unpocketable until scattered)
  - cue ball placed with a plausible shot at either of the non-rack balls

Episode:
  - Up to 3 shots per episode
  - Reward = +10 per ball pocketed in the shot
  - Episode ends on: miss (no ball pocketed), scratch, or shot 3

Teaches simultaneously (if it works):
  - Ball selection: save the break ball for when you want to open the rack
  - Position play: land cue on break ball for a good break angle
  - Break quality: drive enough energy into the rack to free at least one ball

Action space matches Phase 4: 4-dim (aim_sin, aim_cos, force_raw, spin_raw).
Observation layout:
  [0:2]   cue (normalized)
  [2:4]   open ball (or -1,-1 if pocketed)
  [4:6]   break ball (or -1,-1 if pocketed)
  [6:16]  five rack balls (or -1,-1 each if pocketed)
  [16:32] unused (all -1)
  [32]    balls_remaining / 15
  [33:38] reserved
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

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
R = 1.125

OPEN_ID = 1
BREAK_ID = 2
RACK_IDS = (3, 4, 5, 6, 7)

RACK_APEX = (80.0, 25.0)
# 5-ball "rack" cluster: apex at foot-spot-ish, two rows behind.
# Ball spacing = 2R + epsilon so they're touching (unpocketable until scattered).
_d = 2 * R + 0.02
_RACK_OFFSETS = [
    (0.0,           0.0),
    (_d * 0.866,   -_d * 0.5),
    (_d * 0.866,    _d * 0.5),
    (_d * 1.732,   -_d),
    (_d * 1.732,    _d),
]
RACK_POSITIONS = [(RACK_APEX[0] + dx, RACK_APEX[1] + dy) for dx, dy in _RACK_OFFSETS]


def sample_phase5_setup():
    """Place cue, open ball, break ball. Rack is fixed at RACK_POSITIONS.
    Returns (cue_pos, balls_dict)."""
    # Break ball: 5-10 inches from rack apex, within a forward-facing cone
    # (toward the head of the table so the cue has line-of-sight to it).
    for _ in range(40):
        dist = 5.0 + random.random() * 5.0
        # Angle measured from rack apex toward the head (−x direction).
        # Cone of ±40 degrees.
        angle = math.pi + (random.random() - 0.5) * math.radians(80)
        bx = RACK_APEX[0] + dist * math.cos(angle)
        by = RACK_APEX[1] + dist * math.sin(angle)
        if not (3 * R < bx < TABLE_LENGTH - 3 * R):
            continue
        if not (3 * R < by < TABLE_WIDTH - 3 * R):
            continue
        break_ball = (bx, by)
        break
    else:
        break_ball = (72.0, 25.0)  # fallback

    # Open ball: anywhere in the head-half, clear of break ball.
    for _ in range(40):
        ox = 15.0 + random.random() * 45.0
        oy = 8.0 + random.random() * 34.0
        too_close_break = math.hypot(ox - break_ball[0], oy - break_ball[1]) < 8.0
        too_close_rack = math.hypot(ox - RACK_APEX[0], oy - RACK_APEX[1]) < 12.0
        if too_close_break or too_close_rack:
            continue
        open_ball = (ox, oy)
        break
    else:
        open_ball = (40.0, 25.0)

    # Cue: in head-quarter of the table with line-of-sight to the open ball.
    # We don't enforce clear line-of-sight rigorously — just keep it spaced out.
    for _ in range(40):
        cx = 10.0 + random.random() * 20.0
        cy = 8.0 + random.random() * 34.0
        too_close_open = math.hypot(cx - open_ball[0], cy - open_ball[1]) < 8.0
        too_close_break = math.hypot(cx - break_ball[0], cy - break_ball[1]) < 8.0
        if too_close_open or too_close_break:
            continue
        cue = [cx, cy]
        break
    else:
        cue = [15.0, 25.0]

    balls = {OPEN_ID: list(open_ball), BREAK_ID: list(break_ball)}
    for bid, pos in zip(RACK_IDS, RACK_POSITIONS):
        balls[bid] = list(pos)
    return cue, balls


class Phase5Env:
    """3-shot break curriculum env."""

    def __init__(self, pocket_reward=10.0, max_shots=3):
        self.pocket_reward = pocket_reward
        self.max_shots = max_shots
        self.reset()

    def reset(self):
        self.cue, self.balls = sample_phase5_setup()
        self.shot_idx = 0
        self.done = False
        self.last_shot_info = {}
        return self.get_obs()

    def get_obs(self):
        obs = np.full(38, -1.0, dtype=np.float32)
        obs[0] = self.cue[0] / TABLE_LENGTH
        obs[1] = self.cue[1] / TABLE_WIDTH
        # Fixed slots per semantic role:
        if OPEN_ID in self.balls:
            obs[2] = self.balls[OPEN_ID][0] / TABLE_LENGTH
            obs[3] = self.balls[OPEN_ID][1] / TABLE_WIDTH
        if BREAK_ID in self.balls:
            obs[4] = self.balls[BREAK_ID][0] / TABLE_LENGTH
            obs[5] = self.balls[BREAK_ID][1] / TABLE_WIDTH
        for i, bid in enumerate(RACK_IDS):
            if bid in self.balls:
                obs[6 + i * 2] = self.balls[bid][0] / TABLE_LENGTH
                obs[7 + i * 2] = self.balls[bid][1] / TABLE_WIDTH
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

        reward = 0.0
        info = {
            'pocketed_count': pocketed_obj_count,
            'pocketed_open': OPEN_ID in pocketed_ids,
            'pocketed_break': BREAK_ID in pocketed_ids,
            'pocketed_rack': len(pocketed_ids & set(RACK_IDS)),
            'scratch': scratch,
            'hit_ball': result.hit_ball,
        }

        if scratch:
            self.done = True
            return self.get_obs(), 0.0, True, info

        if pocketed_obj_count == 0:
            # Miss → episode ends (no reward).
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


class VecPhase5:
    def __init__(self, num_envs, pocket_reward=10.0, max_shots=3):
        self.num_envs = num_envs
        self.envs = [Phase5Env(pocket_reward=pocket_reward, max_shots=max_shots)
                     for _ in range(num_envs)]

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, raw_actions):
        obs = np.zeros((self.num_envs, 38), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        stats = {
            'shots_by_idx': [0, 0, 0],
            'pockets_by_idx': [0, 0, 0],
            'breaks_triggered': 0,  # shots that pocketed >1 ball in one shot
            'run3_episodes': 0,     # episodes that made it to 3 pockets
            'episodes_finished': 0,
            'total_pockets_this_step': 0,
        }
        for i, (env, raw) in enumerate(zip(self.envs, raw_actions)):
            shot_idx_before = env.shot_idx
            aim, force, spin = decode_action(raw)
            next_obs, r, d, info = env.step(aim, force, spin)
            rewards[i] = r
            dones[i] = d
            stats['shots_by_idx'][shot_idx_before] += 1
            if info['pocketed_count'] > 0:
                stats['pockets_by_idx'][shot_idx_before] += 1
            if info['pocketed_count'] > 1:
                stats['breaks_triggered'] += 1
            stats['total_pockets_this_step'] += info['pocketed_count']
            if d:
                stats['episodes_finished'] += 1
                # "run3" = pocketed on all 3 shots (or reached shot 3 with something pocketed)
                # We use: episode ended after 3 successful shots.
                if env.shot_idx >= env.max_shots and info['pocketed_count'] > 0:
                    stats['run3_episodes'] += 1
                next_obs = env.reset()
            obs[i] = next_obs
        return obs, rewards, dones, stats


def train_phase5(num_envs=32, device_name='cpu', max_iters=1500,
                 tag='p5_baseline', lr=1e-4, steps_per_update=64,
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
    print(f'Phase 5: 3-shot break curriculum. PoolAttentionNet {n_params:,} params on {device}', flush=True)
    print(f'config: tag={tag} lr={lr} steps={steps_per_update} envs={num_envs} '
          f'ent={entropy_coef} pocket_r={pocket_reward}', flush=True)

    env = VecPhase5(num_envs, pocket_reward=pocket_reward)
    obs = env.reset()

    batch_size = min(512, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    value_coef = 0.5
    buffer = RolloutBuffer(num_envs, steps_per_update, obs_dim=38, act_dim=ACT_DIM)

    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    best_avg_total = 0.0
    recent_total = []

    for iteration in range(max_iters):
        buffer.ptr = 0
        tot_shots = [0, 0, 0]
        tot_pockets = [0, 0, 0]
        tot_breaks = 0
        tot_run3 = 0
        tot_episodes = 0
        tot_pockets_all = 0

        for step in range(steps_per_update):
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                actions, log_probs, values = net.get_action(obs_t)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, stats = env.step(actions_np)
            buffer.add(obs, actions_np, rewards, dones.astype(np.float32),
                       log_probs.cpu().numpy(), values.cpu().numpy())
            obs = next_obs
            for i in range(3):
                tot_shots[i] += stats['shots_by_idx'][i]
                tot_pockets[i] += stats['pockets_by_idx'][i]
            tot_breaks += stats['breaks_triggered']
            tot_run3 += stats['run3_episodes']
            tot_episodes += stats['episodes_finished']
            tot_pockets_all += stats['total_pockets_this_step']

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

        hit_rates = [tot_pockets[i] / max(1, tot_shots[i]) for i in range(3)]
        avg_pockets_per_ep = tot_pockets_all / max(1, tot_episodes)
        recent_total.append(avg_pockets_per_ep)

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_last = float(np.mean(recent_total[-50:]))
            print(f'Iter {iteration+1:5d} | '
                  f'HR1={hit_rates[0]:.1%} HR2={hit_rates[1]:.1%} HR3={hit_rates[2]:.1%} | '
                  f'AvgPkt/ep={avg_pockets_per_ep:.2f} Recent={avg_last:.2f} | '
                  f'BreakShots={tot_breaks} Run3={tot_run3}/{tot_episodes} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)
            if avg_last > best_avg_total:
                best_avg_total = avg_last
                torch.save(net.state_dict(), f'checkpoints/phase5_{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f'checkpoints/phase5_{tag}_latest.pt')

    print(f'Done. Best avg pockets/episode: {best_avg_total:.2f} in {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=1500)
    parser.add_argument('--tag', default='p5_baseline')
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
    train_phase5(
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
