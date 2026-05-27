"""
Curriculum training for attention network - Phase 1: Learn to hit a ball.

Single ball close to the cue ball. The network must discover that
aim_sin/cos should point toward the ball position. Dense reward for
contact makes this learnable quickly even with continuous actions.

CPU-friendly: small batch, fast iterations for local testing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import time
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
from pool_attention_net import PoolAttentionNet
from pool_sim import simulate_shot

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
R = 1.125


class Phase1Env:
    """
    Phase 1: One ball, close to cue. Learn to hit it.

    The ball is placed 5-15 inches from the cue ball at a random angle.
    Contact = big reward. Proximity after shot = small reward.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Cue ball in a random position (away from rails)
        self.cue = [20 + random.random() * 60, 10 + random.random() * 30]

        # One ball, 3-8 inches from cue (close = easier to hit randomly)
        angle = random.random() * 2 * math.pi
        dist = 3 + random.random() * 5
        bx = self.cue[0] + math.cos(angle) * dist
        by = self.cue[1] + math.sin(angle) * dist
        # Clamp to table
        bx = max(R * 2, min(TABLE_LENGTH - R * 2, bx))
        by = max(R * 2, min(TABLE_WIDTH - R * 2, by))
        self.ball_pos = [bx, by]
        self.pocketed = False
        return self.get_obs()

    def get_obs(self):
        """38-dim observation matching PoolAttentionNet input."""
        obs = np.full(38, -1.0, dtype=np.float32)
        # Cue ball
        obs[0] = self.cue[0] / TABLE_LENGTH
        obs[1] = self.cue[1] / TABLE_WIDTH
        # Ball 1
        if not self.pocketed:
            obs[2] = self.ball_pos[0] / TABLE_LENGTH
            obs[3] = self.ball_pos[1] / TABLE_WIDTH
        # Game state (minimal)
        obs[32] = 0.0  # score
        obs[33] = 0.0  # opp score
        obs[34] = 1.0 / 15.0 if not self.pocketed else 0.0  # balls on table
        obs[35] = 0.0  # fouls
        obs[36] = 0.0  # rerack
        obs[37] = 0.0  # reserved
        return obs

    def step(self, aim_angle, force, contact_y):
        """Execute shot, return (reward, done, info)."""
        # Map contact_y to spin
        spin = 1 if contact_y > 0.33 else (2 if contact_y < -0.33 else 0)

        aim_dx = math.cos(aim_angle)
        aim_dy = math.sin(aim_angle)

        if not self.pocketed:
            balls = {1: (self.ball_pos[0], self.ball_pos[1])}
        else:
            balls = {}

        result = simulate_shot(
            tuple(self.cue), balls,
            aim_dx * force, aim_dy * force,
            spin, aim_dx, aim_dy
        )

        # Update cue position
        self.cue = list(result.final_positions[0])

        # Check contact and pocketing
        reward = 0.0
        hit = result.hit_ball
        pocketed = 1 in result.pocketed_ids

        if pocketed:
            reward += 10.0  # huge reward for pocketing
            self.pocketed = True
        elif hit:
            reward += 3.0   # big reward for contact
            # Update ball position
            if 1 in result.final_positions:
                self.ball_pos = list(result.final_positions[1])
        else:
            # No contact — penalty based on how far the aim was from the ball
            # This gives gradient: "you were close to hitting, adjust a little"
            dx = self.ball_pos[0] - self.cue[0]
            dy = self.ball_pos[1] - self.cue[1]
            ball_angle = math.atan2(dy, dx)
            angle_diff = abs(aim_angle - ball_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            # Reward inversely proportional to angle error (max 1.0 for perfect aim)
            aim_quality = max(0, 1.0 - angle_diff / math.pi)
            reward += aim_quality * 0.5 - 0.5  # ranges from -0.5 (opposite) to 0 (perfect miss)

        info = {'hit': hit, 'pocketed': pocketed}
        done = pocketed or hit  # Phase 1: reset every shot so ball doesn't drift far from cue

        return reward, done, info


class VecPhase1:
    """Vectorized Phase 1 environments."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [Phase1Env() for _ in range(num_envs)]

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs = np.zeros((self.num_envs, 38), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        hits = 0
        pockets = 0

        for i, (env, act) in enumerate(zip(self.envs, actions)):
            aim_angle = np.arctan2(float(act[0]), float(act[1]))
            force = 30.0  # FIXED: medium speed, only learn to aim
            contact_y = 0.0  # FIXED: no spin, only learn to aim

            reward, done, info = env.step(aim_angle, force, contact_y)
            rewards[i] = reward
            dones[i] = done
            if info['hit']:
                hits += 1
            if info['pocketed']:
                pockets += 1

            if done:
                obs[i] = env.reset()
            else:
                obs[i] = env.get_obs()

        return obs, rewards, dones, {'hits': hits, 'pockets': pockets}


class RolloutBuffer:
    def __init__(self, num_envs, steps, obs_dim=38, act_dim=5):
        self.num_envs = num_envs
        self.steps = steps
        self.obs = np.zeros((steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((steps, num_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((steps, num_envs), dtype=np.float32)
        self.values = np.zeros((steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((steps, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, actions, rewards, dones, log_probs, values):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.ptr += 1

    def compute_returns(self, last_values, gamma=0.99, gae_lambda=0.95):
        last_gae = 0.0
        for t in reversed(range(self.steps)):
            next_values = last_values if t == self.steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        total = self.steps * self.num_envs
        indices = np.random.permutation(total)
        obs_flat = self.obs.reshape(total, -1)
        act_flat = self.actions.reshape(total, -1)
        lp_flat = self.log_probs.reshape(total)
        ret_flat = self.returns.reshape(total)
        adv_flat = self.advantages.reshape(total)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield (torch.FloatTensor(obs_flat[idx]),
                   torch.FloatTensor(act_flat[idx]),
                   torch.FloatTensor(lp_flat[idx]),
                   torch.FloatTensor(ret_flat[idx]),
                   torch.FloatTensor(adv_flat[idx]))


def train_phase1(num_envs=32, device_name='cpu', max_iters=2000):
    device = torch.device(device_name)
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2).to(device)
    # Only 2 outputs: aim_sin, aim_cos. log_std for these 2 dims only.
    net.log_std = nn.Parameter(torch.full((2,), -0.5))  # std=0.6, good exploration for angles
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)

    n_params = sum(p.numel() for p in net.parameters())
    print(f'Phase 1: Learn to hit a ball', flush=True)
    print(f'PoolAttentionNet: {n_params:,} params on {device}', flush=True)

    env = VecPhase1(num_envs)
    obs = env.reset()

    steps_per_update = 32
    batch_size = min(256, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    entropy_coef = 0.001
    value_coef = 0.5

    buffer = RolloutBuffer(num_envs, steps_per_update, act_dim=2)
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    best_hit_rate = 0.0
    hit_rates = []

    for iteration in range(max_iters):
        buffer.ptr = 0
        total_hits = 0
        total_pockets = 0
        total_shots = 0

        for step in range(steps_per_update):
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                actions, log_probs, values = net.get_action(obs_t)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, info = env.step(actions_np)

            buffer.add(obs, actions_np, rewards, dones.astype(np.float32),
                       log_probs.cpu().numpy(), values.cpu().numpy())

            obs = next_obs
            total_hits += info['hits']
            total_pockets += info['pockets']
            total_shots += num_envs

        # GAE
        with torch.no_grad():
            _, last_values = net(torch.FloatTensor(obs).to(device))
            last_values = last_values.cpu().numpy()
        buffer.compute_returns(last_values)

        # PPO update
        total_pg = 0.0
        total_vl = 0.0
        total_ent = 0.0
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
                total_pg += pg_loss.item()
                total_vl += v_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        hit_rate = total_hits / total_shots
        pocket_rate = total_pockets / total_shots
        hit_rates.append(hit_rate)

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_hr = np.mean(hit_rates[-50:])
            print(f'Iter {iteration+1:5d} | '
                  f'HitRate={hit_rate:.1%} AvgHR={avg_hr:.1%} '
                  f'PocketRate={pocket_rate:.1%} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | '
                  f'{elapsed:.0f}s', flush=True)

            if avg_hr > best_hit_rate:
                best_hit_rate = avg_hr
                torch.save(net.state_dict(), 'checkpoints/phase1_best.pt')
                if avg_hr > 0.5:
                    print(f'  -> Hit rate {avg_hr:.1%}! Aiming is emerging.', flush=True)
                if avg_hr > 0.8:
                    print(f'  -> HIT RATE {avg_hr:.1%}! Phase 1 COMPLETE. Ready for Phase 2.', flush=True)

        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), 'checkpoints/phase1_latest.pt')

    print(f'Done. Best hit rate: {best_hit_rate:.1%} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=2000)
    args = parser.parse_args()
    train_phase1(num_envs=args.envs, device_name=args.device, max_iters=args.iters)
