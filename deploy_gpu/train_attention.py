"""
PPO training with attention-based pool network on GPU.

End-to-end learning: raw ball positions -> continuous shot parameters.
No hand-coded heuristics. Everything discovered through self-play.

For H100: set num_envs=1024, device='cuda'
For CPU testing: set num_envs=32, device='cpu'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from vec_pool_env import VectorizedPoolEnv


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

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

    def compute_returns(self, last_values, gamma=0.995, gae_lambda=0.95):
        """Compute GAE advantages and returns."""
        last_gae = 0.0
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Yield random minibatches for PPO epochs."""
        total = self.steps * self.num_envs
        indices = np.random.permutation(total)

        obs_flat = self.obs.reshape(total, -1)
        act_flat = self.actions.reshape(total, -1)
        lp_flat = self.log_probs.reshape(total)
        ret_flat = self.returns.reshape(total)
        adv_flat = self.advantages.reshape(total)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield (
                torch.FloatTensor(obs_flat[idx]),
                torch.FloatTensor(act_flat[idx]),
                torch.FloatTensor(lp_flat[idx]),
                torch.FloatTensor(ret_flat[idx]),
                torch.FloatTensor(adv_flat[idx]),
            )


def train(num_envs=32, device_name='cpu', total_iterations=10000):
    device = torch.device(device_name)

    # Network
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'PoolAttentionNet: {n_params:,} params on {device}', flush=True)

    # Environment
    env = VectorizedPoolEnv(num_envs)
    obs = env.reset()

    # Hyperparameters
    steps_per_update = 64       # steps per env before PPO update
    batch_size = min(512, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    gamma = 0.995
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5

    buffer = RolloutBuffer(num_envs, steps_per_update)
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    # Tracking
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = -999

    for iteration in range(total_iterations):
        # --- Collect rollouts ---
        buffer.ptr = 0
        iter_rewards = []

        for step in range(steps_per_update):
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                actions, log_probs, values = net.get_action(obs_t)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, infos = env.step(actions_np)

            buffer.add(obs, actions_np, rewards, dones.astype(np.float32),
                       log_probs.cpu().numpy(), values.cpu().numpy())

            obs = next_obs

            for i, (d, info) in enumerate(zip(dones, infos)):
                if d:
                    ep_reward = info.get('score_p1', 0) + info.get('score_p2', 0)
                    episode_rewards.append(ep_reward)

        # --- Compute GAE ---
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(device)
            _, last_values = net(obs_t)
            last_values = last_values.cpu().numpy()
        buffer.compute_returns(last_values, gamma, gae_lambda)

        # --- PPO update ---
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for epoch in range(ppo_epochs):
            for batch in buffer.get_batches(batch_size):
                b_obs, b_actions, b_old_lp, b_returns, b_advantages = [
                    x.to(device) for x in batch
                ]

                new_lp, entropy, values = net.evaluate_actions(b_obs, b_actions)

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(values, b_returns)
                ent_loss = -entropy.mean()

                loss = pg_loss + value_coef * v_loss + entropy_coef * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                opt.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_frac += clip_frac
                n_updates += 1

        # --- Logging ---
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_pg = total_pg_loss / max(n_updates, 1)
            avg_vl = total_v_loss / max(n_updates, 1)
            avg_ent = total_entropy / max(n_updates, 1)
            avg_clip = total_clip_frac / max(n_updates, 1)

            # Recent episode stats
            recent = episode_rewards[-100:] if episode_rewards else [0]
            avg_reward = np.mean(recent)
            max_reward = np.max(recent) if recent else 0

            steps_done = (iteration + 1) * steps_per_update * num_envs
            sps = steps_done / elapsed

            print(f'Iter {iteration+1:6d} | '
                  f'AvgReward={avg_reward:.1f} MaxReward={max_reward:.0f} | '
                  f'PG={avg_pg:.4f} VL={avg_vl:.2f} '
                  f'Ent={avg_ent:.3f} Clip={avg_clip:.2f} | '
                  f'Steps={steps_done:,} ({sps:.0f}/s) | '
                  f'{elapsed:.0f}s', flush=True)

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(net.state_dict(), 'checkpoints/best_attention.pt')
                print(f'  -> Best avg reward: {avg_reward:.1f}', flush=True)

        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), 'checkpoints/latest_attention.pt')

    print(f'Done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32, help='Parallel environments')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--iters', type=int, default=10000, help='Training iterations')
    args = parser.parse_args()
    train(num_envs=args.envs, device_name=args.device, total_iterations=args.iters)
