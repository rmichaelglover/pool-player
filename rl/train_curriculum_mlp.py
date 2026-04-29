"""
Phase 1 curriculum with a TINY MLP instead of the 438K transformer.

Sanity check: is the aim-only task (1 ball close, 2 outputs, fixed force/spin)
actually learnable with the current reward shaping? If this tiny MLP crosses
the 15% random baseline and climbs toward 50-80%, then the transformer stall
is an architecture/capacity issue, not a reward/task issue.

Network: 38 -> 64 -> 64 -> [actor(2), critic(1)]  (~7K params)
Everything else (Phase1Env, reward, PPO hyperparams) mirrors train_curriculum.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_curriculum import VecPhase1, RolloutBuffer


class TinyMLP(nn.Module):
    """Configurable MLP. Defaults to 38 -> 128 -> 128 -> actor/critic with ReLU."""

    def __init__(self, obs_dim=38, hidden=128, act_dim=2, activation='relu'):
        super().__init__()
        self.act_dim = act_dim
        Act = nn.ReLU if activation == 'relu' else nn.Tanh
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            Act(),
            nn.Linear(hidden, hidden),
            Act(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))  # std=0.6

        # Orthogonal init matching PoolAttentionNet conventions
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs, deterministic=False):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def train_phase1_mlp(num_envs=32, device_name='cpu', max_iters=500):
    device = torch.device(device_name)
    net = TinyMLP().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)

    n_params = sum(p.numel() for p in net.parameters())
    print(f'Phase 1 MLP sanity test', flush=True)
    print(f'TinyMLP: {n_params:,} params on {device}', flush=True)

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

        with torch.no_grad():
            _, last_values = net(torch.FloatTensor(obs).to(device))
            last_values = last_values.cpu().numpy()
        buffer.compute_returns(last_values)

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
                torch.save(net.state_dict(), 'checkpoints/phase1_mlp_best.pt')

        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), 'checkpoints/phase1_mlp_latest.pt')

    print(f'Done. Best hit rate: {best_hit_rate:.1%} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=500)
    args = parser.parse_args()
    train_phase1_mlp(num_envs=args.envs, device_name=args.device, max_iters=args.iters)
