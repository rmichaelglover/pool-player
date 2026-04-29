"""
PPO training with Transformer policy for 14.1 Continuous pool.
Usage: python3 train_pool.py [test]
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Install with: pip3 install torch")

from pool_env import PoolEnv
from pool_env_geometric import PoolEnvGeometric
from config import Config

if HAS_TORCH:
    from policy_network import TransformerActorCritic, RolloutBuffer


class VecEnv:
    """Simple vectorized environment."""

    def __init__(self, num_envs, target_score=10, curriculum_balls=15, use_geometric=True):
        EnvClass = PoolEnvGeometric if use_geometric else PoolEnv
        self.envs = [EnvClass(target_score=target_score, num_object_balls=curriculum_balls)
                     if use_geometric else
                     EnvClass(target_score=target_score, curriculum_balls=curriculum_balls)
                     for _ in range(num_envs)]
        self.num_envs = num_envs
        self.use_geometric = use_geometric

    def reset(self):
        return np.array([env.reset()[0] for env in self.envs])

    def step(self, actions):
        results = [env.step(actions[i]) for i, env in enumerate(self.envs)]
        obs = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        terminateds = np.array([r[2] for r in results])
        truncateds = np.array([r[3] for r in results])
        infos = [r[4] for r in results]
        dones = terminateds | truncateds
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()[0]
        return obs, rewards, dones, infos

    def update_config(self, target_score, curriculum_balls):
        for env in self.envs:
            env.target_score = target_score
            env.curriculum_balls = curriculum_balls


def train():
    if not HAS_TORCH:
        print("Cannot train without PyTorch.")
        return

    cfg = Config()

    print("=" * 60)
    print("14.1 Pool -- PPO + Transformer Training")
    print("=" * 60)
    print(f"Architecture: embed={cfg.embed_dim}, heads={cfg.num_heads}, "
          f"layers={cfg.num_layers}, ff={cfg.ff_dim}")
    print(f"Tokens: 16 balls + 6 pockets = 22")
    print(f"Envs: {cfg.num_envs}, Steps/env: {cfg.num_steps_per_env}")
    act_dim = 2 if cfg.use_geometric else cfg.act_dim  # geometric: just aim + force
    print(f"Action dim: {act_dim} ({'aim+force' if act_dim==2 else 'aim+force+spin+elev'})")
    print(f"LR: {cfg.learning_rate}, Gamma: {cfg.gamma}")
    print()

    # Create environments
    init_curriculum = cfg.curriculum_schedule[0]
    vec_env = VecEnv(cfg.num_envs,
                     target_score=init_curriculum['target_score'],
                     curriculum_balls=init_curriculum['curriculum_balls'],
                     use_geometric=cfg.use_geometric)
    env_type = "Geometric (instant)" if cfg.use_geometric else "Physics (simulated)"
    print(f"Environment: {env_type}")
    obs = vec_env.reset()

    # Create transformer policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    policy = TransformerActorCritic(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        act_dim=act_dim,
    ).to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,}")
    print()

    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    buffer = RolloutBuffer(cfg.num_envs, cfg.num_steps_per_env, cfg.obs_dim, act_dim)

    # Logging
    total_steps = 0
    ep_rewards = []
    ep_balls_pocketed = []
    best_avg_reward = -float('inf')
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()

    for iteration in range(cfg.max_iterations):
        # Curriculum update
        curr = init_curriculum
        for threshold, settings in sorted(cfg.curriculum_schedule.items()):
            if iteration >= threshold:
                curr = settings
        vec_env.update_config(curr['target_score'], curr['curriculum_balls'])

        # Collect rollout
        buffer.reset()
        iter_start = time.time()

        for step in range(cfg.num_steps_per_env):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                actions, log_probs, values = policy.get_action(obs_tensor)
                actions_np = actions.cpu().numpy()
                # Clamp actions
                actions_np[:, 0] = np.clip(actions_np[:, 0], 0, 2 * np.pi)  # aim angle
                actions_np[:, 1] = np.clip(actions_np[:, 1], 0, 1)          # force
                if act_dim > 2:
                    actions_np[:, 2] = np.clip(actions_np[:, 2], -1, 1)     # english x
                    actions_np[:, 3] = np.clip(actions_np[:, 3], -1, 1)     # english y
                    actions_np[:, 4] = np.clip(actions_np[:, 4], 0, 0.5)    # elevation

            next_obs, rewards, dones, infos = vec_env.step(actions_np)

            buffer.add(obs, actions_np, rewards, dones.astype(np.float32),
                      log_probs.cpu().numpy(), values.cpu().numpy())

            obs = next_obs
            total_steps += cfg.num_envs

            for i, info in enumerate(infos):
                if info.get('pocketed') and len(info['pocketed']) > 0:
                    ep_balls_pocketed.append(len(info['pocketed']))
                if dones[i]:
                    ep_rewards.append(info.get('score', 0))

        iter_collect_time = time.time() - iter_start

        # Compute returns
        with torch.no_grad():
            last_obs = torch.FloatTensor(obs).to(device)
            last_values = policy.get_action(last_obs)[2].cpu().numpy()
        buffer.compute_returns(last_values, gamma=cfg.gamma, lam=cfg.gae_lambda)

        # PPO update
        update_start = time.time()
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(cfg.num_epochs):
            for batch in buffer.get_batches(cfg.batch_size):
                obs_b, actions_b, old_log_probs_b, returns_b, advantages_b = [
                    x.to(device) for x in batch]

                log_probs_b, entropy_b, values_b = policy.evaluate_actions(obs_b, actions_b)

                ratio = torch.exp(log_probs_b - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1 - cfg.clip_param, 1 + cfg.clip_param) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values_b, returns_b)
                entropy_loss = -entropy_b.mean()

                loss = policy_loss + cfg.value_loss_coef * value_loss + cfg.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_b.mean().item()
                n_updates += 1

        update_time = time.time() - update_start

        # Logging
        if (iteration + 1) % cfg.log_interval == 0:
            elapsed = time.time() - start_time
            avg_policy_loss = total_policy_loss / max(n_updates, 1)
            avg_value_loss = total_value_loss / max(n_updates, 1)
            avg_entropy = total_entropy / max(n_updates, 1)
            avg_reward = np.mean(ep_rewards[-100:]) if ep_rewards else 0
            avg_balls = np.mean(ep_balls_pocketed[-100:]) if ep_balls_pocketed else 0

            print(f"Iter {iteration+1:6d} | "
                  f"Steps {total_steps:8d} | "
                  f"Collect {iter_collect_time:5.1f}s | "
                  f"Update {update_time:4.2f}s | "
                  f"Reward {avg_reward:6.2f} | "
                  f"Balls {avg_balls:4.1f} | "
                  f"PLoss {avg_policy_loss:.4f} | "
                  f"VLoss {avg_value_loss:.4f} | "
                  f"Ent {avg_entropy:.3f} | "
                  f"Curr: {curr['curriculum_balls']}balls/{curr['target_score']}pts")

            if avg_reward > best_avg_reward and len(ep_rewards) > 10:
                best_avg_reward = avg_reward
                torch.save(policy.state_dict(), os.path.join(save_dir, 'best_policy.pt'))
                print(f"  -> New best: {avg_reward:.2f}")

        if (iteration + 1) % cfg.save_interval == 0:
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_steps': total_steps,
                'config': {
                    'embed_dim': cfg.embed_dim, 'num_heads': cfg.num_heads,
                    'num_layers': cfg.num_layers, 'act_dim': cfg.act_dim,
                },
            }, os.path.join(save_dir, f'checkpoint_{iteration+1}.pt'))
            print(f"  Checkpoint saved")

    torch.save(policy.state_dict(), os.path.join(save_dir, 'final_policy.pt'))
    print(f"\nTraining complete. {total_steps:,} total steps in {time.time()-start_time:.0f}s")


def test_env():
    """Quick environment test."""
    print("Testing PoolEnv...")
    env = PoolEnv(target_score=3, curriculum_balls=3)
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")

    total_reward = 0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info.get('pocketed'):
            print(f"  Step {i}: pocketed {info['pocketed']}, reward={reward:.2f}")
        if info.get('foul'):
            print(f"  Step {i}: FOUL {info['foul']}, reward={reward:.2f}")
        if terminated or truncated:
            print(f"  Game ended. Score: {info.get('score', '?')}")
            break
    print(f"  Total reward: {total_reward:.2f}")

    if HAS_TORCH:
        print("\nTesting TransformerActorCritic...")
        policy = TransformerActorCritic()
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = policy.get_action(obs_t)
        print(f"  Action: {action.squeeze().numpy()}")
        print(f"  Value: {value.item():.3f}")
        print(f"  Log prob: {log_prob.item():.3f}")
        num_params = sum(p.numel() for p in policy.parameters())
        print(f"  Parameters: {num_params:,}")

    print("All tests passed!")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_env()
    else:
        if HAS_TORCH:
            train()
        else:
            test_env()
