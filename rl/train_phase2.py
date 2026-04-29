"""
Phase 2: Pocket one ball. Cue and ball placed so a straight shot toward
the ball sends it roughly toward the nearest pocket.

Starts easy (ball close to pocket, cue in line) and stays consistent with
Phase 1's aim-only action space (2 outputs). Phase 3 will introduce
position play and contact offsets.

Reward shape:
  +10.0 pocketed
  +1.0  hit ball (partial credit)
  -0.5 + 0.5 * aim_quality  on full miss (same shaping as Phase 1)
  small bonus if the ball ends up near a pocket (near-miss on pocketing)

Single-shot episode (done=True every step), matching the bandit framing
that fixed Phase 1.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from pool_sim import simulate_shot
from train_curriculum import RolloutBuffer

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
R = 1.125

# Approximate pocket centers on the table (matching pool_sim geometry).
_mo = 2.5 * 0.45
_smo = 2.75 * 0.15
POCKETS = [
    (_mo, _mo),                       # top-left corner
    (TABLE_LENGTH / 2, _smo),         # top-side
    (TABLE_LENGTH - _mo, _mo),        # top-right corner
    (_mo, TABLE_WIDTH - _mo),         # bottom-left corner
    (TABLE_LENGTH / 2, TABLE_WIDTH - _smo),  # bottom-side
    (TABLE_LENGTH - _mo, TABLE_WIDTH - _mo), # bottom-right
]


def sample_phase2_setup():
    """
    Pick a random pocket, place the ball near it, then place the cue
    on the line from ball to pocket, extended backward.

    The straight-line aim "cue -> ball" is a near-perfect pocket shot,
    so the policy only needs Phase 1 aiming to succeed.
    """
    pocket = random.choice(POCKETS)
    # Ball distance from pocket: 2-8 inches
    ball_to_pocket = 2 + random.random() * 6
    # Cue distance behind ball (from cue toward ball -> pocket)
    cue_to_ball = 6 + random.random() * 18  # 6-24 inches

    # Unit vector from pocket to ball (so ball is "in front of" pocket)
    # Pick an angle that keeps cue in the table.
    # Use the angle from pocket toward the table interior with some jitter.
    cx, cy = TABLE_LENGTH / 2, TABLE_WIDTH / 2
    base_angle = math.atan2(cy - pocket[1], cx - pocket[0])
    jitter = (random.random() - 0.5) * math.radians(40)  # +-20 deg
    angle = base_angle + jitter

    bx = pocket[0] + math.cos(angle) * ball_to_pocket
    by = pocket[1] + math.sin(angle) * ball_to_pocket
    cue_x = bx + math.cos(angle) * cue_to_ball
    cue_y = by + math.sin(angle) * cue_to_ball

    # Clamp to table interior, preserve layout (ball between cue and pocket)
    bx = max(R * 2, min(TABLE_LENGTH - R * 2, bx))
    by = max(R * 2, min(TABLE_WIDTH - R * 2, by))
    cue_x = max(R * 2, min(TABLE_LENGTH - R * 2, cue_x))
    cue_y = max(R * 2, min(TABLE_WIDTH - R * 2, cue_y))
    return [cue_x, cue_y], [bx, by], pocket


class Phase2Env:
    """One shot, one ball, geometry favors pocketing."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cue, self.ball_pos, self.target_pocket = sample_phase2_setup()
        return self.get_obs()

    def get_obs(self):
        obs = np.full(38, -1.0, dtype=np.float32)
        obs[0] = self.cue[0] / TABLE_LENGTH
        obs[1] = self.cue[1] / TABLE_WIDTH
        obs[2] = self.ball_pos[0] / TABLE_LENGTH
        obs[3] = self.ball_pos[1] / TABLE_WIDTH
        obs[32] = 0.0
        obs[33] = 0.0
        obs[34] = 1.0 / 15.0
        obs[35] = 0.0
        obs[36] = 0.0
        obs[37] = 0.0
        return obs

    def step(self, aim_angle, force=30.0):
        aim_dx = math.cos(aim_angle)
        aim_dy = math.sin(aim_angle)
        balls = {1: (self.ball_pos[0], self.ball_pos[1])}
        result = simulate_shot(
            tuple(self.cue), balls,
            aim_dx * force, aim_dy * force,
            0, aim_dx, aim_dy,
        )
        hit = result.hit_ball
        pocketed = 1 in result.pocketed_ids

        reward = 0.0
        if pocketed:
            reward += 10.0
        elif hit:
            reward += 1.0
            # Bonus: ball ended near any pocket
            if 1 in result.final_positions:
                fx, fy = result.final_positions[1]
                min_pocket_dist = min(
                    math.hypot(fx - p[0], fy - p[1]) for p in POCKETS
                )
                # Max 1.5 bonus when very close, 0 when > 10 inches
                proximity_bonus = max(0.0, 1.5 * (1 - min_pocket_dist / 10))
                reward += proximity_bonus
        else:
            # Aim-quality shaping same as Phase 1
            dx = self.ball_pos[0] - self.cue[0]
            dy = self.ball_pos[1] - self.cue[1]
            ball_angle = math.atan2(dy, dx)
            angle_diff = abs(aim_angle - ball_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            aim_quality = max(0, 1.0 - angle_diff / math.pi)
            reward += aim_quality * 0.5 - 0.5

        return reward, True, {'hit': hit, 'pocketed': pocketed}


class VecPhase2:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [Phase2Env() for _ in range(num_envs)]

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs = np.zeros((self.num_envs, 38), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.ones(self.num_envs, dtype=bool)  # always done in bandit mode
        hits = 0
        pockets = 0
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            aim_angle = np.arctan2(float(act[0]), float(act[1]))
            r, d, info = env.step(aim_angle)
            rewards[i] = r
            if info['hit']:
                hits += 1
            if info['pocketed']:
                pockets += 1
            obs[i] = env.reset()
        return obs, rewards, dones, {'hits': hits, 'pockets': pockets}


def train_phase2(num_envs=32, device_name='cpu', max_iters=2000,
                 warm_start='checkpoints/phase1_best.pt'):
    device = torch.device(device_name)
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2).to(device)
    net.log_std = nn.Parameter(torch.full((2,), -0.5).to(device))

    if warm_start and os.path.exists(warm_start):
        state = torch.load(warm_start, map_location=device, weights_only=True)
        net.load_state_dict(state)
        print(f'Warm-started from {warm_start}', flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Phase 2: Pocket a ball (warm-start={"yes" if warm_start else "no"})', flush=True)
    print(f'PoolAttentionNet: {n_params:,} params on {device}', flush=True)

    env = VecPhase2(num_envs)
    obs = env.reset()

    steps_per_update = 32
    batch_size = min(256, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    entropy_coef = 0.01  # 10x higher to resist collapse
    value_coef = 0.5
    log_std_min = -1.5  # floor: std >= 0.22, prevents overconfident collapse

    buffer = RolloutBuffer(num_envs, steps_per_update, act_dim=2)
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    best_pocket = 0.0
    hit_rates = []
    pocket_rates = []

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

        hr = total_hits / total_shots
        pr = total_pockets / total_shots
        hit_rates.append(hr)
        pocket_rates.append(pr)

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg_hr = np.mean(hit_rates[-50:])
            avg_pr = np.mean(pocket_rates[-50:])
            print(f'Iter {iteration+1:5d} | '
                  f'HR={hr:.1%} AvgHR={avg_hr:.1%} '
                  f'Pocket={pr:.1%} AvgPocket={avg_pr:.1%} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)

            if avg_pr > best_pocket:
                best_pocket = avg_pr
                torch.save(net.state_dict(), 'checkpoints/phase2_best.pt')

        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), 'checkpoints/phase2_latest.pt')

    print(f'Done. Best pocket rate: {best_pocket:.1%} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--warm', default='checkpoints/phase1_best.pt',
                        help='Warm-start checkpoint path (empty to train from scratch)')
    args = parser.parse_args()
    warm = args.warm if args.warm else None
    train_phase2(num_envs=args.envs, device_name=args.device,
                 max_iters=args.iters, warm_start=warm)
