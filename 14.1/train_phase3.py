"""
Phase 3: Pocket one ball with cut angles. The cue is NOT on the line through
the ball and pocket — the policy must learn ghost-ball aiming (aim through
the point on the ball opposite the pocket, not at the ball center).

Geometry:
  - Random pocket P chosen.
  - Ball B placed 8-30 inches from P, not too close to rails.
  - Cue C placed 10-30 inches from B at a random cut angle up to ~60 deg
    off the B->P line.

Reward (same shape as Phase 2 but shaping uses ghost-ball angle):
  +10  pocketed
  +1 + proximity bonus when hit but not pocketed
  aim-quality shaping on full miss, against the ghost-ball angle

Still bandit framing (done=True each shot). act_dim=2 (aim-only), force=30
fixed. If Phase 3 stalls we'll promote to force+spin control.
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
from train_curriculum import RolloutBuffer

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
R = 1.125

# Pocket detection centers MUST match the C sim (pool_sim.c PX/PY).
# Physics checks ball-center-to-pocket-center distance against radius 2.5
# (corner) or 2.75 (side). Using offset visual positions (as the
# train_phase2.POCKETS constant does) breaks ghost-ball geometry at non-zero
# cut angles.
POCKETS = [
    (0.0,                 0.0),            # TL corner
    (TABLE_LENGTH / 2,    0.0),            # T side
    (TABLE_LENGTH,        0.0),            # TR corner
    (0.0,                 TABLE_WIDTH),    # BL corner
    (TABLE_LENGTH / 2,    TABLE_WIDTH),    # B side
    (TABLE_LENGTH,        TABLE_WIDTH),    # BR corner
]


def ghost_ball(ball, pocket):
    """Point the cue ball center should occupy at contact to send ball->pocket."""
    bx, by = ball
    px, py = pocket
    dx = px - bx
    dy = py - by
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return ball  # degenerate
    return (bx - 2 * R * dx / d, by - 2 * R * dy / d)


def _line_point_dist(p0, v, q):
    """Closest distance from point q to the ray starting at p0 with unit dir v.
    Returns (perp_distance, signed_t_along_v).  t<0 means q is behind p0."""
    qx = q[0] - p0[0]
    qy = q[1] - p0[1]
    t = qx * v[0] + qy * v[1]
    perp_x = qx - t * v[0]
    perp_y = qy - t * v[1]
    return math.hypot(perp_x, perp_y), t


def cue_traj_closest_to_best_ghost(cue, ball, aim_angle):
    """
    Minimum forward closest-approach distance from cue trajectory to the
    ghost-ball position of any of the 6 pockets. Ignores ghosts that are
    behind the cue along the aim direction.
    """
    v = (math.cos(aim_angle), math.sin(aim_angle))
    best = float('inf')
    for p in POCKETS:
        g = ghost_ball(ball, p)
        d, t = _line_point_dist(cue, v, g)
        if t > 0:
            best = min(best, d)
    return best


def object_ball_exit_direction(cue, ball, aim_angle):
    """
    Analytic object-ball exit direction after ideal ball-ball contact.
    Returns exit_angle or None if the aim geometrically misses the ball.
    """
    v = (math.cos(aim_angle), math.sin(aim_angle))
    bx = ball[0] - cue[0]
    by = ball[1] - cue[1]
    proj = bx * v[0] + by * v[1]
    closest_sq = max(0.0, bx * bx + by * by - proj * proj)
    if closest_sq > (2 * R) ** 2 or proj <= 0:
        return None
    t_contact = proj - math.sqrt((2 * R) ** 2 - closest_sq)
    cx = cue[0] + t_contact * v[0]
    cy = cue[1] + t_contact * v[1]
    ex = ball[0] - cx
    ey = ball[1] - cy
    n = math.hypot(ex, ey)
    return math.atan2(ey / n, ex / n)


def obj_traj_closest_to_best_pocket(ball, exit_angle):
    """
    For an object ball leaving `ball` in direction `exit_angle`, return the
    minimum perpendicular distance to any of the 6 pockets, counting only
    pockets ahead of the ball.
    """
    v = (math.cos(exit_angle), math.sin(exit_angle))
    best = float('inf')
    for p in POCKETS:
        d, t = _line_point_dist(ball, v, p)
        if t > 0:
            best = min(best, d)
    return best


def sample_phase3_setup(max_cut_deg=60.0):
    """
    Place ball, pick a pocket, then place cue at a random cut angle.
    Enforce that the cue has line-of-sight to the ghost ball (trivially true
    with one ball). Clamp into the table interior.
    """
    for _ in range(40):
        pocket = random.choice(POCKETS)
        # Ball placed 4-15 inches from the pocket. Larger distances exhaust
        # object-ball momentum at big cut angles with our fixed force=60.
        ball_to_pocket = 4 + random.random() * 11
        base_angle = math.atan2(
            TABLE_WIDTH / 2 - pocket[1], TABLE_LENGTH / 2 - pocket[0]
        )
        jitter = (random.random() - 0.5) * math.radians(80)
        angle = base_angle + jitter
        bx = pocket[0] + math.cos(angle) * ball_to_pocket
        by = pocket[1] + math.sin(angle) * ball_to_pocket
        if not (R * 3 < bx < TABLE_LENGTH - R * 3): continue
        if not (R * 3 < by < TABLE_WIDTH - R * 3): continue

        # Ghost ball position (where the cue must be at contact)
        g = ghost_ball((bx, by), pocket)

        # Cut angle: random in [-max_cut, max_cut]. Direction from ghost
        # toward the cue, defined by rotating the ghost->pocket direction.
        bp_angle = math.atan2(pocket[1] - by, pocket[0] - bx)
        cut = math.radians((random.random() * 2 - 1) * max_cut_deg)
        aim_away = bp_angle + math.pi + cut  # cue sits opposite pocket + cut
        cue_dist = 10 + random.random() * 20
        cue_x = g[0] + math.cos(aim_away) * cue_dist
        cue_y = g[1] + math.sin(aim_away) * cue_dist
        if not (R * 3 < cue_x < TABLE_LENGTH - R * 3): continue
        if not (R * 3 < cue_y < TABLE_WIDTH - R * 3): continue
        # Ensure cue not overlapping ball
        if math.hypot(cue_x - bx, cue_y - by) < 4 * R: continue
        return [cue_x, cue_y], [bx, by], pocket

    # Fallback: Phase 2-style straight shot if the sampler struggles.
    pocket = random.choice(POCKETS)
    bx = TABLE_LENGTH / 2
    by = TABLE_WIDTH / 2
    g = ghost_ball((bx, by), pocket)
    return [g[0] - 10, g[1]], [bx, by], pocket


class Phase3Env:
    def __init__(self, max_cut_deg=60.0, pocket_reward=10.0,
                 proximity_reward=1.5, miss_shape='linear',
                 gauss_sigma_deg=2.0, approach_sigma_in=0.3,
                 hit_shape='final_pos', hit_sigma_in=2.5):
        self.max_cut_deg = max_cut_deg
        self.pocket_reward = pocket_reward
        self.proximity_reward = proximity_reward
        self.miss_shape = miss_shape    # 'linear' | 'gauss' | 'approach'
        self.gauss_sigma = math.radians(gauss_sigma_deg)
        self.approach_sigma = approach_sigma_in
        self.hit_shape = hit_shape      # 'final_pos' | 'straight_line'
        self.hit_sigma = hit_sigma_in
        self.reset()

    def reset(self):
        self.cue, self.ball_pos, self.target_pocket = sample_phase3_setup(self.max_cut_deg)
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

    def step(self, aim_angle, force=60.0):
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
            reward += self.pocket_reward
        elif hit:
            reward += 1.0
            if self.hit_shape == 'straight_line':
                # Reward based on the analytic straight-line trajectory the
                # object ball takes after contact. Ignores rail bounces.
                exit_angle = object_ball_exit_direction(
                    self.cue, self.ball_pos, aim_angle)
                if exit_angle is not None:
                    d = obj_traj_closest_to_best_pocket(
                        self.ball_pos, exit_angle)
                    if d != float('inf'):
                        # Gaussian, width = hit_sigma (default 2.5 = pocket radius)
                        reward += self.proximity_reward * math.exp(
                            -(d / self.hit_sigma) ** 2)
            else:
                # Legacy final-position-to-nearest-pocket proximity bonus
                if 1 in result.final_positions:
                    fx, fy = result.final_positions[1]
                    min_pocket_dist = min(
                        math.hypot(fx - p[0], fy - p[1]) for p in POCKETS
                    )
                    reward += max(0.0, self.proximity_reward * (1 - min_pocket_dist / 10))
        else:
            if self.miss_shape == 'approach':
                # Cue-trajectory closest-approach to best-pocket ghost.
                # Distance metric; reward shaped as a Gaussian bump (0 at
                # large distances, +1 at 0). Uses pre-shot cue position.
                d = cue_traj_closest_to_best_ghost(
                    self.cue, self.ball_pos, aim_angle)
                if d == float('inf'):
                    reward += -0.5
                else:
                    reward += math.exp(-(d / self.approach_sigma) ** 2) - 0.5
            elif self.miss_shape == 'gauss':
                g = ghost_ball(self.ball_pos, self.target_pocket)
                gdx = g[0] - self.cue[0]
                gdy = g[1] - self.cue[1]
                ghost_angle = math.atan2(gdy, gdx)
                angle_diff = abs(aim_angle - ghost_angle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                reward += math.exp(-(angle_diff / self.gauss_sigma) ** 2) - 0.5
            else:
                g = ghost_ball(self.ball_pos, self.target_pocket)
                gdx = g[0] - self.cue[0]
                gdy = g[1] - self.cue[1]
                ghost_angle = math.atan2(gdy, gdx)
                angle_diff = abs(aim_angle - ghost_angle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                aim_quality = max(0, 1.0 - angle_diff / math.pi)
                reward += aim_quality * 0.5 - 0.5

        return reward, True, {'hit': hit, 'pocketed': pocketed}


class VecPhase3:
    def __init__(self, num_envs, max_cut_deg=60.0, **env_kwargs):
        self.num_envs = num_envs
        self.envs = [Phase3Env(max_cut_deg=max_cut_deg, **env_kwargs)
                     for _ in range(num_envs)]

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs = np.zeros((self.num_envs, 38), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.ones(self.num_envs, dtype=bool)
        hits = pockets = 0
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            aim_angle = np.arctan2(float(act[0]), float(act[1]))
            r, d, info = env.step(aim_angle)
            rewards[i] = r
            if info['hit']: hits += 1
            if info['pocketed']: pockets += 1
            obs[i] = env.reset()
        return obs, rewards, dones, {'hits': hits, 'pockets': pockets}


def train_phase3(num_envs=32, device_name='cpu', max_iters=2000,
                 warm_start='checkpoints/phase2_best.pt',
                 max_cut_deg=60.0, tag='v2',
                 log_std_min=-3.0, entropy_coef=0.01, lr=3e-4,
                 steps_per_update=32, pocket_reward=10.0,
                 proximity_reward=1.5, miss_shape='linear',
                 gauss_sigma_deg=2.0, approach_sigma_in=0.3,
                 hit_shape='final_pos', hit_sigma_in=2.5):
    device = torch.device(device_name)
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2).to(device)
    net.log_std = nn.Parameter(torch.full((2,), -0.5).to(device))

    if warm_start and os.path.exists(warm_start):
        state = torch.load(warm_start, map_location=device, weights_only=True)
        net.load_state_dict(state)
        print(f'Warm-started from {warm_start}', flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Phase 3: Pocket a ball with cut angles', flush=True)
    print(f'PoolAttentionNet: {n_params:,} params on {device}', flush=True)

    env = VecPhase3(num_envs, max_cut_deg=max_cut_deg,
                    pocket_reward=pocket_reward,
                    proximity_reward=proximity_reward,
                    miss_shape=miss_shape, gauss_sigma_deg=gauss_sigma_deg,
                    approach_sigma_in=approach_sigma_in,
                    hit_shape=hit_shape, hit_sigma_in=hit_sigma_in)
    obs = env.reset()
    print(f'config: tag={tag} cut={max_cut_deg} log_std_min={log_std_min} '
          f'ent={entropy_coef} lr={lr} steps={steps_per_update} envs={num_envs} '
          f'pocket_r={pocket_reward} miss={miss_shape} hit={hit_shape}', flush=True)

    batch_size = min(512, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    value_coef = 0.5

    buffer = RolloutBuffer(num_envs, steps_per_update, act_dim=2)
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    best_pocket = 0.0
    hit_rates = []
    pocket_rates = []

    for iteration in range(max_iters):
        buffer.ptr = 0
        total_hits = total_pockets = total_shots = 0

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
            print(f'Iter {iteration+1:5d} | HR={hr:.1%} AvgHR={avg_hr:.1%} '
                  f'Pocket={pr:.1%} AvgPocket={avg_pr:.1%} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)
            if avg_pr > best_pocket:
                best_pocket = avg_pr
                torch.save(net.state_dict(), f'checkpoints/phase3{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f'checkpoints/phase3{tag}_latest.pt')

    print(f'Done. Best avg pocket rate: {best_pocket:.1%} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--warm', default='checkpoints/phase2_best.pt')
    parser.add_argument('--cut', type=float, default=60.0)
    parser.add_argument('--tag', default='a')
    parser.add_argument('--log_std_min', type=float, default=-3.0)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--steps_per_update', type=int, default=32)
    parser.add_argument('--pocket_reward', type=float, default=10.0)
    parser.add_argument('--proximity_reward', type=float, default=1.5)
    parser.add_argument('--miss_shape', default='linear',
                        choices=['linear', 'gauss', 'approach'])
    parser.add_argument('--gauss_sigma_deg', type=float, default=2.0)
    parser.add_argument('--approach_sigma_in', type=float, default=0.3)
    parser.add_argument('--hit_shape', default='final_pos',
                        choices=['final_pos', 'straight_line'])
    parser.add_argument('--hit_sigma_in', type=float, default=2.5)
    args = parser.parse_args()
    warm = args.warm if args.warm else None
    train_phase3(num_envs=args.envs, device_name=args.device,
                 max_iters=args.iters, warm_start=warm,
                 max_cut_deg=args.cut, tag=args.tag,
                 log_std_min=args.log_std_min,
                 entropy_coef=args.entropy_coef, lr=args.lr,
                 steps_per_update=args.steps_per_update,
                 pocket_reward=args.pocket_reward,
                 proximity_reward=args.proximity_reward,
                 miss_shape=args.miss_shape,
                 gauss_sigma_deg=args.gauss_sigma_deg,
                 approach_sigma_in=args.approach_sigma_in,
                 hit_shape=args.hit_shape,
                 hit_sigma_in=args.hit_sigma_in)
