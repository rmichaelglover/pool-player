"""
Phase 4: 2-ball sequence. Pocket ball 1, then pocket ball 2 from the cue
ball's new position. Introduces continuous action (aim, force, spin) and
multi-step episodes — the first step where where-the-cue-ball-ends-up matters.

Reward (sparse):
  +10 per object ball pocketed
  Episode ends: target ball missed, cue scratched, or both balls pocketed

Action (act_dim=4):
  [0:2] aim_sin, aim_cos -> aim_angle via atan2
  [2]   force_raw -> sigmoid -> [FORCE_LO, FORCE_HI]
  [3]   spin_raw  -> tanh    -> [-SPIN_MAX, +SPIN_MAX]

Observation (38-dim, shared layout with Phase 3):
  [0:2]   cue pos (normalized)
  [2:4]   current target ball (normalized)  — Phase-3-compatible slot
  [4:6]   next ball  (or -1/-1 if none)
  [6:32]  other balls (all -1 — pocketed marker)
  [32]    balls_remaining / 15
  [33:38] reserved / zero
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from pool_sim import simulate_shot
from train_curriculum import RolloutBuffer
from train_phase3 import POCKETS, ghost_ball, sample_phase3_setup

# Position-quality shaping: asymmetric Gaussian on the best-pocket cut angle
# at the next ball. Peak at 32.5°, gentler σ below (ramp up from straight
# shots), steeper σ above (fast drop for steep / unpocketable cuts).
POSITION_PEAK_DEG = 32.5
POSITION_SIGMA_LO = 25.0
POSITION_SIGMA_HI = 15.0


def best_pocket_cut_angle_deg(cue, ball):
    """Minimum cut angle (degrees, in [0, 90]) across the 6 pockets from
    the cue's position to `ball`. Ignores pockets "behind" the ball relative
    to the cue (cut > 90°); returns 90° if no pocket is reachable."""
    cx, cy = cue
    bx, by = ball
    cb_dx, cb_dy = bx - cx, by - cy
    cb_mag = math.hypot(cb_dx, cb_dy)
    if cb_mag < 1e-6:
        return 90.0
    cb_dx /= cb_mag
    cb_dy /= cb_mag
    min_cut = 90.0
    for p in POCKETS:
        bp_dx, bp_dy = p[0] - bx, p[1] - by
        bp_mag = math.hypot(bp_dx, bp_dy)
        if bp_mag < 1e-6:
            continue
        bp_dx /= bp_mag
        bp_dy /= bp_mag
        dot = cb_dx * bp_dx + cb_dy * bp_dy
        dot = max(-1.0, min(1.0, dot))
        cut = math.degrees(math.acos(dot))
        if cut < 90.0 and cut < min_cut:
            min_cut = cut
    return min_cut


def position_bonus_factor(cut_deg):
    """Asymmetric Gaussian peaked at POSITION_PEAK_DEG. Returns value in (0, 1]."""
    d = cut_deg - POSITION_PEAK_DEG
    sigma = POSITION_SIGMA_LO if d < 0 else POSITION_SIGMA_HI
    return math.exp(-(d / sigma) ** 2)

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
R = 1.125

FORCE_LO = 50.0
FORCE_HI = 250.0
SPIN_MAX = 2.0
ACT_DIM = 4


def decode_action(raw):
    """Convert raw network output (4,) -> (aim_angle, force, spin_factor)."""
    aim_angle = math.atan2(float(raw[0]), float(raw[1]))
    force = FORCE_LO + (FORCE_HI - FORCE_LO) / (1.0 + math.exp(-float(raw[2])))
    spin = SPIN_MAX * math.tanh(float(raw[3]))
    return aim_angle, force, spin


def sample_phase4_setup():
    """Place 2 balls and cue. Ball 1 gets a Phase-3-style setup (guaranteed
    plausible first shot). Ball 2 is scattered elsewhere on the table.

    Returns: cue [x,y], {1: [x,y], 2: [x,y]}
    """
    cue, ball1, _pocket = sample_phase3_setup(max_cut_deg=45.0)
    # Place ball 2 anywhere on the table clear of cue and ball1.
    for _ in range(40):
        bx = R * 3 + random.random() * (TABLE_LENGTH - R * 6)
        by = R * 3 + random.random() * (TABLE_WIDTH - R * 6)
        if math.hypot(bx - cue[0], by - cue[1]) < 6 * R:
            continue
        if math.hypot(bx - ball1[0], by - ball1[1]) < 6 * R:
            continue
        return cue, {1: list(ball1), 2: [bx, by]}
    # Fallback: just place at table center if we can't find clear spot.
    return cue, {1: list(ball1), 2: [TABLE_LENGTH / 2, TABLE_WIDTH / 2]}


class Phase4Env:
    """2-ball sequence env. Up to 2 shots per episode."""

    def __init__(self, pocket_reward=10.0, position_bonus_weight=0.0):
        self.pocket_reward = pocket_reward
        self.position_bonus_weight = position_bonus_weight
        self.reset()

    def reset(self):
        self.cue, self.balls = sample_phase4_setup()
        # Target order: always ball 1 first, then ball 2.
        self.shot_order = [1, 2]
        self.shot_idx = 0
        self.done = False
        return self.get_obs()

    @property
    def target_ball(self):
        if self.shot_idx >= len(self.shot_order):
            return None
        return self.shot_order[self.shot_idx]

    def get_obs(self):
        obs = np.full(38, -1.0, dtype=np.float32)
        obs[0] = self.cue[0] / TABLE_LENGTH
        obs[1] = self.cue[1] / TABLE_WIDTH
        # Slot [2:4]: current target ball (Phase-3 compatible).
        tgt = self.target_ball
        if tgt is not None and tgt in self.balls:
            obs[2] = self.balls[tgt][0] / TABLE_LENGTH
            obs[3] = self.balls[tgt][1] / TABLE_WIDTH
        # Slot [4:6]: next ball (the one after target).
        next_idx = self.shot_idx + 1
        if next_idx < len(self.shot_order):
            nxt = self.shot_order[next_idx]
            if nxt in self.balls:
                obs[4] = self.balls[nxt][0] / TABLE_LENGTH
                obs[5] = self.balls[nxt][1] / TABLE_WIDTH
        # Slot [32]: balls remaining / 15.
        obs[32] = len(self.balls) / 15.0
        return obs

    def step(self, aim_angle, force, spin_factor):
        if self.done or self.target_ball is None:
            # Shouldn't happen, but guard.
            return self.get_obs(), 0.0, True, {'pocketed_target': False}

        aim_dx = math.cos(aim_angle)
        aim_dy = math.sin(aim_angle)
        balls_in_sim = {bid: tuple(pos) for bid, pos in self.balls.items()}
        result = simulate_shot(
            tuple(self.cue), balls_in_sim,
            aim_dx * force, aim_dy * force,
            spin_factor, aim_dx, aim_dy,
        )

        target = self.target_ball
        target_pocketed = target in result.pocketed_ids
        scratch = result.cue_scratched

        reward = 0.0
        if scratch:
            # Cue scratched → episode ends with 0 reward regardless of
            # what the object did. (Could have been a positive shot
            # otherwise but scratch overrides in real pool too.)
            self.done = True
            return self.get_obs(), 0.0, True, {
                'pocketed_target': False, 'scratch': True,
                'shot': self.shot_idx + 1,
            }

        if target_pocketed:
            reward += self.pocket_reward
            # Remove pocketed balls (both target + any other incidentally pocketed).
            for bid in list(result.pocketed_ids):
                if bid in self.balls:
                    del self.balls[bid]
            # Update cue position from sim.
            if 0 in result.final_positions:
                self.cue = list(result.final_positions[0])
            # Also update any remaining object ball positions.
            for bid, pos in result.final_positions.items():
                if bid in self.balls:
                    self.balls[bid] = list(pos)
            # Advance to next shot.
            self.shot_idx += 1
            # Position-quality bonus: on a successful shot with a remaining
            # target ahead, shape the reward by the best-pocket cut angle at
            # that next ball. Encourages leaving a playable cut angle
            # (peak 32.5°) rather than straight or steep shots.
            if (self.position_bonus_weight > 0
                    and self.shot_idx < len(self.shot_order)
                    and self.shot_order[self.shot_idx] in self.balls):
                next_ball = self.balls[self.shot_order[self.shot_idx]]
                cut = best_pocket_cut_angle_deg(self.cue, next_ball)
                reward += self.position_bonus_weight * position_bonus_factor(cut)
            if self.shot_idx >= len(self.shot_order) or not self.balls:
                self.done = True
        else:
            # Target ball not pocketed → episode ends.
            self.done = True

        return self.get_obs(), reward, self.done, {
            'pocketed_target': target_pocketed,
            'scratch': False,
            'shot': self.shot_idx,
        }


class VecPhase4:
    """Vectorized wrapper. Unlike bandit envs, we don't auto-reset each step —
    only after done=True. Observations passed back reflect the current state
    of each env (post-reset if it terminated)."""

    def __init__(self, num_envs, pocket_reward=10.0, position_bonus_weight=0.0):
        self.num_envs = num_envs
        self.envs = [Phase4Env(pocket_reward=pocket_reward,
                               position_bonus_weight=position_bonus_weight)
                     for _ in range(num_envs)]

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, raw_actions):
        obs = np.zeros((self.num_envs, 38), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        shot1_attempted = shot1_pocketed = 0
        shot2_attempted = shot2_pocketed = 0
        episodes_finished = 0
        episodes_perfect = 0
        for i, (env, raw) in enumerate(zip(self.envs, raw_actions)):
            shot_num_before = env.shot_idx + 1  # 1 or 2
            aim, force, spin = decode_action(raw)
            next_obs, r, d, info = env.step(aim, force, spin)
            rewards[i] = r
            dones[i] = d
            # Track per-shot stats BEFORE the env resets.
            if shot_num_before == 1:
                shot1_attempted += 1
                if info.get('pocketed_target'):
                    shot1_pocketed += 1
            elif shot_num_before == 2:
                shot2_attempted += 1
                if info.get('pocketed_target'):
                    shot2_pocketed += 1
            if d:
                episodes_finished += 1
                if not env.balls:  # both pocketed
                    episodes_perfect += 1
                next_obs = env.reset()
            obs[i] = next_obs
        return obs, rewards, dones, {
            'shot1_attempted': shot1_attempted,
            'shot1_pocketed': shot1_pocketed,
            'shot2_attempted': shot2_attempted,
            'shot2_pocketed': shot2_pocketed,
            'episodes_finished': episodes_finished,
            'episodes_perfect': episodes_perfect,
        }


def train_phase4(num_envs=32, device_name='cpu', max_iters=500,
                 tag='smoke', lr=1e-4, steps_per_update=64,
                 pocket_reward=10.0, log_std_min=-3.0,
                 entropy_coef=0.01, warm_start=None,
                 position_bonus_weight=0.0,
                 embed_dim=96, num_heads=6, num_layers=4, ff_dim=None):
    device = torch.device(device_name)
    if ff_dim is None:
        ff_dim = embed_dim * 2
    net = PoolAttentionNet(
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        ff_dim=ff_dim, act_dim=ACT_DIM,
    ).to(device)
    net.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5).to(device))

    if warm_start and os.path.exists(warm_start):
        # Best-effort partial load — shape-matched tensors copy directly.
        # Additionally: if the source is a Phase 3 (act_dim=2) checkpoint,
        # copy its actor-final-layer aim weights into rows [0:2] of our
        # (act_dim=4) actor final layer, and its log_std into slots [0:2].
        # This preserves learned aim mapping; force/spin rows stay at the
        # fresh small-init (orthogonal gain 0.01) → near-zero initial output.
        src = torch.load(warm_start, map_location=device, weights_only=True)
        dst = net.state_dict()
        loaded = 0
        aim_head_copied = False
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k] = v
                loaded += 1
        # Targeted copy for shape-mismatched aim-output tensors.
        # log_std is deliberately NOT copied: Phase 3's clamped -3 would
        # suppress aim exploration, but the trunk needs to re-adapt to the
        # 2-ball observation and needs aim-gradient signal to do so.
        for k in ['actor.4.weight', 'actor.4.bias']:
            if k in src and k in dst:
                src_v = src[k]
                dst_v = dst[k].clone()
                n = min(src_v.shape[0], dst_v.shape[0])
                if dst_v.shape[0] > src_v.shape[0]:
                    # Phase 3 shape smaller: copy aim dims into front slots.
                    dst_v[:n] = src_v[:n]
                    dst[k] = dst_v
                    aim_head_copied = True
                    loaded += 1
        net.load_state_dict(dst)
        aim_note = ' (+aim head copied)' if aim_head_copied else ''
        print(f'Warm-started {loaded}/{len(src)} tensors from {warm_start}{aim_note}', flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Phase 4: 2-ball sequence. PoolAttentionNet {n_params:,} params on {device}', flush=True)
    print(f'config: tag={tag} lr={lr} steps={steps_per_update} envs={num_envs} '
          f'ent={entropy_coef} pocket_r={pocket_reward} '
          f'pos_bonus_w={position_bonus_weight}', flush=True)

    env = VecPhase4(num_envs, pocket_reward=pocket_reward,
                    position_bonus_weight=position_bonus_weight)
    obs = env.reset()

    batch_size = min(512, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    value_coef = 0.5
    buffer = RolloutBuffer(num_envs, steps_per_update, obs_dim=38, act_dim=ACT_DIM)

    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    best_avg_perfect = 0.0
    recent = {'shot1': [], 'shot2': [], 'perfect': []}

    for iteration in range(max_iters):
        buffer.ptr = 0
        tot_shot1_att = tot_shot1_pkt = 0
        tot_shot2_att = tot_shot2_pkt = 0
        tot_eps_fin = tot_eps_perf = 0

        for step in range(steps_per_update):
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                actions, log_probs, values = net.get_action(obs_t)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, info = env.step(actions_np)
            buffer.add(obs, actions_np, rewards, dones.astype(np.float32),
                       log_probs.cpu().numpy(), values.cpu().numpy())
            obs = next_obs
            tot_shot1_att += info['shot1_attempted']
            tot_shot1_pkt += info['shot1_pocketed']
            tot_shot2_att += info['shot2_attempted']
            tot_shot2_pkt += info['shot2_pocketed']
            tot_eps_fin += info['episodes_finished']
            tot_eps_perf += info['episodes_perfect']

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

        shot1_rate = tot_shot1_pkt / max(1, tot_shot1_att)
        shot2_rate = tot_shot2_pkt / max(1, tot_shot2_att)
        perfect_rate = tot_eps_perf / max(1, tot_eps_fin)
        recent['shot1'].append(shot1_rate)
        recent['shot2'].append(shot2_rate)
        recent['perfect'].append(perfect_rate)

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            avg1 = np.mean(recent['shot1'][-50:])
            avg2 = np.mean(recent['shot2'][-50:])
            avgp = np.mean(recent['perfect'][-50:])
            print(f'Iter {iteration+1:5d} | '
                  f'Shot1={shot1_rate:.1%} AvgShot1={avg1:.1%} | '
                  f'Shot2={shot2_rate:.1%} AvgShot2={avg2:.1%} | '
                  f'Perfect={perfect_rate:.1%} AvgPerfect={avgp:.1%} | '
                  f'eps_fin={tot_eps_fin} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)
            if avgp > best_avg_perfect:
                best_avg_perfect = avgp
                torch.save(net.state_dict(), f'checkpoints/phase4_{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f'checkpoints/phase4_{tag}_latest.pt')

    print(f'Done. Best avg perfect rate: {best_avg_perfect:.1%} in {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--tag', default='smoke')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps_per_update', type=int, default=64)
    parser.add_argument('--pocket_reward', type=float, default=10.0)
    parser.add_argument('--log_std_min', type=float, default=-3.0)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--warm', default=None)
    parser.add_argument('--position_bonus_weight', type=float, default=0.0,
                        help='β for position-quality bonus on successful '
                             'shot with a remaining target ahead. 0 = sparse.')
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=None,
                        help='Transformer FFN hidden dim. Defaults to 2*embed_dim.')
    args = parser.parse_args()
    train_phase4(
        num_envs=args.envs, device_name=args.device,
        max_iters=args.iters, tag=args.tag,
        lr=args.lr, steps_per_update=args.steps_per_update,
        pocket_reward=args.pocket_reward,
        log_std_min=args.log_std_min,
        entropy_coef=args.entropy_coef,
        warm_start=args.warm,
        position_bonus_weight=args.position_bonus_weight,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, ff_dim=args.ff_dim,
    )
