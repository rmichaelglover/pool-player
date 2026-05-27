"""
Phase 6b: Full 14.1 continuous with rerack. Multi-rack runs.

Rules:
  - Start with a 15-ball rack.
  - Pocket balls until only 1 object ball remains → rerack:
      • 14 balls placed back in positions RACK_POSITIONS[1:] (apex empty, per 14.1 rules)
      • Remaining ball stays where it is, unless it falls in the racking area (then → head spot)
      • Cue ball stays where it is
  - Episode ends on: miss, scratch, or max_shots reached
  - Reward = +10 per ball pocketed per shot
  - Score: total run length across all racks

Observation layout identical to Phase 6 (16-ball capacity, all 15 slots used).
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
from shot_utils import (first_ball_struck, called_pocket_index, pocket_index_of,
                        HEAD_SPOT, HEAD_SPOT_ALT, rerack_positions,
                        relocate_break_ball)


class Phase6bEnv:
    """14.1 continuous env with rerack and call-shot rule.

    Call-shot: the "called ball" is the first ball the cue's aim line would
    strike. If the called ball is pocketed on the shot, reward = pocket_reward
    × number of balls pocketed (including incidentals — matches 14.1 where
    called-ball success makes all balls count). If the called ball misses,
    episode ends with no reward, regardless of incidental pockets.

    Set call_shot=False to revert to "any ball any pocket" (spray-friendly)
    rule — kept for comparison / ablation.
    """

    def __init__(self, pocket_reward=10.0, max_shots=50, call_shot=True,
                  lenient_break=False):
        self.pocket_reward = pocket_reward
        self.max_shots = max_shots
        self.call_shot = call_shot
        # When True, a break shot that pockets nothing (without scratching)
        # does NOT end the episode — the cue+balls keep their post-shot
        # positions and the next shot is treated as a regular (non-break)
        # shot. Used by the demo so a missed break doesn't force a restart.
        # Default off (training behavior unchanged).
        self.lenient_break = lenient_break
        self.reset()

    def reset(self):
        self.cue, self.balls = sample_phase6_setup()
        self.shot_idx = 0
        self.done = False
        self.rerack_count = 0
        self.total_pocketed = 0
        # Break shot: first shot of a rack (intact cluster). Under 14.1 rules
        # the opening break is NOT a called shot — you can't cleanly pocket
        # a ball that's touching others in the rack. Track this so we can
        # bypass call-shot for the break and then enforce it for subsequent shots.
        self.is_break_shot = True
        return self.get_obs()

    def get_obs(self):
        obs = np.full(38, -1.0, dtype=np.float32)
        obs[0] = self.cue[0] / TABLE_LENGTH
        obs[1] = self.cue[1] / TABLE_WIDTH
        # Emit up to 15 object balls into slots [2:32] in the order they appear
        # in self.balls (dict insertion order). Pocketed balls are absent → slot stays -1.
        slot = 2
        for bid in sorted(self.balls.keys()):
            if slot >= 32: break
            obs[slot] = self.balls[bid][0] / TABLE_LENGTH
            obs[slot + 1] = self.balls[bid][1] / TABLE_WIDTH
            slot += 2
        obs[32] = len(self.balls) / 15.0
        obs[33] = self.rerack_count / 3.0  # rerack count hint (normalized)
        return obs

    def _do_rerack(self):
        """Invoked after a successful shot leaves exactly 1 object ball remaining.
        Places 14 new balls in rack positions (apex empty), relocates break ball
        if needed."""
        remaining_bid = next(iter(self.balls.keys()))
        remaining_pos = list(self.balls[remaining_bid])
        remaining_pos = relocate_break_ball(remaining_pos, self.cue)
        self.balls = {remaining_bid: remaining_pos}
        # Assign 14 new balls (reuse available IDs 1-15 excluding remaining_bid)
        positions = rerack_positions()
        available_ids = [i for i in range(1, 16) if i != remaining_bid]
        for bid, pos in zip(available_ids, positions):
            self.balls[bid] = list(pos)
        self.rerack_count += 1
        # Next shot is a break into the newly-racked cluster.
        self.is_break_shot = True

    def step(self, aim_angle, force, spin_factor, record_trajectory=False,
             traj_max_frames=600):
        if self.done:
            return self.get_obs(), 0.0, True, {'pocketed_count': 0}

        # Compute the called ball AND called pocket BEFORE the shot (call-shot rule).
        called_id, _ = first_ball_struck(self.cue, aim_angle, self.balls)
        called_pocket = -1
        if called_id is not None:
            called_pocket = called_pocket_index(
                self.cue, aim_angle, self.balls[called_id]
            )

        aim_dx = math.cos(aim_angle)
        aim_dy = math.sin(aim_angle)
        balls_in_sim = {bid: tuple(pos) for bid, pos in self.balls.items()}
        # Sim input ordering (cue first, then ball IDs ascending) so the
        # caller can map trajectory indices back to ball IDs.
        ordered_ids = [0] + sorted(balls_in_sim.keys())
        result = simulate_shot(
            tuple(self.cue), balls_in_sim,
            aim_dx * force, aim_dy * force,
            spin_factor, aim_dx, aim_dy,
            record_trajectory=record_trajectory,
            traj_max_frames=traj_max_frames,
        )

        scratch = result.cue_scratched
        pocketed_ids = set(result.pocketed_ids)
        pocketed_obj_count = len(pocketed_ids)

        # Determine which pocket the called ball ended in (if it was pocketed).
        called_actual_pocket = -1
        if called_id is not None and called_id in pocketed_ids:
            called_final = result.final_positions.get(called_id)
            if called_final is not None:
                called_actual_pocket = pocket_index_of(called_final)

        called_shot_valid = (
            called_id is not None
            and called_pocket >= 0
            and called_id in pocketed_ids
            and called_actual_pocket == called_pocket
        )

        info = {
            'pocketed_count': pocketed_obj_count,
            'pocketed_ids': list(pocketed_ids),
            'scratch': scratch,
            'hit_ball': result.hit_ball,
            'called_id': called_id,
            'called_pocket': called_pocket,
            'called_actual_pocket': called_actual_pocket,
            'called_shot_valid': called_shot_valid,
            'total_pocketed_so_far': self.total_pocketed,
            'rerack_count': self.rerack_count,
        }
        if record_trajectory and result.trajectory is not None:
            info['trajectory'] = result.trajectory.tolist()
            info['trajectory_ball_ids'] = ordered_ids

        if scratch:
            self.done = True
            return self.get_obs(), 0.0, True, info

        if pocketed_obj_count == 0:
            is_break = getattr(self, 'is_break_shot', False)
            if is_break and self.lenient_break:
                # Lenient-break path (demo only): a break that pockets nothing
                # is treated as a free attempt — apply final positions, advance
                # shot counter, demote to regular-shot mode. Next miss ends as
                # usual.
                if 0 in result.final_positions:
                    self.cue = list(result.final_positions[0])
                for bid, pos in result.final_positions.items():
                    if bid in self.balls:
                        self.balls[bid] = list(pos)
                self.shot_idx += 1
                info['is_break_shot'] = True
                self.is_break_shot = False
                if self.shot_idx >= self.max_shots:
                    self.done = True
                return self.get_obs(), 0.0, self.done, info
            self.done = True
            return self.get_obs(), 0.0, True, info

        # Strict call-shot rule: need (a) aimed-at ball exists, (b) its natural
        # exit direction points at some pocket, (c) that specific ball lands
        # in that specific pocket. Exception: the break shot is exempt — any
        # pocket counts. (14.1 opening break / post-rerack break rule.)
        is_break = getattr(self, 'is_break_shot', False)
        info['is_break_shot'] = is_break
        if self.call_shot and not is_break and not called_shot_valid:
            self.done = True
            return self.get_obs(), 0.0, True, info
        # Break shot with nothing pocketed still ends the episode (already
        # handled above by pocketed_obj_count == 0 check).
        # Clear the break-shot flag — next shot will be regular call-shot.
        self.is_break_shot = False

        # Called ball (or called-off mode) succeeded — apply reward for ALL pocketed balls.
        reward = self.pocket_reward * pocketed_obj_count
        for bid in pocketed_ids:
            if bid in self.balls:
                del self.balls[bid]
        if 0 in result.final_positions:
            self.cue = list(result.final_positions[0])
        for bid, pos in result.final_positions.items():
            if bid in self.balls:
                self.balls[bid] = list(pos)

        self.total_pocketed += pocketed_obj_count
        self.shot_idx += 1

        # Rerack condition: exactly 1 object ball left and we still have shots.
        if len(self.balls) == 1 and self.shot_idx < self.max_shots:
            self._do_rerack()
        elif len(self.balls) == 0:
            # Accidentally pocketed the last ball too — rerack fresh (no break ball available;
            # treat this as game over by convention).
            self.done = True

        if self.shot_idx >= self.max_shots:
            self.done = True

        info['total_pocketed_so_far'] = self.total_pocketed
        info['rerack_count'] = self.rerack_count

        return self.get_obs(), reward, self.done, info


class VecPhase6b:
    def __init__(self, num_envs, pocket_reward=10.0, max_shots=50, call_shot=True):
        self.num_envs = num_envs
        self.envs = [Phase6bEnv(pocket_reward=pocket_reward, max_shots=max_shots,
                                 call_shot=call_shot)
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
            'run_lengths': [],
            'rerack_counts': [],
            'break_pockets': 0,
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
                stats['run_lengths'].append(info['total_pocketed_so_far'])
                stats['rerack_counts'].append(info['rerack_count'])
                next_obs = env.reset()
            obs[i] = next_obs
        return obs, rewards, dones, stats


def train_phase6b(num_envs=32, device_name='cpu', max_iters=500,
                  tag='p6b_baseline', lr=1e-4, steps_per_update=64,
                  pocket_reward=10.0, log_std_min=-3.0,
                  entropy_coef=0.01, warm_start=None,
                  embed_dim=96, num_heads=6, num_layers=4, ff_dim=None,
                  max_shots=50, call_shot=True):
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
    print(f'Phase 6b: 14.1 continuous (rerack). PoolAttentionNet {n_params:,} params on {device}', flush=True)
    print(f'config: tag={tag} lr={lr} steps={steps_per_update} envs={num_envs} '
          f'ent={entropy_coef} pocket_r={pocket_reward} max_shots={max_shots}', flush=True)

    env = VecPhase6b(num_envs, pocket_reward=pocket_reward, max_shots=max_shots,
                     call_shot=call_shot)
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
        iter_reracks = []
        iter_episodes = 0
        iter_total_pockets = 0

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
            iter_reracks.extend(stats['rerack_counts'])
            iter_episodes += stats['episodes_finished']
            iter_total_pockets += stats['total_pockets']

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
        max_rerack = int(np.max(iter_reracks)) if iter_reracks else 0
        recent_runlens.extend(iter_run_lengths)
        recent_runlens = recent_runlens[-500:]

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            rolling_avg = float(np.mean(recent_runlens)) if recent_runlens else 0.0
            print(f'Iter {iteration+1:5d} | '
                  f'AvgRun={avg_runlen:5.2f} Rolling={rolling_avg:5.2f} MaxRun={max_runlen:3d} '
                  f'MaxRerack={max_rerack} | '
                  f'Eps={iter_episodes} Pkts={iter_total_pockets} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)
            if rolling_avg > best_avg_runlen:
                best_avg_runlen = rolling_avg
                torch.save(net.state_dict(), f'checkpoints/phase6b_{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f'checkpoints/phase6b_{tag}_latest.pt')

    print(f'Done. Best rolling avg run length: {best_avg_runlen:.2f} in {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--tag', default='p6b_baseline')
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
    parser.add_argument('--max_shots', type=int, default=50)
    parser.add_argument('--call_shot', type=int, default=1,
                        help='1 = strict call-shot rule (default); 0 = any ball any pocket')
    args = parser.parse_args()
    train_phase6b(
        num_envs=args.envs, device_name=args.device,
        max_iters=args.iters, tag=args.tag,
        lr=args.lr, steps_per_update=args.steps_per_update,
        pocket_reward=args.pocket_reward,
        log_std_min=args.log_std_min,
        entropy_coef=args.entropy_coef,
        warm_start=args.warm,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, ff_dim=args.ff_dim,
        max_shots=args.max_shots,
        call_shot=bool(args.call_shot),
    )
