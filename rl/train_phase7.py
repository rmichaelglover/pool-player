"""
Phase 7: Token-based 14.1 agent. Policy attends over balls + pockets + legal-shot
tokens, picks a shot (categorical) and force/spin for it (continuous).

Env:
  - Starts with full 15-ball rack; opening break is auto-executed in reset().
  - Agent steps in from shot 2, picking from the enumerated legal-shot list.
  - Reward = +10 per object ball pocketed on the shot, IF the called-shot
    (target ball in target pocket) succeeds. Otherwise 0 and episode ends.
  - Rerack when 1 ball remains; break-ball (the remaining ball) becomes the
    opener for the new rack. For post-rerack break, the rack has free space
    at the apex so legal shots usually exist.
  - Max shots is configurable (default 60 to allow a few rack clears).

Action: (shot_idx: int, force_raw: float, spin_raw: float)
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_sim import simulate_shot
from pool_game_net import (PoolGameNet, Phase7Obs, MAX_BALLS, MAX_POCKETS,
                            MAX_SHOTS, FORCE_LO, FORCE_HI, SPIN_MAX,
                            decode_force, decode_spin, TABLE_LENGTH, TABLE_WIDTH)
from shot_enumerator import (generate_legal_shots, POCKETS, POCKET_NAMES,
                              POCKET_RADII, R, LegalShot)
from train_phase6 import RACK_APEX, RACK_POSITIONS, sample_phase6_setup
from train_phase6b import pocket_index_of, HEAD_SPOT, HEAD_SPOT_ALT


# ── Phase 7 env ───────────────────────────────────────────────────────────

class Phase7Env:
    def __init__(self, pocket_reward=10.0, max_shots=60,
                 opening_break_force=240.0, scratch_penalty=-10.0,
                 aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0):
        self.pocket_reward = pocket_reward
        self.max_shots = max_shots
        self.opening_break_force = opening_break_force
        self.scratch_penalty = scratch_penalty
        # Execution-noise parameters: when > 0, every shot's aim/force/spin
        # is perturbed by Gaussian noise before simulation. Models real-world
        # execution variability — the agent must pick robust shots (low cut
        # angle, controllable force) rather than just deterministic-optimal.
        self.aim_noise_deg = aim_noise_deg
        self.force_noise_pct = force_noise_pct
        self.spin_noise = spin_noise
        self.reset()

    def reset(self):
        self.cue, self.balls = sample_phase6_setup()
        self.shot_idx = 0
        self.done = False
        self.rerack_count = 0
        self.total_pocketed = 0
        self.pending_rerack = False
        # Auto-execute opening break so the agent sees a scattered table on its first real decision.
        self._execute_opening_break()
        return self.get_obs()

    def _execute_opening_break(self):
        """Hard-coded opening break: aim at rack apex with high force. Results
        update the env state like a regular shot (but bypasses call-shot).
        Noise also applies here so break shots have realistic variability."""
        dx = RACK_APEX[0] - self.cue[0]
        dy = RACK_APEX[1] - self.cue[1]
        aim = math.atan2(dy, dx)
        force = self.opening_break_force
        if self.aim_noise_deg > 0:
            aim = aim + np.random.randn() * self.aim_noise_deg * (math.pi / 180.0)
        if self.force_noise_pct > 0:
            force = force * (1.0 + np.random.randn() * self.force_noise_pct)
            force = max(20.0, min(280.0, force))
        aim_dx = math.cos(aim); aim_dy = math.sin(aim)
        balls_in_sim = {bid: tuple(pos) for bid, pos in self.balls.items()}
        result = simulate_shot(
            tuple(self.cue), balls_in_sim,
            aim_dx * force, aim_dy * force,
            0.0, aim_dx, aim_dy,
        )
        pocketed_ids = set(result.pocketed_ids)
        if result.cue_scratched:
            # Rare, but handle: start over instead of ending episode.
            self.cue, self.balls = sample_phase6_setup()
            self._execute_opening_break()
            return
        for bid in pocketed_ids:
            if bid in self.balls:
                del self.balls[bid]
        if 0 in result.final_positions:
            self.cue = list(result.final_positions[0])
        for bid, pos in result.final_positions.items():
            if bid in self.balls:
                self.balls[bid] = list(pos)
        self.total_pocketed += len(pocketed_ids)
        self.shot_idx += 1
        if len(self.balls) == 1 and self.shot_idx < self.max_shots:
            self._do_rerack()
        elif len(self.balls) == 0:
            self.done = True

    def _do_rerack(self):
        remaining_bid = next(iter(self.balls.keys()))
        remaining_pos = list(self.balls[remaining_bid])
        # Relocate if in rack area
        if math.hypot(remaining_pos[0] - RACK_APEX[0],
                      remaining_pos[1] - RACK_APEX[1]) < 8.0:
            for cand in [HEAD_SPOT, HEAD_SPOT_ALT, (25.0, 30.0), (30.0, 25.0)]:
                if math.hypot(self.cue[0] - cand[0],
                              self.cue[1] - cand[1]) > 3 * R:
                    remaining_pos = list(cand); break
            else:
                remaining_pos = list(HEAD_SPOT)
        self.balls = {remaining_bid: remaining_pos}
        positions = RACK_POSITIONS[1:]  # 14 positions, apex empty
        available_ids = [i for i in range(1, 16) if i != remaining_bid]
        for bid, pos in zip(available_ids, positions):
            self.balls[bid] = list(pos)
        self.rerack_count += 1

    def get_legal_shots(self) -> list[LegalShot]:
        return generate_legal_shots(self.cue, self.balls, max_cut_deg=75.0)

    def get_obs(self) -> Phase7Obs:
        balls_arr = np.full((MAX_BALLS, 2), -1.0, dtype=np.float32)
        ball_mask = np.zeros(MAX_BALLS, dtype=bool)
        ball_is_cue = np.zeros(MAX_BALLS, dtype=np.float32)
        # Slot 0 = cue
        balls_arr[0] = [self.cue[0] / TABLE_LENGTH, self.cue[1] / TABLE_WIDTH]
        ball_mask[0] = True
        ball_is_cue[0] = 1.0
        # Object balls in slots 1-15, in sorted-id order
        for i, bid in enumerate(sorted(self.balls.keys())):
            if i + 1 >= MAX_BALLS: break
            balls_arr[i + 1] = [self.balls[bid][0] / TABLE_LENGTH,
                                 self.balls[bid][1] / TABLE_WIDTH]
            ball_mask[i + 1] = True

        pockets_arr = np.zeros((MAX_POCKETS, 3), dtype=np.float32)
        for i, (px, py) in enumerate(POCKETS):
            pockets_arr[i] = [px / TABLE_LENGTH, py / TABLE_WIDTH,
                              1.0 if POCKET_RADII[i] < 2.6 else 0.0]

        legal = self.get_legal_shots()
        legal = legal[:MAX_SHOTS]  # hard cap
        shots_arr = np.zeros((MAX_SHOTS, 9), dtype=np.float32)
        shot_mask = np.zeros(MAX_SHOTS, dtype=bool)
        for i, s in enumerate(legal):
            bx, by = self.balls[s.ball_id]
            pocket_pos = POCKETS[s.pocket_idx]
            is_corner = 1.0 if POCKET_RADII[s.pocket_idx] < 2.6 else 0.0
            shots_arr[i] = [
                s.ghost_pos[0] / TABLE_LENGTH, s.ghost_pos[1] / TABLE_WIDTH,
                bx / TABLE_LENGTH, by / TABLE_WIDTH,
                pocket_pos[0] / TABLE_LENGTH, pocket_pos[1] / TABLE_WIDTH,
                s.cut_angle_deg / 90.0,
                s.cue_to_ghost_dist / TABLE_LENGTH,
                s.ball_to_pocket_dist / TABLE_LENGTH,
            ]
            shot_mask[i] = True

        return Phase7Obs(
            balls=balls_arr, ball_mask=ball_mask, ball_is_cue=ball_is_cue,
            pockets=pockets_arr, shots=shots_arr, shot_mask=shot_mask,
            shot_meta=legal,
        )

    def step(self, shot_idx: int, force_raw: float, spin_raw: float, obs: Phase7Obs,
             record_trajectory: bool = False, traj_max_frames: int = 600):
        """Execute the shot corresponding to obs.shot_meta[shot_idx] with the
        decoded (force, spin). If shot_idx is invalid (out of legal list),
        episode ends with 0 reward. If record_trajectory, includes trajectory
        frames and ordered ball ids in info."""
        if self.done:
            return self.get_obs(), 0.0, True, {'reason': 'already done'}

        legal = obs.shot_meta
        if shot_idx >= len(legal):
            self.done = True
            return self.get_obs(), 0.0, True, {'reason': 'invalid shot index'}

        shot = legal[shot_idx]
        aim = shot.aim_angle
        force = decode_force(force_raw)
        spin = decode_spin(spin_raw)
        # Apply execution noise (Gaussian perturbations) to model real-world
        # variability. With noise, hard shots (thin cuts, high force) become
        # statistically risky and the value function learns to avoid them.
        if self.aim_noise_deg > 0:
            aim = aim + np.random.randn() * self.aim_noise_deg * (math.pi / 180.0)
        if self.force_noise_pct > 0:
            force = force * (1.0 + np.random.randn() * self.force_noise_pct)
            force = max(20.0, min(280.0, force))   # keep in reasonable range
        if self.spin_noise > 0:
            spin = spin + np.random.randn() * self.spin_noise
            spin = max(-2.5, min(2.5, spin))
        aim_dx = math.cos(aim); aim_dy = math.sin(aim)

        balls_in_sim = {bid: tuple(pos) for bid, pos in self.balls.items()}
        ordered_ids = [0] + sorted(balls_in_sim.keys())
        result = simulate_shot(
            tuple(self.cue), balls_in_sim,
            aim_dx * force, aim_dy * force,
            spin, aim_dx, aim_dy,
            record_trajectory=record_trajectory,
            traj_max_frames=traj_max_frames,
        )
        pocketed_ids = set(result.pocketed_ids)
        scratch = result.cue_scratched

        info = {
            'shot': shot,
            'aim_angle': aim,
            'force': force, 'spin': spin,
            'pocketed_ids': list(pocketed_ids),
            'scratch': scratch,
        }
        if record_trajectory and result.trajectory is not None:
            info['trajectory'] = result.trajectory.tolist()
            info['trajectory_ball_ids'] = ordered_ids

        if scratch:
            self.done = True
            return self.get_obs(), self.scratch_penalty, True, {**info, 'reason': 'scratch'}

        # Called-shot: target ball must land in the target pocket.
        target_pocketed = shot.ball_id in pocketed_ids
        called_ok = False
        if target_pocketed:
            final_pos = result.final_positions.get(shot.ball_id)
            if final_pos is not None:
                actual = pocket_index_of(final_pos)
                called_ok = (actual == shot.pocket_idx)
        info['called_ok'] = called_ok

        if not called_ok:
            self.done = True
            return self.get_obs(), 0.0, True, {**info, 'reason': 'called shot missed'}

        # Success: reward for all balls pocketed (14.1 rule: incidentals count when call succeeds).
        reward = self.pocket_reward * len(pocketed_ids)
        for bid in pocketed_ids:
            if bid in self.balls:
                del self.balls[bid]
        if 0 in result.final_positions:
            self.cue = list(result.final_positions[0])
        for bid, pos in result.final_positions.items():
            if bid in self.balls:
                self.balls[bid] = list(pos)
        self.total_pocketed += len(pocketed_ids)
        self.shot_idx += 1

        if len(self.balls) == 1 and self.shot_idx < self.max_shots:
            self._do_rerack()
        elif len(self.balls) == 0:
            self.done = True
        if self.shot_idx >= self.max_shots:
            self.done = True

        info['total_pocketed'] = self.total_pocketed
        info['rerack_count'] = self.rerack_count
        info['rerack_happened'] = (len(self.balls) > 1 and
                                    self.rerack_count > 0 and
                                    getattr(self, '_last_rerack_count', 0) != self.rerack_count)
        self._last_rerack_count = self.rerack_count
        return self.get_obs(), reward, self.done, info


# ── Rollout buffer adapted for Phase 7 obs/action ────────────────────────

class Phase7Buffer:
    def __init__(self, num_envs, steps):
        self.num_envs = num_envs
        self.steps = steps
        self.ptr = 0
        N = steps; E = num_envs
        self.balls = np.zeros((N, E, MAX_BALLS, 2), dtype=np.float32)
        self.ball_mask = np.zeros((N, E, MAX_BALLS), dtype=bool)
        self.ball_is_cue = np.zeros((N, E, MAX_BALLS), dtype=np.float32)
        self.pockets = np.zeros((N, E, MAX_POCKETS, 3), dtype=np.float32)
        self.shots = np.zeros((N, E, MAX_SHOTS, 9), dtype=np.float32)
        self.shot_mask = np.zeros((N, E, MAX_SHOTS), dtype=bool)
        self.shot_idx = np.zeros((N, E), dtype=np.int64)
        self.force_raw = np.zeros((N, E), dtype=np.float32)
        self.spin_raw = np.zeros((N, E), dtype=np.float32)
        self.rewards = np.zeros((N, E), dtype=np.float32)
        self.dones = np.zeros((N, E), dtype=np.float32)
        self.log_probs = np.zeros((N, E), dtype=np.float32)
        self.values = np.zeros((N, E), dtype=np.float32)
        self.advantages = np.zeros((N, E), dtype=np.float32)
        self.returns = np.zeros((N, E), dtype=np.float32)

    def add(self, obs_batch, actions, rewards, dones, log_probs, values):
        p = self.ptr
        self.balls[p] = obs_batch['balls'].cpu().numpy()
        self.ball_mask[p] = obs_batch['ball_mask'].cpu().numpy()
        self.ball_is_cue[p] = obs_batch['ball_is_cue'].cpu().numpy()
        self.pockets[p] = obs_batch['pockets'].cpu().numpy()
        self.shots[p] = obs_batch['shots'].cpu().numpy()
        self.shot_mask[p] = obs_batch['shot_mask'].cpu().numpy()
        self.shot_idx[p] = actions[0]
        self.force_raw[p] = actions[1]
        self.spin_raw[p] = actions[2]
        self.rewards[p] = rewards
        self.dones[p] = dones
        self.log_probs[p] = log_probs
        self.values[p] = values
        self.ptr += 1

    def compute_returns(self, last_values, gamma=0.99, gae_lambda=0.95):
        last_gae = 0.0
        for t in reversed(range(self.steps)):
            next_values = last_values if t == self.steps - 1 else self.values[t + 1]
            not_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * not_terminal * last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size, device):
        total = self.steps * self.num_envs
        idx = np.random.permutation(total)
        flat = lambda a: a.reshape((total,) + a.shape[2:])
        balls_f = flat(self.balls); bm_f = flat(self.ball_mask); bic_f = flat(self.ball_is_cue)
        pockets_f = flat(self.pockets); shots_f = flat(self.shots); sm_f = flat(self.shot_mask)
        si_f = flat(self.shot_idx); fr_f = flat(self.force_raw); sr_f = flat(self.spin_raw)
        lp_f = flat(self.log_probs); ret_f = flat(self.returns); adv_f = flat(self.advantages)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            b = idx[start:end]
            yield {
                'balls': torch.from_numpy(balls_f[b]).to(device),
                'ball_mask': torch.from_numpy(bm_f[b]).to(device),
                'ball_is_cue': torch.from_numpy(bic_f[b]).to(device),
                'pockets': torch.from_numpy(pockets_f[b]).to(device),
                'shots': torch.from_numpy(shots_f[b]).to(device),
                'shot_mask': torch.from_numpy(sm_f[b]).to(device),
            }, (
                torch.from_numpy(si_f[b]).long().to(device),
                torch.from_numpy(fr_f[b]).to(device),
                torch.from_numpy(sr_f[b]).to(device),
            ), (
                torch.from_numpy(lp_f[b]).to(device),
                torch.from_numpy(ret_f[b]).to(device),
                torch.from_numpy(adv_f[b]).to(device),
            )


# ── Vectorized env ────────────────────────────────────────────────────────

class VecPhase7:
    def __init__(self, num_envs, max_shots=60, env_class=None, env_kwargs=None):
        self.num_envs = num_envs
        if env_class is None:
            env_class = Phase7Env
        kw = env_kwargs or {}
        self.envs = [env_class(max_shots=max_shots, **kw) for _ in range(num_envs)]
        self.last_obs = None

    def reset(self):
        self.last_obs = [e.reset() for e in self.envs]
        return self._batch_obs(self.last_obs)

    def _batch_obs(self, obs_list):
        return {
            'balls': torch.from_numpy(np.stack([o.balls for o in obs_list])),
            'ball_mask': torch.from_numpy(np.stack([o.ball_mask for o in obs_list])),
            'ball_is_cue': torch.from_numpy(np.stack([o.ball_is_cue for o in obs_list])),
            'pockets': torch.from_numpy(np.stack([o.pockets for o in obs_list])),
            'shots': torch.from_numpy(np.stack([o.shots for o in obs_list])),
            'shot_mask': torch.from_numpy(np.stack([o.shot_mask for o in obs_list])),
        }, obs_list  # also return the raw list so we can access shot_meta

    def step(self, shot_idx_np, force_raw_np, spin_raw_np):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        stats = {'run_lengths': [], 'episodes_finished': 0, 'reracks': []}
        new_obs = [None] * self.num_envs
        for i, env in enumerate(self.envs):
            next_obs, r, d, info = env.step(
                int(shot_idx_np[i]), float(force_raw_np[i]), float(spin_raw_np[i]),
                self.last_obs[i],
            )
            rewards[i] = r
            dones[i] = d
            if d:
                stats['episodes_finished'] += 1
                stats['run_lengths'].append(env.total_pocketed)
                stats['reracks'].append(env.rerack_count)
                next_obs = env.reset()
            new_obs[i] = next_obs
        self.last_obs = new_obs
        return self._batch_obs(new_obs), rewards, dones, stats


# ── Training loop ────────────────────────────────────────────────────────

def train_phase7(num_envs=16, device_name='cpu', max_iters=500,
                 tag='p7_baseline', lr=1e-4, steps_per_update=32,
                 entropy_coef=0.01, log_std_min=-2.5,
                 embed_dim=128, num_heads=8, num_layers=4,
                 warm_start=None, env_class=None, env_kwargs=None,
                 label='Phase 7: token-based 14.1', ckpt_prefix='phase7',
                 aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0):
    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs.update(dict(aim_noise_deg=aim_noise_deg,
                            force_noise_pct=force_noise_pct,
                            spin_noise=spin_noise))
    device = torch.device(device_name)
    net = PoolGameNet(embed_dim=embed_dim, num_heads=num_heads,
                      num_layers=num_layers).to(device)
    if warm_start and os.path.exists(warm_start):
        state = torch.load(warm_start, map_location=device, weights_only=True)
        net.load_state_dict(state)
        print(f'Warm-started from {warm_start}', flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'{label}. PoolGameNet {n_params:,} params on {device}', flush=True)
    print(f'config: tag={tag} lr={lr} steps={steps_per_update} envs={num_envs} '
          f'ent={entropy_coef}', flush=True)

    env = VecPhase7(num_envs, env_class=env_class, env_kwargs=env_kwargs)
    obs_batch, obs_list = env.reset()
    obs_batch = {k: v.to(device) for k, v in obs_batch.items()}

    batch_size = min(256, steps_per_update * num_envs)
    ppo_epochs = 4
    clip_eps = 0.2
    value_coef = 0.5
    buffer = Phase7Buffer(num_envs, steps_per_update)

    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    best_rolling = 0.0
    recent_runs = deque(maxlen=500)

    for iteration in range(max_iters):
        buffer.ptr = 0
        iter_run_lengths = []
        iter_reracks = []
        iter_episodes = 0

        for step in range(steps_per_update):
            with torch.no_grad():
                shot_idx, force_raw, spin_raw, log_prob, value = net.get_action(obs_batch)
            buffer.add(
                obs_batch,
                (shot_idx.cpu().numpy(), force_raw.cpu().numpy(), spin_raw.cpu().numpy()),
                np.zeros(num_envs),   # placeholder — filled just below
                np.zeros(num_envs),
                log_prob.cpu().numpy(),
                value.cpu().numpy(),
            )
            # Use env's shot_meta via obs_list
            (next_obs_batch, next_obs_list), rewards, dones, stats = env.step(
                shot_idx.cpu().numpy(),
                force_raw.cpu().numpy(),
                spin_raw.cpu().numpy(),
            )
            # Overwrite the reward/done slots we just wrote.
            buffer.rewards[buffer.ptr - 1] = rewards
            buffer.dones[buffer.ptr - 1] = dones.astype(np.float32)

            obs_batch = {k: v.to(device) for k, v in next_obs_batch.items()}
            obs_list = next_obs_list
            iter_run_lengths.extend(stats['run_lengths'])
            iter_reracks.extend(stats['reracks'])
            iter_episodes += stats['episodes_finished']

        with torch.no_grad():
            _, _, _, last_value = net.forward(**obs_batch)
        buffer.compute_returns(last_value.cpu().numpy())

        total_pg = total_vl = total_ent = 0.0
        n_updates = 0
        for epoch in range(ppo_epochs):
            for b_obs, b_act, b_trg in buffer.get_batches(batch_size, device):
                shot_i, f_raw, s_raw = b_act
                b_old_lp, b_ret, b_adv = b_trg
                new_lp, entropy, values = net.evaluate_actions(b_obs, shot_i, f_raw, s_raw)
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values, b_ret)
                loss = pg_loss + value_coef * v_loss - entropy_coef * entropy.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(min=log_std_min)
                total_pg += pg_loss.item()
                total_vl += v_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        recent_runs.extend(iter_run_lengths)
        avg_iter = float(np.mean(iter_run_lengths)) if iter_run_lengths else 0.0
        rolling = float(np.mean(recent_runs)) if recent_runs else 0.0
        max_iter = int(np.max(iter_run_lengths)) if iter_run_lengths else 0

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'Iter {iteration+1:5d} | AvgRun={avg_iter:5.2f} Rolling={rolling:5.2f} '
                  f'MaxRun={max_iter:3d} | Eps={iter_episodes} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)
            if rolling > best_rolling:
                best_rolling = rolling
                torch.save(net.state_dict(), f'checkpoints/{ckpt_prefix}_{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f'checkpoints/{ckpt_prefix}_{tag}_latest.pt')

    print(f'Done. Best rolling avg run: {best_rolling:.2f} in {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--envs', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--tag', default='p7_baseline')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--steps_per_update', type=int, default=32)
    p.add_argument('--entropy_coef', type=float, default=0.01)
    p.add_argument('--log_std_min', type=float, default=-2.5)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--warm', default=None)
    p.add_argument('--aim_noise_deg', type=float, default=0.0)
    p.add_argument('--force_noise_pct', type=float, default=0.0)
    p.add_argument('--spin_noise', type=float, default=0.0)
    args = p.parse_args()
    train_phase7(
        num_envs=args.envs, device_name=args.device, max_iters=args.iters,
        tag=args.tag, lr=args.lr, steps_per_update=args.steps_per_update,
        entropy_coef=args.entropy_coef, log_std_min=args.log_std_min,
        embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
        warm_start=args.warm,
        aim_noise_deg=args.aim_noise_deg, force_noise_pct=args.force_noise_pct,
        spin_noise=args.spin_noise,
    )
