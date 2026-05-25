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
# shot_search_phase7 imports Phase7Env from this module — defer to inside
# train_phase7() to avoid circular import.


# ── Phase 7 env ───────────────────────────────────────────────────────────

class Phase7Env:
    def __init__(self, pocket_reward=10.0, max_shots=60,
                 opening_break_force=240.0, scratch_penalty=-10.0,
                 aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0,
                 shape_bonus_max=0.0, movement_penalty_weight=1.0,
                 cue_movement_penalty_weight=0.0,
                 cue_ricochet_penalty_weight=0.0,
                 force_efficiency_penalty_weight=0.0,
                 rail_shot_bonus_weight=0.0,
                 next_shape_bonus_max=0.0,
                 eor_bonus_max=0.0):
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
        # Shape shaping: after each successful shot, evaluate the resulting
        # cue position by the difficulty of the easiest legal next shot.
        # Bonus magnitude scales with shape_bonus_max (0 disables). +max for
        # ideal shape (next shot is dead straight short), 0 for medium-hard,
        # -2*max if snookered (no legal next shot at all).
        self.shape_bonus_max = shape_bonus_max
        # Object-ball movement penalty: subtract from shape score when
        # non-pocketed object balls move significantly (= scatter). Captures
        # "scatter only when necessary" — clean shots cost nothing, scatter
        # shots pay unless they actually pocket and reveal good shape.
        self.movement_penalty_weight = movement_penalty_weight
        # Cue-ball movement penalty: penalize total cue-ball path length per
        # shot (sum of |v|*dt across the sim, including rail bounces). Curbs
        # "send the cue around the table on every shot" behavior. Normalized
        # to 100″ (≈ table length) and capped at 1.5× before applying weight.
        self.cue_movement_penalty_weight = cue_movement_penalty_weight
        # Cue-ricochet penalty: penalize each cue→OB collision beyond the
        # first. Encodes "clean isolation" — touch only the called OB. Capped
        # at 3 extra contacts so a heavy cluster smash doesn't dominate; the
        # EOR bonus rewards deliberate cluster breaks separately.
        self.cue_ricochet_penalty_weight = cue_ricochet_penalty_weight
        # Force efficiency penalty: penalize the agent for using more force
        # than necessary. Free up to medium force (100 in/s); linear penalty
        # above that, capped at max-force shots. Encodes "use the minimum
        # power needed for the shot" — a fundamental real-pool principle.
        self.force_efficiency_penalty_weight = force_efficiency_penalty_weight
        # Rail-shot bonus: positive reward for pocketing balls that were
        # close to a rail (within 3″ of any cushion) pre-shot. Counters the
        # learned aversion to short/medium rail shots — the network was
        # giving these too low probability despite their actual ease.
        self.rail_shot_bonus_weight = rail_shot_bonus_weight
        # Next-shot shape bonus: after each shot, evaluate the EASIEST legal
        # next shot and reward proportional to its ease. Encodes "leave good
        # shape" as a deliberate per-shot signal, not just at end-of-rack.
        # 0 disables; +max for "next shot is dead straight short", 0 for hard
        # follow-up, -max if snookered (no legal next shot).
        self.next_shape_bonus_max = next_shape_bonus_max
        # Natural end-of-rack reward shaping. Fires only at organic states:
        #   - At len(balls)==2 pre-shot: +eor_bonus_max if the agent pocketed
        #     the ball FURTHER from the rack apex (= preserved the closer
        #     ball as a break ball), -eor_bonus_max if they pocketed the
        #     near-apex ball (= wasted the better break candidate).
        #   - On the FIRST shot after a rerack: +2*eor_bonus_max if at least
        #     4 racked balls have moved off their rack positions (= the break
        #     shot actually broke the rack open).
        # No artificial state spawning — only triggers when training organically
        # reaches end-of-rack positions, so distribution is unchanged.
        self.eor_bonus_max = eor_bonus_max
        self.reset()

    def reset(self):
        self.cue, self.balls = sample_phase6_setup()
        self.shot_idx = 0
        self.done = False
        self.rerack_count = 0
        self.total_pocketed = 0
        self.pending_rerack = False
        # Track the break ball (the one preserved through rerack) so we can
        # exclude it from rack-scatter counts on the post-rerack break shot.
        self._break_ball_id_after_rerack = None
        # When True, the next step() is a "break shot" (first shot after
        # rerack) and should be evaluated for scatter bonus.
        self._post_rerack_break_pending = False
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
        # Relocate only if the break ball overlaps a rack position. The
        # previous check used an 8″ radius around RACK_APEX, which is a
        # circle that includes points in FRONT of the apex (outside the
        # rack body) and excludes some balls at the back of the rack.
        # Correct check: ball overlaps any of the 14 reracked positions
        # if its center is within 2R (+small margin) of any of them.
        rack_positions = RACK_POSITIONS[1:]    # 14 positions, apex empty
        overlap_thresh = 2.0 * R + 0.1
        overlaps_rack = any(
            math.hypot(remaining_pos[0] - rx,
                       remaining_pos[1] - ry) < overlap_thresh
            for rx, ry in rack_positions
        )
        if overlaps_rack:
            for cand in [HEAD_SPOT, HEAD_SPOT_ALT, (25.0, 30.0), (30.0, 25.0)]:
                if math.hypot(self.cue[0] - cand[0],
                              self.cue[1] - cand[1]) > 3 * R:
                    remaining_pos = list(cand); break
            else:
                remaining_pos = list(HEAD_SPOT)
        self.balls = {remaining_bid: remaining_pos}
        available_ids = [i for i in range(1, 16) if i != remaining_bid]
        for bid, pos in zip(available_ids, rack_positions):
            self.balls[bid] = list(pos)
        self.rerack_count += 1

    def get_legal_shots(self) -> list[LegalShot]:
        return generate_legal_shots(self.cue, self.balls, max_cut_deg=80.0)

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

        # End-of-rack reward shaping snapshots (only used if eor_bonus_max > 0).
        # Captured BEFORE the shot fires so we know the pre-state.
        pre_n_balls = len(self.balls)
        pre_balls_snapshot = dict(self.balls) if pre_n_balls == 2 else None
        was_post_rerack_break = self._post_rerack_break_pending
        # Clear the flag — we're handling this break shot (or it's not one).
        self._post_rerack_break_pending = False

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
            'cue_path_len': float(result.cue_path_len),
            'cue_contacts': int(result.cue_contacts),
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

        # Shot difficulty penalty (optional): penalize the agent for taking
        # unnecessarily hard shots when easier ones exist. Encodes "prefer
        # easy shots all else equal" — the simulator's noise alone is too
        # weak a signal at aim_noise=0.03°. Fires when shape_bonus_max > 0.
        if self.shape_bonus_max > 0 and not self.done:
            reward += self._shape_bonus(shot=shot)
        # Cue-control penalties always fire (independent of shape bonus).
        # These tax wandering cue paths and incidental contacts directly,
        # whether or not we use a shape formula.
        if not self.done:
            reward += self._cue_penalty(cue_path_len=result.cue_path_len,
                                         cue_contacts=result.cue_contacts,
                                         force=force,
                                         cut_deg=shot.cut_angle_deg)
        # OB-scatter penalty — taxes secondary ball movement (clean isolation).
        # Independent of shape bonus, so it can fire even with shape_bonus_max=0.
        if not self.done:
            reward += self._scatter_penalty(pre_shot_balls=balls_in_sim)
        # Rail-shot bonus: reward pocketing balls that were near a rail.
        if self.rail_shot_bonus_weight > 0:
            reward += self._rail_shot_bonus(pocketed_ids, balls_in_sim)
        # Next-shot shape bonus — reward leaving easy follow-up shots.
        # The "shape" half of shape vs difficulty: difficulty (above) penalizes
        # taking hard shots; this rewards leaving easy ones for next time.
        if self.next_shape_bonus_max > 0 and not self.done:
            reward += self._next_shape_bonus()

        info['total_pocketed'] = self.total_pocketed
        info['rerack_count'] = self.rerack_count
        info['rerack_happened'] = (len(self.balls) > 1 and
                                    self.rerack_count > 0 and
                                    getattr(self, '_last_rerack_count', 0) != self.rerack_count)
        self._last_rerack_count = self.rerack_count

        # Natural end-of-rack reward shaping (only fires if eor_bonus_max > 0).
        if self.eor_bonus_max > 0:
            eor = self._eor_bonus(pre_n_balls=pre_n_balls,
                                    pre_balls_snapshot=pre_balls_snapshot,
                                    pocketed_ids=pocketed_ids,
                                    was_post_rerack_break=was_post_rerack_break)
            reward += eor
            info['eor_bonus'] = eor

        # If rerack just fired, mark the next step as the break shot so the
        # next step()'s scatter check fires.
        if info['rerack_happened']:
            self._post_rerack_break_pending = True
            # Track the break ball ID (the one preserved through rerack — i.e.,
            # the ball not at any RACK_POSITIONS[1:] right now).
            rack_pos_set = RACK_POSITIONS[1:]
            for bid, pos in self.balls.items():
                at_rack = any(math.hypot(pos[0]-rp[0], pos[1]-rp[1]) < 2 * R
                              for rp in rack_pos_set)
                if not at_rack:
                    self._break_ball_id_after_rerack = bid
                    break

        return self.get_obs(), reward, self.done, info

    def _eor_bonus(self, pre_n_balls, pre_balls_snapshot, pocketed_ids,
                   was_post_rerack_break):
        """Natural end-of-rack reward.

        At pre_n_balls==2: identify which remaining ball is the better
        break-ball candidate (multi-criterion: in range 4–14″ from apex,
        sweet-spot at 9″, clear line of sight to apex). Bonus for pocketing
        the OTHER one (preserving the better break ball). Falls back to
        nearest-to-apex if neither ball scores as a usable break ball
        candidate.

        At pre_n_balls==3: half-strength version of the same — preserve the
        better-quality break-ball candidate.

        After a rerack (was_post_rerack_break==True): bonus for breaking the
        rack open — at least 4 of the 14 reracked balls have moved off their
        rack positions.
        """
        bonus = 0.0
        # Save-break-ball decision at len==2.
        if pre_n_balls == 2 and pre_balls_snapshot and pocketed_ids:
            best_break_ball = self._select_break_ball_candidate(
                pre_balls_snapshot)
            if best_break_ball in pocketed_ids:
                bonus -= self.eor_bonus_max         # wasted the break ball
            else:
                bonus += self.eor_bonus_max         # saved it for last
        # Key-ball-1 → break-ball sequencing at len==3.
        elif pre_n_balls == 3 and pre_balls_snapshot and pocketed_ids:
            best_break_ball = self._select_break_ball_candidate(
                pre_balls_snapshot)
            if best_break_ball in pocketed_ids:
                bonus -= 0.5 * self.eor_bonus_max   # broke the sequence
            else:
                bonus += 0.5 * self.eor_bonus_max   # preserved break ball
        # Break shot: first shot after a rerack.
        if was_post_rerack_break:
            rack_positions = RACK_POSITIONS[1:]
            still_at_rack = 0
            for bid, pos in self.balls.items():
                if bid == self._break_ball_id_after_rerack:
                    continue
                if any(math.hypot(pos[0] - rp[0], pos[1] - rp[1]) < 2 * R
                       for rp in rack_positions):
                    still_at_rack += 1
            scatter = 14 - still_at_rack    # 14 was the post-rerack count
            if scatter >= 3:
                bonus += 2.0 * self.eor_bonus_max
        return bonus

    def _break_ball_quality(self, ball_pos, snapshot):
        """Score for how good a ball at `ball_pos` would be as a preserved
        break ball. Combines:
          (a) distance to RACK_APEX — sweet spot at d=9″, full score 1.0
              within [4, 14]″, decays exponentially outside that band.
          (b) clear line of sight from ball → RACK_APEX (no other ball
              within 2R of the segment). Blocked → 0 (hard cutoff).
        A ball outside the ideal range still scores positive if its line
        is clear, so it always beats a blocked ball. Returns 0 only when
        the line is blocked or the ball coincides with the apex.
        """
        bx, by = ball_pos
        apex_x, apex_y = RACK_APEX
        dx_a = apex_x - bx
        dy_a = apex_y - by
        d = math.hypot(dx_a, dy_a)
        if d < 1e-6:
            return 0.0
        if 4.0 <= d <= 14.0:
            dist_score = (d - 4.0) / 5.0 if d <= 9.0 else (14.0 - d) / 5.0
            dist_score = max(0.05, dist_score)
        else:
            out_of_band = abs(d - 9.0) - 5.0
            dist_score = 0.3 * math.exp(-out_of_band / 5.0)
        ux = dx_a / d
        uy = dy_a / d
        clearance_sq = (2.0 * R) * (2.0 * R)
        for _bid, pos in snapshot.items():
            if pos[0] == bx and pos[1] == by:
                continue
            ex = pos[0] - bx
            ey = pos[1] - by
            t = ex * ux + ey * uy
            if t < 0.0 or t > d:
                continue
            perp_x = ex - t * ux
            perp_y = ey - t * uy
            if perp_x * perp_x + perp_y * perp_y < clearance_sq:
                return 0.0
        return dist_score

    def _select_break_ball_candidate(self, snapshot):
        """Return the ball_id to preserve as the break ball, by max quality.
        Quality is always > 0 unless the line to the apex is blocked; in
        the degenerate case where all balls are blocked, falls back to
        nearest-to-apex so the n=2/n=3 EOR signal still fires."""
        if not snapshot:
            return None
        quality = {bid: self._break_ball_quality(pos, snapshot)
                   for bid, pos in snapshot.items()}
        if max(quality.values()) > 0.0:
            return max(quality, key=quality.get)
        apex_dists = {bid: math.hypot(pos[0] - RACK_APEX[0],
                                        pos[1] - RACK_APEX[1])
                      for bid, pos in snapshot.items()}
        return min(apex_dists, key=apex_dists.get)

    def _shape_bonus(self, shot=None, **_ignored):
        """Shot-difficulty penalty. Penalizes the agent in proportion to the
        current shot's difficulty so it prefers easier shots when they're
        available.

        difficulty = cut_norm + dist_norm   (no longer hard-clamped at 1.0)
            cut_norm  = cut_angle_deg / 45.0           (1.0 at 45° cut,
                                                         1.73 at 78°, 2.0 at 90°)
            dist_norm = max(0, total_dist − 25) / 50.0 (1.0 at 75″ total)
        Clamped at 3.0 to bound worst-case penalty at -3·shape_bonus_max.
        Returns difficulty * -shape_bonus_max.

        Examples (with shape_bonus_max = 2.0):
            cut=17°, total=25″: difficulty = 0.38  → reward −0.76
            cut=45°, total=30″: difficulty = 1.10  → reward −2.20
            cut=60°, total=30″: difficulty = 1.43  → reward −2.86
            cut=78°, total=30″: difficulty = 1.83  → reward −3.66
            cut=80°, total=80″: difficulty = 2.88  → reward −5.76
        """
        if shot is None:
            return 0.0
        cut_norm = max(0.0, shot.cut_angle_deg) / 45.0
        total_dist = shot.cue_to_ghost_dist + shot.ball_to_pocket_dist
        dist_norm = max(0.0, total_dist - 25.0) / 50.0
        difficulty = min(3.0, cut_norm + dist_norm)
        return -self.shape_bonus_max * difficulty

    def _cue_penalty(self, cue_path_len=0.0, cue_contacts=1, force=0.0,
                     cut_deg=0.0):
        """Cue-control penalty applied every shot, independent of shape bonus.
        Penalizes long cue ball travel (wandering), ricochets through
        non-target balls (lack of clean isolation), and unnecessarily high
        force (use minimum power needed). Returns a negative value or zero.

        Force efficiency is CUT-AWARE: harder cuts need more force to deliver
        the same OB velocity (since OB receives only cos(cut) of the impulse).
        Threshold = 100 / cos(cut), capped at 250. So a 0° cut gets penalty-
        free force ≤100, a 50° cut gets ≤156, a 70° cut gets ≤250."""
        cue_norm = min(1.5, max(0.0, cue_path_len) / 100.0)
        extra_contacts = min(3, max(0, cue_contacts - 1))
        cut_rad = math.radians(max(0.0, cut_deg))
        threshold = min(250.0, 100.0 / max(0.3, math.cos(cut_rad)))
        force_excess = min(1.0, max(0.0, force - threshold) / 150.0)
        return -(self.cue_movement_penalty_weight * cue_norm
                 + self.cue_ricochet_penalty_weight * extra_contacts
                 + self.force_efficiency_penalty_weight * force_excess)

    def _rail_shot_bonus(self, pocketed_ids, pre_balls):
        """Bonus per pocketed ball that was within 3″ of a cushion pre-shot.
        Counters the learned rail-shot aversion observed in v8-v10 demos.
        Capped at +rail_shot_bonus_weight per ball (no chain-multiplier)."""
        if not pre_balls:
            return 0.0
        bonus = 0.0
        for bid in pocketed_ids:
            pre = pre_balls.get(bid)
            if pre is None:
                continue
            x, y = pre
            dist_to_rail = min(x, y, 100.0 - x, 50.0 - y)
            if dist_to_rail <= 3.0:
                bonus += self.rail_shot_bonus_weight
        return bonus

    def _next_shape_bonus(self):
        """Reward leaving good shape: scales with ease of the EASIEST legal
        next shot from the resulting cue position. This is the "shape" half
        of shape vs difficulty — difficulty penalty taxes hard CURRENT shots;
        this rewards leaving easy NEXT shots, on every shot, not just EOR.

        ease(shot) = 1 - min(1, cut/45 + max(0, total_dist-25)/50)
                   = 1 at 0° straight 25″ shot, 0 at 45°+/75″+ shot
        Returns:
            +next_shape_bonus_max × best_ease   (good shape)
            -next_shape_bonus_max               (snookered, no legal shots)
        """
        if not self.balls:
            return 0.0
        next_shots = generate_legal_shots(self.cue, self.balls, max_cut_deg=80.0)
        if not next_shots:
            return -self.next_shape_bonus_max
        best_ease = 0.0
        for s in next_shots:
            cut_norm = s.cut_angle_deg / 45.0
            total_dist = s.cue_to_ghost_dist + s.ball_to_pocket_dist
            dist_norm = max(0.0, total_dist - 25.0) / 50.0
            difficulty = min(1.0, cut_norm + dist_norm)
            ease = 1.0 - difficulty
            if ease > best_ease:
                best_ease = ease
        return self.next_shape_bonus_max * best_ease

    def _scatter_penalty(self, pre_shot_balls=None):
        """OB-scatter penalty: penalize total displacement of non-pocketed
        object balls vs pre-shot positions. Encodes "preserve table layout":
        a clean isolation pot moves only the called ball; cluster-disturbing
        shots cost reward unless they pay back via pocket bonus or EOR.
        Normalized to 50″ of total OB displacement, capped at 1.0."""
        if not pre_shot_balls or self.movement_penalty_weight <= 0:
            return 0.0
        ob_movement = 0.0
        for bid, post_pos in self.balls.items():
            pre = pre_shot_balls.get(bid)
            if pre is None:
                continue
            ob_movement += math.hypot(post_pos[0] - pre[0],
                                       post_pos[1] - pre[1])
        movement_norm = min(1.0, ob_movement / 50.0)
        return -self.movement_penalty_weight * movement_norm


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
        stats = {'run_lengths': [], 'episodes_finished': 0, 'reracks': [],
                 'cue_path_lens': [], 'cue_contacts': []}
        new_obs = [None] * self.num_envs
        for i, env in enumerate(self.envs):
            next_obs, r, d, info = env.step(
                int(shot_idx_np[i]), float(force_raw_np[i]), float(spin_raw_np[i]),
                self.last_obs[i],
            )
            rewards[i] = r
            dones[i] = d
            cpl = info.get('cue_path_len')
            if cpl is not None:
                stats['cue_path_lens'].append(cpl)
            cc = info.get('cue_contacts')
            if cc is not None:
                stats['cue_contacts'].append(cc)
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
                 aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0,
                 shape_bonus_max=0.0, movement_penalty_weight=1.0,
                 cue_movement_penalty_weight=0.0,
                 cue_ricochet_penalty_weight=0.0,
                 force_efficiency_penalty_weight=0.0,
                 rail_shot_bonus_weight=0.0,
                 next_shape_bonus_max=0.0,
                 eor_bonus_max=0.0,
                 search_train=False, search_k=2, search_m=1, search_mc=1):
    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs.update(dict(aim_noise_deg=aim_noise_deg,
                            force_noise_pct=force_noise_pct,
                            spin_noise=spin_noise,
                            shape_bonus_max=shape_bonus_max,
                            movement_penalty_weight=movement_penalty_weight,
                            cue_movement_penalty_weight=cue_movement_penalty_weight,
                            cue_ricochet_penalty_weight=cue_ricochet_penalty_weight,
                            force_efficiency_penalty_weight=force_efficiency_penalty_weight,
                            rail_shot_bonus_weight=rail_shot_bonus_weight,
                            next_shape_bonus_max=next_shape_bonus_max,
                            eor_bonus_max=eor_bonus_max))
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
    if search_train:
        print(f'  SEARCH-TRAINING enabled: K={search_k} M={search_m} MC={search_mc}',
              flush=True)
        from shot_search_phase7 import shot_search_phase7

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
    recent_cue_paths = deque(maxlen=2000)  # last ~2k shots' cue path lengths
    recent_cue_contacts = deque(maxlen=2000)

    for iteration in range(max_iters):
        buffer.ptr = 0
        iter_run_lengths = []
        iter_reracks = []
        iter_episodes = 0
        iter_cue_paths = []
        iter_cue_contacts = []

        for step in range(steps_per_update):
            if search_train:
                # Search-based action selection: for each env, run shot_search_phase7
                # to pick the action whose mean Q (over candidates × MC samples) is
                # highest. Then compute log_prob/value of the search-selected action
                # under the current policy for PPO.
                shot_idx_np = np.zeros(num_envs, dtype=np.int64)
                force_raw_np = np.zeros(num_envs, dtype=np.float32)
                spin_raw_np = np.zeros(num_envs, dtype=np.float32)
                for i, single_env in enumerate(env.envs):
                    action = shot_search_phase7(
                        net, single_env, obs_list[i],
                        K_shots=search_k, M_per_shot=search_m,
                        noise_samples=search_mc, device=device,
                    )
                    if action is not None:
                        shot_idx_np[i] = action[0]
                        force_raw_np[i] = action[1]
                        spin_raw_np[i] = action[2]
                    # else: 0/0/0 — no legal shots, env will end episode anyway
                with torch.no_grad():
                    log_prob, _, value = net.evaluate_actions(
                        obs_batch,
                        torch.from_numpy(shot_idx_np).to(device),
                        torch.from_numpy(force_raw_np).to(device),
                        torch.from_numpy(spin_raw_np).to(device),
                    )
            else:
                with torch.no_grad():
                    shot_idx, force_raw, spin_raw, log_prob, value = net.get_action(obs_batch)
                shot_idx_np = shot_idx.cpu().numpy()
                force_raw_np = force_raw.cpu().numpy()
                spin_raw_np = spin_raw.cpu().numpy()
            buffer.add(
                obs_batch,
                (shot_idx_np, force_raw_np, spin_raw_np),
                np.zeros(num_envs),   # placeholder — filled just below
                np.zeros(num_envs),
                log_prob.cpu().numpy(),
                value.cpu().numpy(),
            )
            # Use env's shot_meta via obs_list
            (next_obs_batch, next_obs_list), rewards, dones, stats = env.step(
                shot_idx_np, force_raw_np, spin_raw_np,
            )
            # Overwrite the reward/done slots we just wrote.
            buffer.rewards[buffer.ptr - 1] = rewards
            buffer.dones[buffer.ptr - 1] = dones.astype(np.float32)

            obs_batch = {k: v.to(device) for k, v in next_obs_batch.items()}
            obs_list = next_obs_list
            iter_run_lengths.extend(stats['run_lengths'])
            iter_reracks.extend(stats['reracks'])
            iter_episodes += stats['episodes_finished']
            iter_cue_paths.extend(stats.get('cue_path_lens', []))
            iter_cue_contacts.extend(stats.get('cue_contacts', []))

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
        recent_cue_paths.extend(iter_cue_paths)
        recent_cue_contacts.extend(iter_cue_contacts)
        avg_iter = float(np.mean(iter_run_lengths)) if iter_run_lengths else 0.0
        rolling = float(np.mean(recent_runs)) if recent_runs else 0.0
        max_iter = int(np.max(iter_run_lengths)) if iter_run_lengths else 0
        cue_path_mean = float(np.mean(recent_cue_paths)) if recent_cue_paths else 0.0
        cue_contacts_mean = float(np.mean(recent_cue_contacts)) if recent_cue_contacts else 0.0

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'Iter {iteration+1:5d} | AvgRun={avg_iter:5.2f} Rolling={rolling:5.2f} '
                  f'MaxRun={max_iter:3d} | Eps={iter_episodes} | '
                  f'PG={total_pg/n_updates:.4f} VL={total_vl/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | CuePath={cue_path_mean:5.1f} '
                  f'Contacts={cue_contacts_mean:4.2f} | {elapsed:.0f}s', flush=True)
            if rolling > best_rolling:
                best_rolling = rolling
                torch.save(net.state_dict(), f'checkpoints/{ckpt_prefix}_{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f'checkpoints/{ckpt_prefix}_{tag}_latest.pt')

    print(f'Done. Best rolling avg run: {best_rolling:.2f} in {time.time()-t0:.0f}s',
          flush=True)


# ── Distillation training (search-improved policy/value targets) ──────────

def train_phase7_distill(
    num_envs=16, device_name='cpu', max_iters=200,
    tag='p7_distill', lr=1e-4, steps_per_update=32,
    embed_dim=128, num_heads=8, num_layers=4,
    warm_start=None, env_class=None, env_kwargs=None,
    label='Phase 7: distillation', ckpt_prefix='phase7',
    aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0,
    shape_bonus_max=0.0, movement_penalty_weight=1.0,
    cue_movement_penalty_weight=0.0,
    cue_ricochet_penalty_weight=0.0,
    force_efficiency_penalty_weight=0.0,
    rail_shot_bonus_weight=0.0,
    next_shape_bonus_max=0.0,
    eor_bonus_max=0.0,
    search_k=4, search_m=1, search_mc=1,
    softmax_temp=1.0,
    ce_weight=1.0, mse_force_weight=0.1, mse_spin_weight=0.1,
    value_weight=0.5, entropy_weight=0.005,
    log_std_min=-2.5,
    teacher_warm_start=None, frozen_teacher_iters=0,
):
    """Distillation training: at each rollout step, run depth-1 search and
    use the search results as targets for cross-entropy / MSE losses, no
    PPO importance ratio. Avoids the gradient blow-up that breaks PPO+search.

    Loss = ce_weight · CE(policy logits, search-Q-softmax)
         + mse_force_weight · MSE(force_mean[chosen], search force)
         + mse_spin_weight  · MSE(spin_mean[chosen],  search spin)
         + value_weight · MSE(value, search best_q)
         − entropy_weight · entropy(policy logits)
    """
    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs.update(dict(aim_noise_deg=aim_noise_deg,
                            force_noise_pct=force_noise_pct,
                            spin_noise=spin_noise,
                            shape_bonus_max=shape_bonus_max,
                            movement_penalty_weight=movement_penalty_weight,
                            cue_movement_penalty_weight=cue_movement_penalty_weight,
                            cue_ricochet_penalty_weight=cue_ricochet_penalty_weight,
                            force_efficiency_penalty_weight=force_efficiency_penalty_weight,
                            rail_shot_bonus_weight=rail_shot_bonus_weight,
                            next_shape_bonus_max=next_shape_bonus_max,
                            eor_bonus_max=eor_bonus_max))
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
          f'K={search_k} M={search_m} MC={search_mc} '
          f'(weights ce={ce_weight} mse_f={mse_force_weight} mse_s={mse_spin_weight} '
          f'v={value_weight} ent={entropy_weight})', flush=True)

    teacher_net = None
    if teacher_warm_start and frozen_teacher_iters > 0:
        if not os.path.exists(teacher_warm_start):
            raise FileNotFoundError(
                f'teacher_warm_start not found: {teacher_warm_start}')
        teacher_net = PoolGameNet(embed_dim=embed_dim, num_heads=num_heads,
                                   num_layers=num_layers).to(device)
        teacher_state = torch.load(teacher_warm_start, map_location=device,
                                    weights_only=True)
        teacher_net.load_state_dict(teacher_state)
        teacher_net.eval()
        for p in teacher_net.parameters():
            p.requires_grad = False
        print(f'Frozen teacher loaded from {teacher_warm_start} '
              f'(used for first {frozen_teacher_iters} iters)', flush=True)

    from shot_search_phase7 import shot_search_distill

    env = VecPhase7(num_envs, env_class=env_class, env_kwargs=env_kwargs)
    obs_batch, obs_list = env.reset()
    obs_batch = {k: v.to(device) for k, v in obs_batch.items()}

    batch_size = min(256, steps_per_update * num_envs)
    epochs_per_iter = 2

    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    best_rolling = 0.0
    recent_runs = deque(maxlen=500)

    for iteration in range(max_iters):
        # Rollout — for each env-step, run search and record targets.
        # Use Python lists and convert to numpy at end of rollout.
        roll_obs = {k: [] for k in ('balls', 'ball_mask', 'ball_is_cue',
                                      'pockets', 'shots', 'shot_mask')}
        roll_shot = []
        roll_force = []
        roll_spin = []
        roll_target_dist = []
        roll_target_value = []
        iter_run_lengths = []
        iter_episodes = 0

        for step in range(steps_per_update):
            # Snapshot the current obs for training targets.
            for k in roll_obs:
                roll_obs[k].append(obs_batch[k].cpu().numpy())

            shot_idx_np = np.zeros(num_envs, dtype=np.int64)
            force_np = np.zeros(num_envs, dtype=np.float32)
            spin_np = np.zeros(num_envs, dtype=np.float32)
            target_dist = np.zeros((num_envs, MAX_SHOTS), dtype=np.float32)
            target_value = np.zeros(num_envs, dtype=np.float32)

            use_teacher = (teacher_net is not None
                           and iteration < frozen_teacher_iters)
            if (teacher_net is not None and iteration == frozen_teacher_iters
                    and step == 0):
                print(f'Iter {iteration+1}: switching from frozen teacher to '
                      f'self-search', flush=True)
            search_net = teacher_net if use_teacher else net
            for i, single_env in enumerate(env.envs):
                best_action, shot_qs, best_q = shot_search_distill(
                    search_net, single_env, obs_list[i],
                    K_shots=search_k, M_per_shot=search_m,
                    noise_samples=search_mc, device=device,
                )
                if best_action is None or not shot_qs:
                    # No legal shots — env will end. Set uniform fallback target.
                    target_dist[i, 0] = 1.0
                    target_value[i] = 0.0
                    continue
                shot_idx_np[i], force_np[i], spin_np[i] = best_action
                target_value[i] = best_q
                # Soft target = softmax(Q / T) over evaluated shot indices.
                idxs = np.array(list(shot_qs.keys()), dtype=np.int64)
                qs = np.array(list(shot_qs.values()), dtype=np.float32)
                # Numerically-stable softmax.
                soft = np.exp((qs - qs.max()) / max(softmax_temp, 1e-6))
                soft = soft / soft.sum()
                target_dist[i, idxs] = soft

            roll_shot.append(shot_idx_np)
            roll_force.append(force_np)
            roll_spin.append(spin_np)
            roll_target_dist.append(target_dist)
            roll_target_value.append(target_value)

            (next_obs_batch, next_obs_list), rewards, dones, stats = env.step(
                shot_idx_np, force_np, spin_np,
            )
            iter_run_lengths.extend(stats['run_lengths'])
            iter_episodes += stats['episodes_finished']
            obs_batch = {k: v.to(device) for k, v in next_obs_batch.items()}
            obs_list = next_obs_list

        # Stack rollout into flat arrays of size N = steps_per_update * num_envs.
        all_obs = {k: np.concatenate(roll_obs[k], axis=0) for k in roll_obs}
        all_shot = np.concatenate(roll_shot)
        all_force = np.concatenate(roll_force)
        all_spin = np.concatenate(roll_spin)
        all_target_dist = np.concatenate(roll_target_dist)
        all_target_value = np.concatenate(roll_target_value)
        N = len(all_shot)

        # Update — multi-epoch SGD on the rollout buffer.
        total_ce = total_mf = total_ms = total_v = total_ent = 0.0
        n_updates = 0
        for epoch in range(epochs_per_iter):
            perm = np.random.permutation(N)
            for start in range(0, N, batch_size):
                bi = perm[start:start + batch_size]
                B = len(bi)
                obs_b = {k: torch.from_numpy(all_obs[k][bi]).to(device)
                         for k in all_obs}
                shot_b = torch.from_numpy(all_shot[bi]).long().to(device)
                force_b = torch.from_numpy(all_force[bi]).to(device)
                spin_b = torch.from_numpy(all_spin[bi]).to(device)
                tgt_dist_b = torch.from_numpy(all_target_dist[bi]).to(device)
                tgt_value_b = torch.from_numpy(all_target_value[bi]).to(device)

                scores, f_means, s_means, value = net.forward(**obs_b)
                log_probs = F.log_softmax(scores, dim=-1)
                # Cross-entropy: target dist already only has mass on top-K.
                ce_loss = -(tgt_dist_b * log_probs).sum(-1).mean()
                # Gather force/spin at the search-chosen shot for MSE.
                f_chosen = f_means.gather(1, shot_b.unsqueeze(-1)).squeeze(-1)
                s_chosen = s_means.gather(1, shot_b.unsqueeze(-1)).squeeze(-1)
                mse_f = F.mse_loss(f_chosen, force_b)
                mse_s = F.mse_loss(s_chosen, spin_b)
                value_loss = F.mse_loss(value, tgt_value_b)
                # Entropy of the policy distribution (over masked shots).
                probs = F.softmax(scores, dim=-1)
                ent = -(probs * log_probs).sum(-1).mean()

                loss = (ce_weight * ce_loss
                        + mse_force_weight * mse_f
                        + mse_spin_weight * mse_s
                        + value_weight * value_loss
                        - entropy_weight * ent)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(min=log_std_min)
                total_ce += ce_loss.item()
                total_mf += mse_f.item()
                total_ms += mse_s.item()
                total_v += value_loss.item()
                total_ent += ent.item()
                n_updates += 1

        recent_runs.extend(iter_run_lengths)
        avg_iter = float(np.mean(iter_run_lengths)) if iter_run_lengths else 0.0
        rolling = float(np.mean(recent_runs)) if recent_runs else 0.0
        max_iter = int(np.max(iter_run_lengths)) if iter_run_lengths else 0

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'Iter {iteration+1:5d} | AvgRun={avg_iter:5.2f} '
                  f'Rolling={rolling:5.2f} MaxRun={max_iter:3d} | '
                  f'Eps={iter_episodes} | '
                  f'CE={total_ce/n_updates:.3f} MF={total_mf/n_updates:.3f} '
                  f'MS={total_ms/n_updates:.3f} VL={total_v/n_updates:.2f} '
                  f'Ent={total_ent/n_updates:.3f} | {elapsed:.0f}s', flush=True)
            if rolling > best_rolling:
                best_rolling = rolling
                torch.save(net.state_dict(),
                           f'checkpoints/{ckpt_prefix}_{tag}_best.pt')
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(),
                       f'checkpoints/{ckpt_prefix}_{tag}_latest.pt')

    print(f'Done. Best rolling avg run: {best_rolling:.2f} in '
          f'{time.time()-t0:.0f}s', flush=True)


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
    p.add_argument('--shape_bonus_max', type=float, default=0.0,
                   help='Per-shot shape shaping reward magnitude (0 disables).')
    p.add_argument('--movement_penalty_weight', type=float, default=1.0,
                   help='Weight on the OB-movement penalty inside the shape '
                        'bonus. 0 disables; 1 (default) makes 50″ of total '
                        'OB scatter cost the same as one full-magnitude '
                        'shape unit.')
    p.add_argument('--cue_movement_penalty_weight', type=float, default=0.0,
                   help='Weight on the cue-ball-path-length penalty inside '
                        'the shape bonus. 0 disables. Cue path normalized to '
                        '100″ and capped at 1.5×, so weight 1.0 makes a '
                        '100″ cue trajectory cost a full shape unit.')
    p.add_argument('--force_efficiency_penalty_weight', type=float, default=0.0,
                   help='Penalty for using force > 100 in/s (medium shot). '
                        'Linear ramp; capped at force=250. Encourages '
                        '"use minimum power needed". Cut-aware: threshold '
                        'rises with cut angle since hard cuts need more force.')
    p.add_argument('--rail_shot_bonus_weight', type=float, default=0.0,
                   help='Per-ball bonus for pocketing OB that was within 3" '
                        'of any cushion pre-shot. Counters learned aversion '
                        'to short/medium rail shots.')
    p.add_argument('--next_shape_bonus_max', type=float, default=0.0,
                   help='Max reward for leaving easy NEXT shot. Independent '
                        'from shape_bonus_max (which penalizes hard CURRENT '
                        'shots). Together they teach "shape" — pick easy and '
                        'leave easy.')
    p.add_argument('--cue_ricochet_penalty_weight', type=float, default=0.0,
                   help='Weight on the cue-ricochet penalty: each cue→OB '
                        'contact beyond the first costs this much (capped at '
                        '3 extra contacts). Encodes "clean isolation" play.')
    p.add_argument('--eor_bonus_max', type=float, default=0.0,
                   help='Natural end-of-rack reward magnitude (0 disables). '
                        'Fires only at organic states: ±max for save/waste '
                        'break-ball decision at len==2, +2*max for '
                        'rack-scatter on the post-rerack break shot.')
    p.add_argument('--search_train', action='store_true',
                   help='Use shot_search_phase7 to select actions during '
                        'training rollouts (mini AlphaZero-style).')
    p.add_argument('--search_k', type=int, default=2,
                   help='Search: top-K candidate shots per state.')
    p.add_argument('--search_m', type=int, default=1,
                   help='Search: force/spin variants per shot candidate.')
    p.add_argument('--search_mc', type=int, default=1,
                   help='Search: Monte-Carlo noise samples per candidate.')
    p.add_argument('--distill', action='store_true',
                   help='Use AlphaZero-style distillation training: search '
                        'targets via cross-entropy/MSE, no PPO ratio.')
    p.add_argument('--softmax_temp', type=float, default=1.0,
                   help='Distill: temperature for softmax over search Qs '
                        '(lower = harder targets).')
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--mse_force_weight', type=float, default=0.1)
    p.add_argument('--mse_spin_weight', type=float, default=0.1)
    p.add_argument('--value_weight', type=float, default=0.5)
    p.add_argument('--distill_entropy', type=float, default=0.005,
                   help='Distill: entropy regularization weight (separate '
                        'from PPO entropy_coef).')
    p.add_argument('--teacher_ckpt', default=None,
                   help='Distill: frozen teacher checkpoint to use for '
                        'search rollouts during the first '
                        '--frozen_teacher_iters iterations. After that, '
                        'switch to self-search (the live network).')
    p.add_argument('--frozen_teacher_iters', type=int, default=0,
                   help='Distill: number of initial iters that use the '
                        'frozen teacher for search; 0 = always self-search.')
    p.add_argument('--curriculum', action='store_true',
                   help='Use Phase9CurriculumEnv (pure key+break-ball drill) '
                        'instead of regular Phase7Env. Note: pure curriculum '
                        'tends to overfit; prefer --mixed for production runs.')
    p.add_argument('--mixed', action='store_true',
                   help='Use Phase9MixedEnv: per-reset random choice between '
                        'curriculum and regular Phase7 setup. Stronger '
                        'curriculum reward magnitudes that fire only on '
                        'curriculum episodes.')
    p.add_argument('--mix_ratio', type=float, default=0.3,
                   help='Fraction of episodes that are curriculum episodes '
                        '(only used with --mixed). Default 0.3.')
    p.add_argument('--rail_drill_share', type=float, default=0.0,
                   help='Within curriculum episodes, fraction that are '
                        'rail-frozen drills vs key+break (only used with '
                        '--mixed). v11: 0.6 to focus on rail aversion.')
    p.add_argument('--threeball_drill_share', type=float, default=0.0,
                   help='Within curriculum episodes, fraction that are '
                        '3-ball end-of-rack drills (key1 + key2 + break). '
                        'Sum of rail + threeball + railbreak share must '
                        'be ≤ 1.0.')
    p.add_argument('--railbreak_drill_share', type=float, default=0.0,
                   help='Within curriculum episodes, fraction that are '
                        'rail-ball-break drills: 14 reracked balls + 1 '
                        'break ball near a rail + cue positioned for the '
                        'break. The first shot is treated as a '
                        'post-rerack break shot so the existing scatter '
                        'bonus rewards high-force commitment. v23: 0.25 '
                        'to fix weak-force-on-rail-break-ball pattern.')
    args = p.parse_args()
    env_class = None
    env_kwargs = None
    if args.mixed:
        from train_phase9 import Phase9MixedEnv
        env_class = Phase9MixedEnv
        env_kwargs = {'mix_ratio': args.mix_ratio,
                      'rail_drill_share': args.rail_drill_share,
                      'threeball_drill_share': args.threeball_drill_share,
                      'railbreak_drill_share': args.railbreak_drill_share}
    elif args.curriculum:
        from train_phase9 import Phase9CurriculumEnv
        env_class = Phase9CurriculumEnv
    if args.distill:
        train_phase7_distill(
            num_envs=args.envs, device_name=args.device, max_iters=args.iters,
            tag=args.tag, lr=args.lr, steps_per_update=args.steps_per_update,
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, warm_start=args.warm,
            env_class=env_class, env_kwargs=env_kwargs,
            aim_noise_deg=args.aim_noise_deg,
            force_noise_pct=args.force_noise_pct, spin_noise=args.spin_noise,
            shape_bonus_max=args.shape_bonus_max,
            movement_penalty_weight=args.movement_penalty_weight,
            cue_movement_penalty_weight=args.cue_movement_penalty_weight,
            cue_ricochet_penalty_weight=args.cue_ricochet_penalty_weight,
            force_efficiency_penalty_weight=args.force_efficiency_penalty_weight,
            rail_shot_bonus_weight=args.rail_shot_bonus_weight,
            next_shape_bonus_max=args.next_shape_bonus_max,
            eor_bonus_max=args.eor_bonus_max,
            search_k=args.search_k, search_m=args.search_m,
            search_mc=args.search_mc, softmax_temp=args.softmax_temp,
            ce_weight=args.ce_weight,
            mse_force_weight=args.mse_force_weight,
            mse_spin_weight=args.mse_spin_weight,
            value_weight=args.value_weight,
            entropy_weight=args.distill_entropy,
            log_std_min=args.log_std_min,
            teacher_warm_start=args.teacher_ckpt,
            frozen_teacher_iters=args.frozen_teacher_iters,
        )
    else:
        train_phase7(
            num_envs=args.envs, device_name=args.device, max_iters=args.iters,
            tag=args.tag, lr=args.lr, steps_per_update=args.steps_per_update,
            entropy_coef=args.entropy_coef, log_std_min=args.log_std_min,
            embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
            warm_start=args.warm, env_class=env_class, env_kwargs=env_kwargs,
            aim_noise_deg=args.aim_noise_deg, force_noise_pct=args.force_noise_pct,
            spin_noise=args.spin_noise,
            shape_bonus_max=args.shape_bonus_max,
            movement_penalty_weight=args.movement_penalty_weight,
            cue_movement_penalty_weight=args.cue_movement_penalty_weight,
            cue_ricochet_penalty_weight=args.cue_ricochet_penalty_weight,
            force_efficiency_penalty_weight=args.force_efficiency_penalty_weight,
            rail_shot_bonus_weight=args.rail_shot_bonus_weight,
            next_shape_bonus_max=args.next_shape_bonus_max,
            eor_bonus_max=args.eor_bonus_max,
            search_train=args.search_train,
            search_k=args.search_k, search_m=args.search_m, search_mc=args.search_mc,
        )
