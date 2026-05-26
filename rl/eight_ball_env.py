"""
8-ball pool environment for RL self-play.

Full APA/BCA 8-ball rules:
  - 15 object balls: solids (1-7), 8-ball (8), stripes (9-15)
  - Break: must hit head ball; >= 4 balls to rail or pocket a ball
  - Open table until first legal pocket post-break assigns groups
  - Must contact own group first; fouls give opponent ball-in-hand
  - 8-ball must be last; early pocket / scratch on 8 = loss
  - Safety: player calls safety, turn passes regardless of outcome
"""
from __future__ import annotations

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_sim import simulate_shot
from shot_enumerator import (generate_legal_shots, POCKETS, POCKET_NAMES,
                              POCKET_RADII, R, LegalShot)
from train_phase6 import RACK_APEX, RACK_POSITIONS
from train_phase6b import first_ball_struck, pocket_index_of, HEAD_SPOT
from eight_ball_net import (EightBallObs, MAX_BALLS, MAX_POCKETS, MAX_SHOTS,
                            GAME_STATE_DIM, TABLE_LENGTH, TABLE_WIDTH,
                            GROUP_CUE, GROUP_MINE, GROUP_NEUTRAL,
                            GROUP_THEIRS, GROUP_8BALL,
                            decode_force, decode_spin)

SOLIDS = set(range(1, 8))
STRIPES = set(range(9, 16))

# Game phases
BREAK = 0
OPEN_TABLE = 1
PLAYING = 2
GAME_OVER = 3

# Head string x-coordinate (cue must be behind this for break)
HEAD_STRING_X = 25.0


def _ball_group_id(ball_id):
    if ball_id == 0:
        return 'cue'
    if ball_id == 8:
        return '8ball'
    if ball_id in SOLIDS:
        return 'solids'
    return 'stripes'


def sample_eight_ball_rack():
    """Standard 8-ball rack: 15 balls in triangle at foot spot.
    8-ball in center (row 2, middle position = index 5 in RACK_POSITIONS).
    One solid and one stripe in the two back corners.
    Head ball (apex) is random. Rest random."""
    # RACK_POSITIONS indices:
    # Row 0: [0]  (apex / head ball)
    # Row 1: [1, 2]
    # Row 2: [3, 4, 5]  — index 4 is the center
    # Row 3: [6, 7, 8, 9]
    # Row 4: [10, 11, 12, 13, 14]  — 10 and 14 are the back corners
    ids = list(range(1, 16))
    random.shuffle(ids)

    slots = [None] * 15

    # Place 8-ball in center (index 4)
    ids.remove(8)
    slots[4] = 8

    # Back corners: one solid, one stripe
    available_solids = [b for b in ids if b in SOLIDS]
    available_stripes = [b for b in ids if b in STRIPES]
    corner_solid = random.choice(available_solids)
    corner_stripe = random.choice(available_stripes)
    ids.remove(corner_solid)
    ids.remove(corner_stripe)
    if random.random() < 0.5:
        slots[10], slots[14] = corner_solid, corner_stripe
    else:
        slots[10], slots[14] = corner_stripe, corner_solid

    # Fill remaining slots randomly
    random.shuffle(ids)
    j = 0
    for i in range(15):
        if slots[i] is None:
            slots[i] = ids[j]
            j += 1

    balls = {slots[i]: list(RACK_POSITIONS[i]) for i in range(15)}

    # Cue ball in the kitchen
    cx = 10.0 + random.random() * 15.0
    cy = 5.0 + random.random() * 40.0
    cue = [cx, cy]
    return cue, balls


def _place_ball_in_hand(balls, behind_head_string=False):
    """Find a good ball-in-hand placement by sampling candidates."""
    best_pos = None
    best_count = -1
    x_lo = R + 0.5
    x_hi = HEAD_STRING_X - 0.5 if behind_head_string else TABLE_LENGTH - R - 0.5
    y_lo = R + 0.5
    y_hi = TABLE_WIDTH - R - 0.5

    for _ in range(50):
        cx = x_lo + random.random() * (x_hi - x_lo)
        cy = y_lo + random.random() * (y_hi - y_lo)
        # Check no overlap with existing balls
        overlap = False
        for bx, by in balls.values():
            if math.hypot(cx - bx, cy - by) < 2.5 * R:
                overlap = True
                break
        if overlap:
            continue
        shots = generate_legal_shots((cx, cy), balls, max_cut_deg=80.0)
        if len(shots) > best_count:
            best_count = len(shots)
            best_pos = [cx, cy]

    if best_pos is None:
        best_pos = [HEAD_SPOT[0], HEAD_SPOT[1]]
    return best_pos


class EightBallEnv:
    def __init__(self, max_shots_per_game=80, opening_break_force=240.0,
                 aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0,
                 shape_reward_weight=0.05):
        self.max_shots_per_game = max_shots_per_game
        self.opening_break_force = opening_break_force
        self.aim_noise_deg = aim_noise_deg
        self.force_noise_pct = force_noise_pct
        self.spin_noise = spin_noise
        self.shape_reward_weight = shape_reward_weight
        self.reset()

    def reset(self):
        self.cue, self.balls = sample_eight_ball_rack()
        self.phase = OPEN_TABLE
        self.current_player = 0
        # groups[player] = 'solids' or 'stripes' or None
        self.groups = {0: None, 1: None}
        self.ball_in_hand = False
        self.ball_in_hand_behind_head = False
        self.winner = None
        self.total_shots = 0
        self.consecutive_fouls = [0, 0]
        self.is_safety = False
        self._execute_break()
        return self.get_obs()

    def _execute_break(self):
        """Auto-execute the opening break: aim at the head ball (apex)."""
        dx = RACK_APEX[0] - self.cue[0]
        dy = RACK_APEX[1] - self.cue[1]
        aim = math.atan2(dy, dx)
        force = self.opening_break_force
        if self.aim_noise_deg > 0:
            aim += np.random.randn() * self.aim_noise_deg * (math.pi / 180.0)
        if self.force_noise_pct > 0:
            force *= (1.0 + np.random.randn() * self.force_noise_pct)
            force = max(20.0, min(280.0, force))
        aim_dx = math.cos(aim)
        aim_dy = math.sin(aim)
        balls_in_sim = {bid: tuple(pos) for bid, pos in self.balls.items()}
        result = simulate_shot(
            tuple(self.cue), balls_in_sim,
            aim_dx * force, aim_dy * force,
            0.0, aim_dx, aim_dy,
        )
        pocketed_ids = set(result.pocketed_ids)
        if result.cue_scratched:
            # Scratch on break: opponent gets ball-in-hand behind head string
            for bid in pocketed_ids:
                if bid in self.balls:
                    del self.balls[bid]
            for bid, pos in result.final_positions.items():
                if bid in self.balls:
                    self.balls[bid] = list(pos)
            self.current_player = 1
            self.ball_in_hand = True
            self.ball_in_hand_behind_head = True
            self.cue = _place_ball_in_hand(self.balls, behind_head_string=True)
        else:
            for bid in pocketed_ids:
                if bid in self.balls:
                    del self.balls[bid]
            if 0 in result.final_positions:
                self.cue = list(result.final_positions[0])
            for bid, pos in result.final_positions.items():
                if bid in self.balls:
                    self.balls[bid] = list(pos)
            # If 8 pocketed on break, re-spot it (standard rule)
            if 8 in pocketed_ids:
                self.balls[8] = list(RACK_APEX)
            # Assign groups if any ball was pocketed (except 8)
            non_8_pocketed = pocketed_ids - {8}
            if non_8_pocketed:
                self._try_assign_groups(non_8_pocketed)
        self.total_shots += 1

    def _my_ball_ids(self, player=None):
        if player is None:
            player = self.current_player
        g = self.groups[player]
        if g is None:
            return set()
        return SOLIDS if g == 'solids' else STRIPES

    def _their_ball_ids(self, player=None):
        if player is None:
            player = self.current_player
        opp = 1 - player
        return self._my_ball_ids(opp)

    def _my_remaining(self, player=None):
        my_ids = self._my_ball_ids(player)
        return len(my_ids & set(self.balls.keys()))

    def _their_remaining(self, player=None):
        return self._my_remaining(1 - (player if player is not None else self.current_player))

    def _on_8ball(self, player=None):
        if player is None:
            player = self.current_player
        return self._my_remaining(player) == 0 and self.groups[player] is not None

    def get_legal_shots(self):
        all_shots = generate_legal_shots(self.cue, self.balls, max_cut_deg=80.0)
        if self.phase == OPEN_TABLE:
            return [s for s in all_shots if s.ball_id != 8]

        # PLAYING phase: only own group (or 8-ball if on 8).
        # If no own-group shots at 80°, widen to 89° to find steep-angle
        # shots — a real player would still aim at their own ball.
        if self._on_8ball():
            eight_shots = [s for s in all_shots if s.ball_id == 8]
            if eight_shots:
                return eight_shots
            wide = generate_legal_shots(self.cue, self.balls, max_cut_deg=89.0)
            return [s for s in wide if s.ball_id == 8]
        my_ids = self._my_ball_ids()
        own_shots = [s for s in all_shots if s.ball_id in my_ids]
        if own_shots:
            return own_shots
        wide = generate_legal_shots(self.cue, self.balls, max_cut_deg=89.0)
        return [s for s in wide if s.ball_id in my_ids]

    def get_obs(self) -> EightBallObs:
        balls_arr = np.full((MAX_BALLS, 2), -1.0, dtype=np.float32)
        ball_mask = np.zeros(MAX_BALLS, dtype=bool)
        ball_group_arr = np.zeros(MAX_BALLS, dtype=np.float32)

        # Slot 0 = cue
        balls_arr[0] = [self.cue[0] / TABLE_LENGTH, self.cue[1] / TABLE_WIDTH]
        ball_mask[0] = True
        ball_group_arr[0] = GROUP_CUE

        # Object balls in slots 1-15
        my_ids = self._my_ball_ids()
        their_ids = self._their_ball_ids()

        for i, bid in enumerate(sorted(self.balls.keys())):
            if i + 1 >= MAX_BALLS:
                break
            balls_arr[i + 1] = [self.balls[bid][0] / TABLE_LENGTH,
                                self.balls[bid][1] / TABLE_WIDTH]
            ball_mask[i + 1] = True
            if bid == 8:
                ball_group_arr[i + 1] = GROUP_8BALL
            elif self.groups[self.current_player] is None:
                ball_group_arr[i + 1] = GROUP_NEUTRAL
            elif bid in my_ids:
                ball_group_arr[i + 1] = GROUP_MINE
            else:
                ball_group_arr[i + 1] = GROUP_THEIRS

        # Pockets
        pockets_arr = np.zeros((MAX_POCKETS, 3), dtype=np.float32)
        for i, (px, py) in enumerate(POCKETS):
            pockets_arr[i] = [px / TABLE_LENGTH, py / TABLE_WIDTH,
                              1.0 if POCKET_RADII[i] < 2.6 else 0.0]

        # Game state features
        gs = np.zeros(GAME_STATE_DIM, dtype=np.float32)
        gs[0] = self._my_remaining() / 7.0
        gs[1] = self._their_remaining() / 7.0
        gs[2] = 1.0 if self.groups[self.current_player] is None else 0.0
        gs[3] = 1.0 if self._on_8ball() else 0.0
        gs[4] = 1.0 if self._on_8ball(1 - self.current_player) else 0.0
        gs[5] = 1.0 if self.ball_in_hand else 0.0
        gs[6] = 1.0 if self.phase == BREAK else 0.0
        gs[7] = min(self.consecutive_fouls[self.current_player], 3) / 3.0

        # Legal shots
        legal = self.get_legal_shots()
        legal = legal[:MAX_SHOTS]
        shots_arr = np.zeros((MAX_SHOTS, 9), dtype=np.float32)
        shot_mask = np.zeros(MAX_SHOTS, dtype=bool)
        for i, s in enumerate(legal):
            bx, by = self.balls[s.ball_id]
            pocket_pos = POCKETS[s.pocket_idx]
            shots_arr[i] = [
                s.ghost_pos[0] / TABLE_LENGTH, s.ghost_pos[1] / TABLE_WIDTH,
                bx / TABLE_LENGTH, by / TABLE_WIDTH,
                pocket_pos[0] / TABLE_LENGTH, pocket_pos[1] / TABLE_WIDTH,
                s.cut_angle_deg / 90.0,
                s.cue_to_ghost_dist / TABLE_LENGTH,
                s.ball_to_pocket_dist / TABLE_LENGTH,
            ]
            shot_mask[i] = True

        return EightBallObs(
            balls=balls_arr, ball_mask=ball_mask, ball_group=ball_group_arr,
            pockets=pockets_arr, game_state=gs,
            shots=shots_arr, shot_mask=shot_mask, shot_meta=legal,
        )

    def step(self, action_idx: int, force_raw: float, spin_raw: float,
             obs: EightBallObs, called_safety: bool = False,
             record_trajectory: bool = False, traj_max_frames: int = 600):
        if self.phase == GAME_OVER:
            return self.get_obs(), 0.0, True, {'reason': 'already over'}

        legal = obs.shot_meta
        is_safety = called_safety or (action_idx == len(legal))

        # If action_idx is the safety slot or beyond legal shots
        if action_idx >= len(legal):
            if not legal:
                # No legal shots at all — forced foul
                return self._handle_foul('no legal shots', obs)
            # Safety: aim at the easiest legal shot with soft force
            shot = min(legal, key=lambda s: s.difficulty)
            force_raw = -2.0  # soft
            spin_raw = 0.0
            is_safety = True
        else:
            shot = legal[action_idx]

        aim = shot.aim_angle
        force = decode_force(force_raw)
        spin = decode_spin(spin_raw)

        if self.aim_noise_deg > 0:
            aim += np.random.randn() * self.aim_noise_deg * (math.pi / 180.0)
        if self.force_noise_pct > 0:
            force *= (1.0 + np.random.randn() * self.force_noise_pct)
            force = max(20.0, min(280.0, force))
        if self.spin_noise > 0:
            spin += np.random.randn() * self.spin_noise
            spin = max(-2.5, min(2.5, spin))

        aim_dx = math.cos(aim)
        aim_dy = math.sin(aim)

        self.ball_in_hand = False
        self.ball_in_hand_behind_head = False

        # Determine first ball struck (for foul detection)
        first_hit_id, _ = first_ball_struck(self.cue, aim, self.balls)

        # Simulate
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
            'player': self.current_player,
            'shot': shot,
            'aim_angle': aim,
            'force': force, 'spin': spin,
            'pocketed_ids': list(pocketed_ids),
            'scratch': scratch,
            'first_hit': first_hit_id,
            'is_safety': is_safety,
        }
        if record_trajectory and result.trajectory is not None:
            info['trajectory'] = result.trajectory.tolist()
            info['trajectory_ball_ids'] = ordered_ids

        self.total_shots += 1

        # --- Check for 8-ball loss conditions ---
        if 8 in pocketed_ids:
            if not self._on_8ball():
                # Pocketed 8 too early → loss
                self.phase = GAME_OVER
                self.winner = 1 - self.current_player
                return self._game_over_obs(-1.0, {**info, 'reason': '8ball pocketed early'})
            if scratch:
                # Scratch on 8 → loss
                self.phase = GAME_OVER
                self.winner = 1 - self.current_player
                return self._game_over_obs(-1.0, {**info, 'reason': 'scratch on 8ball'})
            # Check called pocket for 8
            final_pos_8 = result.final_positions.get(8)
            if final_pos_8 is not None:
                actual_pocket = pocket_index_of(final_pos_8)
                if actual_pocket != shot.pocket_idx:
                    self.phase = GAME_OVER
                    self.winner = 1 - self.current_player
                    return self._game_over_obs(-1.0, {**info, 'reason': '8ball wrong pocket'})
            # Legal 8-ball pocket → win!
            self.phase = GAME_OVER
            self.winner = self.current_player
            return self._game_over_obs(1.0, {**info, 'reason': '8ball pocketed, win'})

        # --- Foul detection ---
        foul = False
        foul_reason = None

        if scratch:
            foul = True
            foul_reason = 'scratch'
        elif first_hit_id is None:
            foul = True
            foul_reason = 'no contact'
        elif self.phase == PLAYING and not self._is_legal_first_contact(first_hit_id):
            foul = True
            foul_reason = 'wrong ball first'
        elif not result.hit_rail and len(pocketed_ids) == 0:
            foul = True
            foul_reason = 'no rail after contact'

        # --- Apply results ---
        # Remove pocketed balls from table
        for bid in pocketed_ids:
            if bid in self.balls:
                del self.balls[bid]
        # Update positions
        if not scratch and 0 in result.final_positions:
            self.cue = list(result.final_positions[0])
        for bid, pos in result.final_positions.items():
            if bid in self.balls:
                self.balls[bid] = list(pos)

        # --- Reward computation ---
        reward = 0.0

        if foul:
            self.consecutive_fouls[self.current_player] += 1
            reward = -0.40
            info['foul'] = foul_reason
            # Opponent gets ball-in-hand
            self._switch_player(ball_in_hand=True)
        else:
            self.consecutive_fouls[self.current_player] = 0

            # Group assignment (open table, first legal pocket)
            if self.phase == OPEN_TABLE and not foul:
                self._try_assign_groups(pocketed_ids)

            my_ids = self._my_ball_ids()
            their_ids = self._their_ball_ids()
            my_pocketed = pocketed_ids & my_ids
            their_pocketed = pocketed_ids & their_ids

            reward += 0.15 * len(my_pocketed)
            reward -= 0.15 * len(their_pocketed)

            # Shape bonus
            if self.shape_reward_weight > 0 and len(self.balls) > 0:
                next_shots = self.get_legal_shots()
                if next_shots:
                    easiest = min(s.difficulty for s in next_shots)
                    ease = max(0.0, 1.0 - easiest / 5.0)
                    reward += self.shape_reward_weight * ease
                else:
                    reward -= self.shape_reward_weight

            # Check if we just cleared our last group ball
            if my_ids and len(my_ids & set(self.balls.keys())) == 0:
                reward += 0.10

            # Turn switching
            if is_safety:
                self._switch_player(ball_in_hand=False)
            elif len(my_pocketed) > 0 and not is_safety:
                pass  # same player continues
            else:
                self._switch_player(ball_in_hand=False)

        # Max shots check
        if self.total_shots >= self.max_shots_per_game:
            self.phase = GAME_OVER
            self.winner = None  # draw
            return self._game_over_obs(0.0, {**info, 'reason': 'max shots'})

        info['total_shots'] = self.total_shots
        done = (self.phase == GAME_OVER)
        return self.get_obs(), reward, done, info

    def _is_legal_first_contact(self, first_hit_id):
        if first_hit_id is None:
            return False
        if self._on_8ball():
            return first_hit_id == 8
        my_ids = self._my_ball_ids()
        if not my_ids:
            return True  # open table or no group yet
        return first_hit_id in my_ids

    def _try_assign_groups(self, pocketed_ids):
        if self.groups[self.current_player] is not None:
            return
        for bid in pocketed_ids:
            if bid in SOLIDS:
                self.groups[self.current_player] = 'solids'
                self.groups[1 - self.current_player] = 'stripes'
                self.phase = PLAYING
                return
            elif bid in STRIPES:
                self.groups[self.current_player] = 'stripes'
                self.groups[1 - self.current_player] = 'solids'
                self.phase = PLAYING
                return

    def _switch_player(self, ball_in_hand=False):
        self.current_player = 1 - self.current_player
        self.ball_in_hand = ball_in_hand
        self.ball_in_hand_behind_head = False
        if ball_in_hand:
            self.cue = _place_ball_in_hand(self.balls, behind_head_string=False)

    def _handle_foul(self, reason, obs):
        self.consecutive_fouls[self.current_player] += 1
        self._switch_player(ball_in_hand=True)
        self.total_shots += 1
        if self.total_shots >= self.max_shots_per_game:
            self.phase = GAME_OVER
            self.winner = None
            return self._game_over_obs(0.0, {'reason': reason, 'foul': reason})
        return self.get_obs(), -0.40, False, {'reason': reason, 'foul': reason,
                                               'player': 1 - self.current_player}

    def _game_over_obs(self, reward, info):
        info['winner'] = self.winner
        return self.get_obs(), reward, True, info


# ── Smoke test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    env = EightBallEnv()
    obs = env.reset()
    print(f'Initial state: {len(env.balls)} balls on table, phase={env.phase}')
    print(f'Cue at ({env.cue[0]:.1f}, {env.cue[1]:.1f})')
    print(f'Legal shots: {len(obs.shot_meta)}')
    print(f'Game state features: {obs.game_state}')

    # Play a full game with random actions
    total_reward = [0.0, 0.0]
    steps = 0
    while True:
        legal = obs.shot_meta
        if not legal:
            # Safety (no legal shots)
            action_idx = 0
        else:
            action_idx = random.randint(0, len(legal))  # includes safety
        force_raw = random.gauss(0, 1)
        spin_raw = random.gauss(0, 1)
        player = env.current_player
        obs, reward, done, info = env.step(action_idx, force_raw, spin_raw, obs)
        total_reward[player] += reward
        steps += 1
        if done:
            print(f'\nGame over after {steps} shots: {info.get("reason", "?")}')
            print(f'Winner: player {env.winner}')
            print(f'Groups: {env.groups}')
            print(f'Rewards: p0={total_reward[0]:.2f} p1={total_reward[1]:.2f}')
            print(f'Balls remaining: {len(env.balls)}')
            break

    # Run 20 quick games to check completion
    print('\n--- Running 20 random games ---')
    wins = {0: 0, 1: 0, None: 0}
    lengths = []
    for g in range(20):
        env2 = EightBallEnv()
        obs2 = env2.reset()
        for _ in range(200):
            legal2 = obs2.shot_meta
            ai = random.randint(0, max(0, len(legal2)))
            obs2, r, d, info2 = env2.step(ai, random.gauss(0, 1), random.gauss(0, 1), obs2)
            if d:
                break
        wins[env2.winner] += 1
        lengths.append(env2.total_shots)
    print(f'Wins: p0={wins[0]} p1={wins[1]} draw={wins[None]}')
    print(f'Game length: mean={np.mean(lengths):.1f} min={min(lengths)} max={max(lengths)}')
