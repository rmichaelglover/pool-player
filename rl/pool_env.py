"""
Gymnasium environment for 14.1 Continuous pool.
Each step = one shot. The agent provides shot parameters,
physics simulates the result, reward is computed.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Fallback for older gym
    import gym
    from gym import spaces

from pool_physics import (
    PoolPhysics, TABLE_LENGTH, TABLE_WIDTH, BALL_RADIUS,
    MAX_CUE_SPEED, MIN_CUE_SPEED, HEAD_SPOT_X, HEAD_SPOT_Y,
    FOOT_SPOT_X, FOOT_SPOT_Y, NUM_BALLS, POCKETS, POCKET_RADII
)


class PoolEnv(gym.Env):
    """14.1 Continuous pool environment for RL training."""

    metadata = {'render_modes': []}

    def __init__(self, target_score=10, curriculum_balls=15):
        super().__init__()
        self.physics = PoolPhysics()
        self.target_score = target_score
        self.curriculum_balls = curriculum_balls  # how many balls to use

        # Observation: ball positions (32) + scores (2) + meta (4) = 38
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(38,), dtype=np.float32)

        # Action: aim_angle, force, contact_x, contact_y, elevation
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0]),
            high=np.array([2 * np.pi, 1.0, 1.0, 1.0, 0.5]),
            dtype=np.float32)

        # Game state
        self.scores = [0, 0]
        self.current_player = 0
        self.consecutive_fouls = [0, 0]
        self.run_length = [0, 0]  # consecutive balls pocketed (for escalating reward)
        self.turn_count = 0
        self.max_turns = 200  # prevent infinite games

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.physics.reset()
        self.physics.setup_cue_ball()

        # Set up rack with curriculum_balls
        if self.curriculum_balls >= 15:
            self.physics.setup_rack_141()
        else:
            # Subset: place random balls in easy positions
            self._setup_curriculum_balls()

        self.scores = [0, 0]
        self.current_player = 0
        self.consecutive_fouls = [0, 0]
        self.run_length = [0, 0]
        self.turn_count = 0

        return self._get_obs(), {}

    def _setup_curriculum_balls(self):
        """Place a subset of balls in various positions for easier learning."""
        n = self.curriculum_balls
        R = BALL_RADIUS
        for i in range(1, n + 1):
            # Random position on the table (not too close to cushions or pockets)
            while True:
                x = R * 3 + np.random.random() * (TABLE_LENGTH - R * 6)
                y = R * 3 + np.random.random() * (TABLE_WIDTH - R * 6)
                # Check not overlapping with other balls
                ok = True
                for j in range(i):
                    dx = self.physics.pos[j, 0] - x
                    dy = self.physics.pos[j, 1] - y
                    if dx*dx + dy*dy < (2.5 * R) ** 2:
                        ok = False
                        break
                if ok:
                    break
            self.physics.pos[i] = [x, y]
            self.physics.pocketed[i] = False
        # Mark remaining balls as pocketed
        for i in range(n + 1, 16):
            self.physics.pocketed[i] = True

    def step(self, action):
        """Execute one shot and return (obs, reward, terminated, truncated, info)."""
        self.turn_count += 1

        # Decode action
        aim_angle = float(action[0])
        force_norm = float(np.clip(action[1], 0, 1))
        contact_x = float(np.clip(action[2], -1, 1))
        contact_y = float(np.clip(action[3], -1, 1))
        elevation = float(np.clip(action[4], 0, 0.5))

        force = MIN_CUE_SPEED + force_norm * (MAX_CUE_SPEED - MIN_CUE_SPEED)

        # Check if cue ball is on the table
        if self.physics.pocketed[0]:
            self.physics.setup_cue_ball()

        # Execute shot
        self.physics.strike_cue_ball(aim_angle, force, contact_x, contact_y, elevation)
        events = self.physics.simulate_until_stopped()

        # Analyze results
        reward, terminated, info = self._evaluate_shot(events)

        truncated = self.turn_count >= self.max_turns

        return self._get_obs(), reward, terminated, truncated, info

    def _evaluate_shot(self, events):
        """Evaluate the shot result and compute reward.

        Dense reward shaping for learning:
        - Hitting any ball: small positive (teaches aiming)
        - Hitting ball near a pocket: medium positive (teaches geometry)
        - Pocketing a ball: large positive
        - Fouls: moderate negative (not so harsh that it prevents exploration)
        """
        reward = 0.0
        terminated = False
        info = {'pocketed': [], 'foul': None, 'score': self.scores[self.current_player]}
        player = self.current_player

        # Count events
        pocketed_balls = [e[1] for e in events if e[0] == 'pocketed']
        cue_scratched = 0 in pocketed_balls
        obj_pocketed = [b for b in pocketed_balls if b != 0]
        info['pocketed'] = obj_pocketed

        ball_hits = [e for e in events if e[0] == 'ball-hit']
        cue_hits = [e for e in ball_hits if e[1] == 0 or e[2] == 0]
        no_contact = len(cue_hits) == 0

        cushion_hits = [e for e in events if e[0] == 'cushion']
        no_rail = len(cushion_hits) == 0 and len(obj_pocketed) == 0 and not no_contact

        # -- Dense shaping rewards (always applied) --

        # Reward for hitting ANY object ball (teaches aiming)
        if len(cue_hits) > 0:
            reward += 0.2  # good: you hit something

            # Extra reward if the hit ball ended up near a pocket
            for hit_event in cue_hits:
                hit_ball_id = hit_event[2] if hit_event[1] == 0 else hit_event[1]
                if hit_ball_id > 0 and not self.physics.pocketed[hit_ball_id]:
                    bx, by = self.physics.pos[hit_ball_id]
                    for p in range(6):
                        dx = bx - POCKETS[p, 0]
                        dy = by - POCKETS[p, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 10:
                            reward += 0.1 * (1 - dist/10)  # closer to pocket = more reward
                            break

        # -- Foul detection --
        foul = False
        if cue_scratched:
            foul = True
            info['foul'] = 'scratch'
            reward -= 0.5
        elif no_contact:
            foul = True
            info['foul'] = 'no_contact'
            reward -= 0.3  # mild: just missed everything
        elif no_rail:
            foul = True
            info['foul'] = 'no_rail'
            reward -= 0.3

        if foul:
            self.scores[player] -= 1
            self.consecutive_fouls[player] += 1
            self.run_length[player] = 0  # reset run on foul

            if cue_scratched:
                self.physics.pocketed[0] = False
                self.physics.setup_cue_ball(HEAD_SPOT_X, HEAD_SPOT_Y)

            self.current_player = 1 - self.current_player
        else:
            # Legal shot
            self.consecutive_fouls[player] = 0

            if len(obj_pocketed) > 0:
                self.scores[player] += len(obj_pocketed)

                # Escalating run reward: 1x, 1.5x, 2x, 2.5x, ...
                # Each consecutive pocket in a run increases the multiplier.
                # This teaches run-building: keep pocketing = much bigger rewards.
                BIG_REWARD = 2.0
                for _ in range(len(obj_pocketed)):
                    self.run_length[player] += 1
                    multiplier = 1.0 + (self.run_length[player] - 1) * 0.5
                    reward += BIG_REWARD * multiplier

                # Position bonus (encourages good cue ball placement for next shot)
                reward += self._position_quality() * 0.5

                # Re-rack check
                on_table = self.physics.count_on_table()
                if on_table <= 1:
                    if on_table == 1:
                        for b in range(1, 16):
                            if not self.physics.pocketed[b]:
                                dx = self.physics.pos[b, 0] - FOOT_SPOT_X
                                dy = self.physics.pos[b, 1] - FOOT_SPOT_Y
                                if np.sqrt(dx*dx + dy*dy) < 20:
                                    reward += 1.0  # break ball bonus
                                break
                    self._do_rerack()

                if self.scores[player] >= self.target_score:
                    reward += 5.0
                    terminated = True
                    info['winner'] = player
            else:
                # Miss: reset run, switch player
                self.run_length[player] = 0
                self.current_player = 1 - self.current_player

        info['score'] = self.scores[player]
        # Clip reward to prevent value function divergence
        # Allow higher positive for run bonuses (10-ball run = 2 * 5.5 = 11)
        reward = np.clip(reward, -5.0, 15.0)
        return reward, terminated, info

    def _position_quality(self):
        """Estimate how good the cue ball position is for the next shot."""
        if self.physics.pocketed[0]:
            return 0.0

        cx, cy = self.physics.pos[0]
        best = 0.0
        R = BALL_RADIUS

        for b in range(1, 16):
            if self.physics.pocketed[b]:
                continue
            bx, by = self.physics.pos[b]
            dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
            if dist < 5 * R:
                continue  # too close

            # Check if any pocket is accessible from this ball
            for p_idx in range(6):
                px, py = self.physics.pos[b, 0], self.physics.pos[b, 1]  # ball pos
                # Simple scoring: closer ball + closer pocket = better
                ball_pocket_dist = np.sqrt(
                    (px - self.physics.pos[0, 0])**2 + (py - self.physics.pos[0, 1])**2)
                if ball_pocket_dist < 40:
                    score = 1.0 - ball_pocket_dist / 40.0
                    best = max(best, score)
                    break

        return best

    def _do_rerack(self):
        """Re-rack after 14 balls pocketed."""
        remaining_id = None
        for b in range(1, 16):
            if not self.physics.pocketed[b]:
                remaining_id = b
                break

        # Check if remaining ball or cue ball is in rack area
        R = BALL_RADIUS
        if remaining_id:
            dx = self.physics.pos[remaining_id, 0] - FOOT_SPOT_X
            dy = self.physics.pos[remaining_id, 1] - FOOT_SPOT_Y
            if np.sqrt(dx*dx + dy*dy) < R * 6:
                self.physics.pos[remaining_id] = [HEAD_SPOT_X, HEAD_SPOT_Y]

        if not self.physics.pocketed[0]:
            dx = self.physics.pos[0, 0] - FOOT_SPOT_X
            dy = self.physics.pos[0, 1] - FOOT_SPOT_Y
            if np.sqrt(dx*dx + dy*dy) < R * 6:
                self.physics.pos[0] = [HEAD_SPOT_X, TABLE_WIDTH / 2]

        self.physics.setup_rack_141(exclude_id=remaining_id)

    def _get_obs(self):
        """Build the observation vector."""
        obs = np.zeros(38, dtype=np.float32)

        # Ball positions (32 dims)
        pos_obs = self.physics.get_state()
        obs[:32] = pos_obs

        # Scores normalized (2 dims)
        obs[32] = self.scores[self.current_player] / max(self.target_score, 1)
        obs[33] = self.scores[1 - self.current_player] / max(self.target_score, 1)

        # Balls on table normalized (1 dim)
        obs[34] = self.physics.count_on_table() / 15.0

        # Consecutive fouls (1 dim)
        obs[35] = self.consecutive_fouls[self.current_player] / 3.0

        # Re-rack flag (1 dim)
        obs[36] = 1.0 if self.physics.count_on_table() <= 1 else 0.0

        # Current player (1 dim)
        obs[37] = float(self.current_player)

        return obs


# Register the environment
if hasattr(gym, 'register'):
    gym.register(
        id='Pool141-v0',
        entry_point='pool_env:PoolEnv',
    )
