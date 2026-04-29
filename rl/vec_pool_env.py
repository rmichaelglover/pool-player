"""
Vectorized pool environment for GPU-accelerated PPO training.

Runs N parallel 14.1 Continuous games using the C physics sim.
Each environment maps continuous actions (aim_sin, aim_cos, force,
contact_x, contact_y) to physics parameters and executes shots.

Designed for H100 training with 1000+ parallel environments.
"""
import numpy as np
import math
import random
from pool_sim import simulate_shot

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
R = 1.125
TARGET_SCORE = 25
MAX_STEPS_PER_GAME = 200

# Pocket aim points (for ghost ball computation)
_mo = 2.5 * 0.45
_smo = 2.75 * 0.15
POCKETS = [
    (_mo, _mo), (TABLE_LENGTH / 2, _smo), (TABLE_LENGTH - _mo, _mo),
    (_mo, TABLE_WIDTH - _mo), (TABLE_LENGTH / 2, TABLE_WIDTH - _smo),
    (TABLE_LENGTH - _mo, TABLE_WIDTH - _mo),
]
POCKET_RADII = [2.5, 2.75, 2.5, 2.5, 2.75, 2.5]


class PoolGame:
    """Single 14.1 Continuous game state."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cue = [R * 3 + random.random() * (TABLE_LENGTH - 6 * R),
                    R * 3 + random.random() * (TABLE_WIDTH - 6 * R)]
        self.balls = {}
        self.pocketed = set()
        self.scores = [0, 0]
        self.current_player = 0
        self.consec_fouls = [0, 0]
        self.run_length = 0
        self.step_count = 0
        self._scatter_balls()

    def _scatter_balls(self):
        placed = [self.cue]
        for bid in range(1, 16):
            for _ in range(400):
                x = R * 2 + random.random() * (TABLE_LENGTH - 4 * R)
                y = R * 2 + random.random() * (TABLE_WIDTH - 4 * R)
                if all(math.sqrt((x - px) ** 2 + (y - py) ** 2) > 2.2 * R
                       for px, py in placed):
                    self.balls[bid] = [x, y]
                    placed.append((x, y))
                    break

    def _setup_rack(self, exclude_id=None):
        comp = 0.998
        rs = R * math.sqrt(3) * comp
        D = 2 * R * comp
        positions = []
        for row in range(5):
            for col in range(row + 1):
                positions.append((75 + row * rs, 25 + (col - row / 2) * D))
        ids = list(range(1, 16))
        if exclude_id and exclude_id in ids:
            ids.remove(exclude_id)
        random.shuffle(ids)
        start = 1 if exclude_id else 0
        bi = 0
        for i in range(start, len(positions)):
            if bi >= len(ids):
                break
            bid = ids[bi]
            self.balls[bid] = list(positions[i])
            if bid in self.pocketed:
                self.pocketed.remove(bid)
            bi += 1

    def get_obs(self):
        """Build 38-dim observation matching PoolAttentionNet input."""
        obs = np.full(38, -1.0, dtype=np.float32)
        # Cue ball
        obs[0] = self.cue[0] / TABLE_LENGTH
        obs[1] = self.cue[1] / TABLE_WIDTH
        # Object balls 1-15
        for bid, (bx, by) in self.balls.items():
            if bid not in self.pocketed and bid <= 15:
                idx = bid * 2
                obs[idx] = bx / TABLE_LENGTH
                obs[idx + 1] = by / TABLE_WIDTH
        # Game state
        p = self.current_player
        obs[32] = self.scores[p] / TARGET_SCORE
        obs[33] = self.scores[1 - p] / TARGET_SCORE
        on_table = sum(1 for b in self.balls if b not in self.pocketed)
        obs[34] = on_table / 15.0
        obs[35] = self.consec_fouls[p] / 3.0
        obs[36] = 1.0 if on_table <= 1 else 0.0  # rerack flag
        obs[37] = 0.0  # reserved
        return obs

    def step(self, aim_angle, force, contact_y):
        """
        Execute one shot with physics.

        Args:
            aim_angle: radians, direction to shoot
            force: in/s, cue ball speed
            contact_y: -1 (draw) to +1 (follow), maps to spin type

        Returns:
            reward, done, info
        """
        self.step_count += 1
        p = self.current_player

        # Map contact_y to spin type for C sim
        # contact_y < -0.33 -> draw, > 0.33 -> follow, else stop
        if contact_y > 0.33:
            spin = 1  # follow
        elif contact_y < -0.33:
            spin = 2  # draw
        else:
            spin = 0  # stop

        # Build physics inputs
        aim_dx = math.cos(aim_angle)
        aim_dy = math.sin(aim_angle)
        cue_vx = aim_dx * force
        cue_vy = aim_dy * force

        active = {bid: (pos[0], pos[1]) for bid, pos in self.balls.items()
                  if bid not in self.pocketed}

        result = simulate_shot(
            tuple(self.cue), active, cue_vx, cue_vy,
            spin, aim_dx, aim_dy
        )

        # Update positions
        for bid, (fx, fy) in result.final_positions.items():
            if bid == 0:
                self.cue = [fx, fy]
            elif bid in self.balls:
                self.balls[bid] = [fx, fy]

        # Handle scratch
        scratched = result.cue_scratched
        if scratched:
            self.cue = [R * 3 + random.random() * (TABLE_LENGTH / 4),
                        R * 3 + random.random() * (TABLE_WIDTH - 6 * R)]

        # Process pocketed
        obj_pocketed = []
        for bid in result.pocketed_ids:
            if bid != 0 and bid not in self.pocketed:
                self.pocketed.add(bid)
                obj_pocketed.append(bid)

        # Foul detection
        foul = False
        if scratched:
            foul = True
        elif not result.hit_ball:
            foul = True
        elif not result.hit_rail and len(obj_pocketed) == 0:
            foul = True

        # Compute reward
        reward = 0.0

        if foul:
            self.scores[p] -= 1
            self.consec_fouls[p] += 1
            if self.consec_fouls[p] >= 3:
                self.scores[p] -= 15
                self.consec_fouls[p] = 0
                reward -= 5.0
            reward -= 1.0
            self.run_length = 0
            self.current_player = 1 - p
        elif len(obj_pocketed) > 0:
            n = len(obj_pocketed)
            self.scores[p] += n
            self.consec_fouls[p] = 0
            self.run_length += n
            # Escalating run reward
            reward += n * (1.0 + (self.run_length - 1) * 0.3)
            # Shape bonus: how easy is the next shot?
            reward += self._shape_bonus()
            # Re-rack check
            on_table = sum(1 for b in self.balls if b not in self.pocketed)
            if on_table <= 1:
                reward += 2.0  # rack cleared bonus
                remain_id = None
                for b in self.balls:
                    if b not in self.pocketed:
                        remain_id = b
                        break
                self._setup_rack(exclude_id=remain_id)
        else:
            # Missed -- no foul, no pocket
            reward -= 0.3
            self.run_length = 0
            self.current_player = 1 - p

        # Check win
        done = False
        if self.scores[p] >= TARGET_SCORE:
            reward += 10.0
            done = True
        elif self.scores[1 - p] >= TARGET_SCORE:
            reward -= 5.0
            done = True
        elif self.step_count >= MAX_STEPS_PER_GAME:
            done = True

        info = {
            'score_p1': self.scores[0],
            'score_p2': self.scores[1],
            'run_length': self.run_length,
            'foul': foul,
            'pocketed': len(obj_pocketed),
        }
        return reward, done, info

    def _shape_bonus(self):
        """Reward for good cue ball position after pocketing."""
        active = [(bid, pos) for bid, pos in self.balls.items()
                  if bid not in self.pocketed]
        if not active:
            return 0.0
        # Find easiest next shot (simplified -- just check distance)
        best = 999.0
        for bid, (bx, by) in active:
            d = math.sqrt((bx - self.cue[0]) ** 2 + (by - self.cue[1]) ** 2)
            if d < best:
                best = d
        # Closer = better shape
        if best < 15:
            return 0.5
        elif best < 30:
            return 0.2
        elif best > 60:
            return -0.3
        return 0.0


class VectorizedPoolEnv:
    """
    N parallel pool games for PPO training.

    All games run independently. When one finishes, it auto-resets.
    Observations and rewards are batched as numpy arrays.
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.games = [PoolGame() for _ in range(num_envs)]

    def reset(self):
        """Reset all environments. Returns (num_envs, 38) observations."""
        for g in self.games:
            g.reset()
        return np.stack([g.get_obs() for g in self.games])

    def step(self, actions):
        """
        Step all environments.

        Args:
            actions: (num_envs, 5) -- [aim_sin, aim_cos, force_raw, contact_x, contact_y]

        Returns:
            obs: (num_envs, 38)
            rewards: (num_envs,)
            dones: (num_envs,) bool
            infos: list of dicts
        """
        obs = np.zeros((self.num_envs, 38), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i, (game, act) in enumerate(zip(self.games, actions)):
            # Parse continuous actions
            aim_angle = np.arctan2(act[0], act[1])
            force_raw = 1.0 / (1.0 + np.exp(-float(act[2])))  # sigmoid
            force = force_raw * 70 + 10  # [10, 80] in/s
            contact_y = np.tanh(float(act[4]))  # draw/follow

            reward, done, info = game.step(aim_angle, force, contact_y)
            rewards[i] = reward
            dones[i] = done
            infos.append(info)

            if done:
                game.reset()

            obs[i] = game.get_obs()

        return obs, rewards, dones, infos


# ─── Quick test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    # Test single game
    game = PoolGame()
    obs = game.get_obs()
    print(f'Obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]')

    # Test vectorized env
    num_envs = 64
    env = VectorizedPoolEnv(num_envs)
    obs = env.reset()
    print(f'Vec obs shape: {obs.shape}')

    # Benchmark: random actions
    t0 = time.time()
    total_steps = 0
    for _ in range(100):
        actions = np.random.randn(num_envs, 5).astype(np.float32)
        obs, rewards, dones, infos = env.step(actions)
        total_steps += num_envs
    elapsed = time.time() - t0
    print(f'{total_steps} steps in {elapsed:.2f}s = {total_steps/elapsed:.0f} steps/sec')
    print(f'Sample reward: {rewards[0]:.2f}, done: {dones[0]}')
