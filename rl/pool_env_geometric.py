"""
Geometric pool environment -- no physics, pure geometry.
Instant shot evaluation: does the cue ball hit an object ball,
and does that object ball line up with a pocket?

This trains the agent to learn aiming and pocket selection
before introducing physics complexity.

~10,000x faster than the physics environment.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# Table geometry
TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
BALL_RADIUS = 1.125
MAX_CUE_SPEED = 150.0
MIN_CUE_SPEED = 20.0
HEAD_SPOT_X = 25.0
HEAD_SPOT_Y = 25.0
FOOT_SPOT_X = 75.0
FOOT_SPOT_Y = 25.0
NUM_BALLS = 16

# Pockets: (x, y, radius)
POCKETS = np.array([
    [0.0, 0.0, 2.5],
    [TABLE_LENGTH/2, 0.0, 2.75],
    [TABLE_LENGTH, 0.0, 2.5],
    [0.0, TABLE_WIDTH, 2.5],
    [TABLE_LENGTH/2, TABLE_WIDTH, 2.75],
    [TABLE_LENGTH, TABLE_WIDTH, 2.5],
])


class PoolEnvGeometric(gym.Env):
    """
    Geometric pool environment for fast RL training.

    Each step:
    1. Agent chooses aim angle and force
    2. Ray from cue ball in aim direction -- find first object ball hit
    3. Compute collision: object ball direction = ghost ball geometry
    4. Check if object ball direction aligns with any pocket
    5. If pocketed: reward. Cue ball deflects ~90deg.
    6. If miss: small reward for hitting, switch player.

    No physics simulation -- pure geometry. Instant evaluation.
    """

    def __init__(self, target_score=5, num_object_balls=15):
        super().__init__()
        self.target_score = target_score
        self.num_object_balls = num_object_balls

        # State: ball positions (32) + game state (6) = 38
        self.observation_space = spaces.Box(-1, 1, shape=(38,), dtype=np.float32)

        # Action: aim_angle (0 to 2pi), force (0 to 1)
        # Start with just 2 actions -- no spin needed for geometry phase
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([2 * np.pi, 1.0]),
            dtype=np.float32)

        self.pos = np.zeros((NUM_BALLS, 2))
        self.pocketed = np.zeros(NUM_BALLS, dtype=bool)
        self.scores = [0, 0]
        self.current_player = 0
        self.run_length = [0, 0]
        self.consecutive_fouls = [0, 0]
        self.turn_count = 0
        self.max_turns = 300

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.pocketed[:] = True
        R = BALL_RADIUS

        # Place cue ball
        margin = TABLE_WIDTH * 0.15
        self.pos[0] = [HEAD_SPOT_X, margin + np.random.random() * (TABLE_WIDTH - 2*margin)]
        self.pocketed[0] = False

        # Place object balls in a rack or random positions
        self._setup_rack()

        self.scores = [0, 0]
        self.current_player = 0
        self.run_length = [0, 0]
        self.consecutive_fouls = [0, 0]
        self.turn_count = 0

        return self._get_obs(), {}

    def _setup_rack(self):
        """Scatter balls randomly across the table.
        Some will naturally end up near pockets, giving the agent
        easy early wins. Much better for learning than a tight rack."""
        R = BALL_RADIUS
        placed = [self.pos[0].copy()]  # cue ball already placed

        ball_ids = list(range(1, 16))
        np.random.shuffle(ball_ids)
        for bid in ball_ids[:self.num_object_balls]:
            for _ in range(100):
                x = R * 3 + np.random.random() * (TABLE_LENGTH - R * 6)
                y = R * 3 + np.random.random() * (TABLE_WIDTH - R * 6)
                # Check not overlapping with already placed balls
                ok = True
                for px, py in placed:
                    if (x-px)**2 + (y-py)**2 < (3*R)**2:
                        ok = False
                        break
                if ok:
                    self.pos[bid] = [x, y]
                    self.pocketed[bid] = False
                    placed.append([x, y])
                    break

    def step(self, action):
        self.turn_count += 1
        aim_angle = float(action[0])
        force_norm = float(np.clip(action[1], 0, 1))

        reward = 0.0
        terminated = False
        info = {'pocketed': [], 'foul': None, 'hit': False}
        player = self.current_player

        if self.pocketed[0]:
            self.pos[0] = [HEAD_SPOT_X, HEAD_SPOT_Y]
            self.pocketed[0] = False

        # Cast ray from cue ball in aim direction
        cue_x, cue_y = self.pos[0]
        aim_dx = np.cos(aim_angle)
        aim_dy = np.sin(aim_angle)

        # Find first object ball hit by the cue ball ray
        hit_ball, hit_dist = self._find_first_hit(cue_x, cue_y, aim_dx, aim_dy)

        if hit_ball is None:
            # Missed everything -- foul (no contact)
            reward -= 0.3
            self.consecutive_fouls[player] += 1
            self.run_length[player] = 0
            info['foul'] = 'no_contact'
            self.current_player = 1 - player

            # No contact: put cue ball at a random spot
            R = BALL_RADIUS
            self.pos[0] = [
                R*3 + np.random.random() * (TABLE_LENGTH - R*6),
                R*3 + np.random.random() * (TABLE_WIDTH - R*6)
            ]
        else:
            info['hit'] = True
            reward += 0.2  # hit something

            # Compute collision geometry
            bx, by = self.pos[hit_ball]
            # Collision normal: cue ball center to hit ball center
            nx = bx - (cue_x + aim_dx * hit_dist)
            ny = by - (cue_y + aim_dy * hit_dist)
            n_len = np.sqrt(nx*nx + ny*ny)
            if n_len > 0.001:
                nx /= n_len
                ny /= n_len
            else:
                nx, ny = aim_dx, aim_dy

            # Object ball moves in the normal direction
            obj_dx, obj_dy = nx, ny

            # Check if object ball direction leads to a pocket
            pocketed_in = self._check_pocket(bx, by, obj_dx, obj_dy)

            if pocketed_in >= 0:
                # Pocketed!
                self.pocketed[hit_ball] = True
                self.scores[player] += 1
                self.run_length[player] += 1
                info['pocketed'] = [hit_ball]

                # Escalating run reward -- pocketing is 5x the hit reward
                BIG_REWARD = 5.0
                multiplier = 1.0 + (self.run_length[player] - 1) * 0.5
                reward += BIG_REWARD * multiplier

                self.consecutive_fouls[player] = 0

                # Check win
                if self.scores[player] >= self.target_score:
                    reward += 5.0
                    terminated = True
                    info['winner'] = player

                # Check re-rack
                on_table = sum(1 for i in range(1, 16) if not self.pocketed[i])
                if on_table <= 1:
                    self._do_rerack()
            else:
                # Hit but didn't pocket -- reward for proximity to pocket
                best_prox = self._pocket_proximity(bx, by, obj_dx, obj_dy)
                reward += best_prox * 0.3  # 0 to 0.3 based on how close

                # Move the contacted ball to a random position (no physics)
                R = BALL_RADIUS
                for _ in range(50):
                    nx2 = R*3 + np.random.random() * (TABLE_LENGTH - R*6)
                    ny2 = R*3 + np.random.random() * (TABLE_WIDTH - R*6)
                    ok = True
                    for b in range(NUM_BALLS):
                        if self.pocketed[b] or b == hit_ball:
                            continue
                        if (self.pos[b,0]-nx2)**2 + (self.pos[b,1]-ny2)**2 < (3*R)**2:
                            ok = False; break
                    if ok:
                        self.pos[hit_ball] = [nx2, ny2]
                        break

                # Miss: reset run, switch
                self.run_length[player] = 0
                self.current_player = 1 - player

            # Leave cue ball at the contact point
            R = BALL_RADIUS
            contact_x = cue_x + aim_dx * (hit_dist - 2 * R)
            contact_y = cue_y + aim_dy * (hit_dist - 2 * R)
            self.pos[0] = [
                np.clip(contact_x, R, TABLE_LENGTH - R),
                np.clip(contact_y, R, TABLE_WIDTH - R)
            ]

        info['score'] = self.scores[player]
        truncated = self.turn_count >= self.max_turns
        reward = np.clip(reward, -5.0, 30.0)

        return self._get_obs(), float(reward), terminated, truncated, info

    def _find_first_hit(self, cx, cy, dx, dy):
        """Find the first object ball hit by a ray from (cx,cy) in direction (dx,dy).
        Returns (ball_id, distance) or (None, None)."""
        R = BALL_RADIUS
        best_ball = None
        best_dist = float('inf')

        for i in range(1, NUM_BALLS):
            if self.pocketed[i]:
                continue
            bx, by = self.pos[i]
            # Vector from ray origin to ball center
            ocx = bx - cx
            ocy = by - cy
            # Project onto ray direction
            proj = ocx * dx + ocy * dy
            if proj < 0:
                continue  # ball is behind
            # Perpendicular distance
            perp_sq = ocx*ocx + ocy*ocy - proj*proj
            hit_radius = 2 * R  # cue ball radius + object ball radius
            if perp_sq > hit_radius * hit_radius:
                continue  # miss
            # Distance to contact point
            dist = proj - np.sqrt(max(0, hit_radius*hit_radius - perp_sq))
            if dist < best_dist and dist > 0:
                best_dist = dist
                best_ball = i

        return best_ball, best_dist if best_ball else None

    def _check_pocket(self, bx, by, obj_dx, obj_dy):
        """Check if a ball at (bx,by) moving in direction (obj_dx,obj_dy) enters a pocket.
        Returns pocket index or -1."""
        for p in range(6):
            px, py, pr = POCKETS[p]
            # Vector to pocket
            tpx = px - bx
            tpy = py - by
            tp_dist = np.sqrt(tpx*tpx + tpy*tpy)
            if tp_dist < 0.1:
                return p  # ball is already at the pocket

            # Angle between ball direction and pocket direction
            tp_nx = tpx / tp_dist
            tp_ny = tpy / tp_dist
            dot = obj_dx * tp_nx + obj_dy * tp_ny

            # The ball needs to be heading roughly toward the pocket
            # and the perpendicular miss distance must be within pocket radius
            if dot < 0.3:
                continue  # not heading toward this pocket

            # Perpendicular distance from pocket center to the ball's path
            perp = abs(-tpx * obj_dy + tpy * obj_dx)

            # Accept if perpendicular distance < pocket radius
            # Scale acceptance by distance (closer = more forgiving)
            effective_radius = pr * min(1.0, 20.0 / max(tp_dist, 1))
            if perp < effective_radius:
                # Check path isn't blocked by other balls
                blocked = False
                for b in range(1, NUM_BALLS):
                    if self.pocketed[b]:
                        continue
                    if self.pos[b, 0] == bx and self.pos[b, 1] == by:
                        continue  # skip self
                    obx = self.pos[b, 0] - bx
                    oby = self.pos[b, 1] - by
                    ob_proj = obx * obj_dx + oby * obj_dy
                    if ob_proj < 0 or ob_proj > tp_dist:
                        continue
                    ob_perp = abs(-obx * obj_dy + oby * obj_dx)
                    if ob_perp < 2 * BALL_RADIUS:
                        blocked = True
                        break
                if not blocked:
                    return p

        return -1

    def _pocket_proximity(self, bx, by, obj_dx, obj_dy):
        """Score 0-1 for how close the ball direction is to any pocket."""
        best = 0.0
        for p in range(6):
            px, py, pr = POCKETS[p]
            tpx = px - bx
            tpy = py - by
            tp_dist = np.sqrt(tpx*tpx + tpy*tpy)
            if tp_dist < 0.1:
                return 1.0
            tp_nx = tpx / tp_dist
            tp_ny = tpy / tp_dist
            dot = obj_dx * tp_nx + obj_dy * tp_ny
            if dot > 0:
                # How close to perfect alignment (dot=1)?
                score = max(0, dot - 0.5) * 2  # 0.5->0, 1.0->1.0
                best = max(best, score)
        return best

    def _move_cue_to_cushion(self, dx, dy):
        """Move cue ball in direction until it hits a cushion."""
        R = BALL_RADIUS
        x, y = self.pos[0]
        # Simple: advance until hitting a wall
        for _ in range(200):
            x += dx * 0.5
            y += dy * 0.5
            if x < R: x = R; dx = -dx
            if x > TABLE_LENGTH - R: x = TABLE_LENGTH - R; dx = -dx
            if y < R: y = R; dy = -dy
            if y > TABLE_WIDTH - R: y = TABLE_WIDTH - R; dy = -dy
            # Stop after first bounce
            break
        self.pos[0] = [x + dx * 20, y + dy * 20]
        self.pos[0, 0] = np.clip(self.pos[0, 0], R, TABLE_LENGTH - R)
        self.pos[0, 1] = np.clip(self.pos[0, 1], R, TABLE_WIDTH - R)

    def _do_rerack(self):
        """Re-rack after clearing most balls."""
        remaining_id = None
        for b in range(1, 16):
            if not self.pocketed[b]:
                remaining_id = b
                break
        self._setup_rack()
        if remaining_id:
            self.pocketed[remaining_id] = True  # it was the remaining ball

    def _get_obs(self):
        obs = np.zeros(38, dtype=np.float32)
        if not self.pocketed[0]:
            obs[0] = self.pos[0, 0] / TABLE_LENGTH
            obs[1] = self.pos[0, 1] / TABLE_WIDTH
        else:
            obs[0] = obs[1] = -1
        for b in range(1, 16):
            idx = 2 + (b-1) * 2
            if not self.pocketed[b]:
                obs[idx] = self.pos[b, 0] / TABLE_LENGTH
                obs[idx+1] = self.pos[b, 1] / TABLE_WIDTH
            else:
                obs[idx] = obs[idx+1] = -1
        obs[32] = self.scores[self.current_player] / max(self.target_score, 1)
        obs[33] = self.scores[1 - self.current_player] / max(self.target_score, 1)
        obs[34] = sum(1 for i in range(1,16) if not self.pocketed[i]) / 15.0
        obs[35] = self.consecutive_fouls[self.current_player] / 3.0
        obs[36] = 1.0 if sum(1 for i in range(1,16) if not self.pocketed[i]) <= 1 else 0.0
        obs[37] = float(self.current_player)
        return obs
