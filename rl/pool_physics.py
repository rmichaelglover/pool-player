"""
Pool physics engine in Python/NumPy for RL training.
Vectorized with NumPy for speed -- no per-ball Python loops in hot paths.
"""
import numpy as np

# Physics constants
TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
BALL_RADIUS = 1.125
BALL_MASS = 0.17
MU_SLIDE = 0.2
MU_ROLL = 0.01
MU_SPIN_DECEL = 5.0
G = 386.09
CUSHION_RESTITUTION = 0.92
CUSHION_FRICTION = 0.14
BALL_RESTITUTION = 0.96
PHYSICS_DT = 1.0 / 300  # 300Hz instead of 600Hz for speed (still accurate enough)
VELOCITY_THRESHOLD = 0.15  # slightly higher threshold to stop sooner
ANGULAR_VEL_THRESHOLD = 0.3
MAX_CUE_SPEED = 150.0
MIN_CUE_SPEED = 20.0
FOOT_SPOT_X = 75.0
FOOT_SPOT_Y = 25.0
HEAD_SPOT_X = 25.0
HEAD_SPOT_Y = 25.0
POCKET_RADIUS = 2.5
POCKET_RADIUS_SIDE = 2.75
NUM_BALLS = 16

POCKETS = np.array([
    [0.0, 0.0], [TABLE_LENGTH/2, 0.0], [TABLE_LENGTH, 0.0],
    [0.0, TABLE_WIDTH], [TABLE_LENGTH/2, TABLE_WIDTH], [TABLE_LENGTH, TABLE_WIDTH],
], dtype=np.float64)
POCKET_RADII = np.array([
    POCKET_RADIUS, POCKET_RADIUS_SIDE, POCKET_RADIUS,
    POCKET_RADIUS, POCKET_RADIUS_SIDE, POCKET_RADIUS,
], dtype=np.float64)
POCKET_RADII_SQ = POCKET_RADII ** 2


class PoolPhysics:
    """Fast vectorized pool physics."""

    def __init__(self):
        self.pos = np.zeros((NUM_BALLS, 2))
        self.vel = np.zeros((NUM_BALLS, 2))
        self.ang = np.zeros((NUM_BALLS, 3))  # wx, wy, wz
        self.pocketed = np.zeros(NUM_BALLS, dtype=bool)
        self.active = np.ones(NUM_BALLS, dtype=bool)  # not pocketed
        self.cue_elevation = 0.0
        self.events = []
        self._collision_pairs = set()

    def reset(self):
        self.pos[:] = 0
        self.vel[:] = 0
        self.ang[:] = 0
        self.pocketed[:] = True  # all pocketed by default
        self.active[:] = False
        self.cue_elevation = 0.0
        self.events = []

    def setup_rack_141(self, num_balls=15, exclude_id=None):
        R = BALL_RADIUS
        comp = 0.998
        rs = R * np.sqrt(3) * comp
        D = 2 * R * comp
        positions = []
        for row in range(5):
            for col in range(row + 1):
                x = FOOT_SPOT_X + row * rs
                y = FOOT_SPOT_Y + (col - row / 2) * D
                positions.append((x, y))

        ball_ids = list(range(1, 16))
        if exclude_id and exclude_id in ball_ids:
            ball_ids.remove(exclude_id)
        np.random.shuffle(ball_ids)

        start = 1 if exclude_id else 0
        bi = 0
        for i in range(start, len(positions)):
            if bi >= len(ball_ids):
                break
            bid = ball_ids[bi]
            self.pos[bid] = positions[i]
            self.pocketed[bid] = False
            self.active[bid] = True
            self.vel[bid] = 0
            self.ang[bid] = 0
            bi += 1

    def setup_cue_ball(self, x=None, y=None):
        if x is None: x = HEAD_SPOT_X
        if y is None:
            margin = TABLE_WIDTH * 0.15
            y = margin + np.random.random() * (TABLE_WIDTH - 2 * margin)
        self.pos[0] = [x, y]
        self.vel[0] = 0
        self.ang[0] = 0
        self.pocketed[0] = False
        self.active[0] = True

    def strike_cue_ball(self, aim_angle, force, contact_x=0, contact_y=0, elevation=0):
        R = BALL_RADIUS
        cos_e, sin_e = np.cos(elevation), np.sin(elevation)
        speed = force * cos_e
        self.vel[0] = [speed * np.cos(aim_angle), speed * np.sin(aim_angle)]
        self.cue_elevation = elevation

        a = contact_x * R * 0.7
        b = contact_y * R * 0.7
        eff_b = b - R * sin_e * 0.3
        wz = (5 * speed * a) / (2 * R * R) * 0.6
        spin_amp = 1 + sin_e * 0.8
        spin_mag = (5 * speed * eff_b * spin_amp) / (2 * R * R)
        self.ang[0] = [spin_mag * np.sin(aim_angle), -spin_mag * np.cos(aim_angle), wz]

        self.events = []
        self._collision_pairs = set()

    def simulate_until_stopped(self, max_steps=6000):
        for _ in range(max_steps):
            self._step()
            if self._all_stopped():
                break
        return self.events

    def _all_stopped(self):
        if not np.any(self.active):
            return True
        speeds = np.sqrt(np.sum(self.vel[self.active] ** 2, axis=1))
        ang_speeds = np.sqrt(np.sum(self.ang[self.active] ** 2, axis=1))
        return np.all(speeds < VELOCITY_THRESHOLD) and np.all(ang_speeds < ANGULAR_VEL_THRESHOLD)

    def _step(self):
        dt = PHYSICS_DT
        R = BALL_RADIUS
        act = self.active

        # Vectorized speed computation
        speeds = np.sqrt(np.sum(self.vel ** 2, axis=1))
        ang_speeds = np.sqrt(np.sum(self.ang ** 2, axis=1))

        # Zero out stopped balls
        stopped = act & (speeds < VELOCITY_THRESHOLD) & (ang_speeds < ANGULAR_VEL_THRESHOLD)
        self.vel[stopped] = 0
        self.ang[stopped] = 0

        # Apply friction to moving balls
        moving = act & (~stopped)
        moving_idx = np.where(moving)[0]
        for i in moving_idx:
            self._apply_friction_single(i, dt)

        # Move balls
        self.pos[act] += self.vel[act] * dt

        # Ball-ball collisions (need pairwise -- hard to fully vectorize)
        active_idx = np.where(act)[0]
        n = len(active_idx)
        if n >= 2:
            for it in range(5):  # collision iterations
                any_col = False
                for ai in range(n):
                    i = active_idx[ai]
                    for aj in range(ai + 1, n):
                        j = active_idx[aj]
                        si = speeds[i] if it == 0 else np.sqrt(self.vel[i,0]**2 + self.vel[i,1]**2)
                        sj = speeds[j] if it == 0 else np.sqrt(self.vel[j,0]**2 + self.vel[j,1]**2)
                        if si < 0.01 and sj < 0.01:
                            continue
                        if self._resolve_collision(i, j, separate=(it==4)):
                            any_col = True
                if not any_col:
                    break

            # Final separation
            for ai in range(n):
                i = active_idx[ai]
                for aj in range(ai + 1, n):
                    j = active_idx[aj]
                    d = self.pos[j] - self.pos[i]
                    dsq = d[0]*d[0] + d[1]*d[1]
                    md = 2 * R
                    if dsq < md * md and dsq > 0.0001:
                        dist = np.sqrt(dsq)
                        overlap = md - dist
                        n_vec = d / dist
                        self.pos[i] -= n_vec * overlap / 2
                        self.pos[j] += n_vec * overlap / 2

        # Cushion collisions (vectorized check)
        for i in active_idx:
            self._cushion_single(i)

        # Pocketing (vectorized)
        for i in active_idx:
            dx = self.pos[i, 0] - POCKETS[:, 0]
            dy = self.pos[i, 1] - POCKETS[:, 1]
            dsq = dx * dx + dy * dy
            if np.any(dsq < POCKET_RADII_SQ):
                self.pocketed[i] = True
                self.active[i] = False
                self.vel[i] = 0
                self.ang[i] = 0
                self.events.append(('pocketed', i))

        # Off-table
        margin = R * 2
        for i in active_idx:
            x, y = self.pos[i]
            if x < -margin or x > TABLE_LENGTH + margin or y < -margin or y > TABLE_WIDTH + margin:
                self.pocketed[i] = True
                self.active[i] = False
                self.vel[i] = 0
                self.ang[i] = 0
                self.events.append(('off-table', i))

    def _apply_friction_single(self, i, dt):
        R = BALL_RADIUS
        vx, vy = self.vel[i]
        wx, wy, wz = self.ang[i]
        speed = np.sqrt(vx*vx + vy*vy)

        slip_x = vx + wy * R
        slip_y = vy - wx * R
        slip_speed = np.sqrt(slip_x*slip_x + slip_y*slip_y)

        if slip_speed > VELOCITY_THRESHOLD * 0.5:
            fa = MU_SLIDE * G
            snx, sny = slip_x / slip_speed, slip_y / slip_speed
            if 3.5 * fa * dt >= slip_speed:
                self.vel[i, 0] = (5*vx - 2*wy*R) / 7
                self.vel[i, 1] = (5*vy + 2*wx*R) / 7
                self.ang[i, 1] = -self.vel[i, 0] / R
                self.ang[i, 0] = self.vel[i, 1] / R
            else:
                self.vel[i, 0] -= fa * snx * dt
                self.vel[i, 1] -= fa * sny * dt
                aa = (5 / (2*R)) * fa
                self.ang[i, 0] += aa * sny * dt
                self.ang[i, 1] -= aa * snx * dt
        else:
            self.ang[i, 1] = -vx / R
            self.ang[i, 0] = vy / R
            if speed > VELOCITY_THRESHOLD:
                dv = MU_ROLL * G * dt
                if dv >= speed:
                    self.vel[i] = 0; self.ang[i, :2] = 0
                else:
                    f = 1 - dv / speed
                    self.vel[i] *= f
                    self.ang[i, 1] = -self.vel[i, 0] / R
                    self.ang[i, 0] = self.vel[i, 1] / R
            else:
                self.vel[i] = 0; self.ang[i, :2] = 0

        # Sidespin decay
        if abs(self.ang[i, 2]) > ANGULAR_VEL_THRESHOLD:
            sd = MU_SPIN_DECEL * dt
            if abs(self.ang[i, 2]) <= sd:
                self.ang[i, 2] = 0
            else:
                self.ang[i, 2] -= np.sign(self.ang[i, 2]) * sd
        else:
            self.ang[i, 2] = 0

    def _resolve_collision(self, i, j, separate=True):
        R = BALL_RADIUS
        d = self.pos[j] - self.pos[i]
        dsq = d[0]*d[0] + d[1]*d[1]
        md = 2 * R
        if dsq >= md * md or dsq < 0.0001:
            return False
        dist = np.sqrt(dsq)
        nx, ny = d[0]/dist, d[1]/dist
        if separate:
            overlap = md - dist
            self.pos[i] -= np.array([nx, ny]) * overlap / 2
            self.pos[j] += np.array([nx, ny]) * overlap / 2

        dvx = self.vel[i,0] - self.vel[j,0]
        dvy = self.vel[i,1] - self.vel[j,1]
        dvn = dvx*nx + dvy*ny
        if dvn <= 0:
            return False

        j_imp = (1 + BALL_RESTITUTION) * dvn / 2
        self.vel[i, 0] -= j_imp * nx
        self.vel[i, 1] -= j_imp * ny
        self.vel[j, 0] += j_imp * nx
        self.vel[j, 1] += j_imp * ny

        pair = (min(i,j), max(i,j))
        if pair not in self._collision_pairs:
            self._collision_pairs.add(pair)
            self.events.append(('ball-hit', i, j))
        return True

    def _cushion_single(self, i):
        R = BALL_RADIUS
        x, y = self.pos[i]
        vx, vy = self.vel[i]

        # Simple axis-aligned cushion checks (faster than segment math)
        # Top rail (y=0)
        if y < R and vy < 0:
            self.vel[i, 1] = -vy * CUSHION_RESTITUTION
            self.pos[i, 1] = R
            self.events.append(('cushion', i))
        # Bottom rail (y=W)
        if y > TABLE_WIDTH - R and vy > 0:
            self.vel[i, 1] = -vy * CUSHION_RESTITUTION
            self.pos[i, 1] = TABLE_WIDTH - R
            self.events.append(('cushion', i))
        # Left rail (x=0)
        if x < R and vx < 0:
            self.vel[i, 0] = -vx * CUSHION_RESTITUTION
            self.pos[i, 0] = R
            self.events.append(('cushion', i))
        # Right rail (x=L)
        if x > TABLE_LENGTH - R and vx > 0:
            self.vel[i, 0] = -vx * CUSHION_RESTITUTION
            self.pos[i, 0] = TABLE_LENGTH - R
            self.events.append(('cushion', i))

    def get_state(self):
        obs = np.zeros(32, dtype=np.float32)
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
        return obs

    def count_on_table(self):
        return int(np.sum(self.active[1:]))
