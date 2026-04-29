"""
Single-player run practice with PPO + physics.

One player pockets balls in sequence, learning position play through
(ball, spin, speed) decisions. No opponent, no safeties — just pocket
balls and play shape. Run length is the primary metric.

The network controls:
  - Which candidate ball (3 diversity-aware slots + pass)
  - Spin (stop / follow / draw)
  - Speed (soft / medium / hard)

Physics determines the actual cue ball outcome, so the network learns
real spin+speed→position relationships.
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import math, time, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sequence import (
    R, TL, TW, POCKETS, NUM_POCKETS,
    best_pocket_for_ball, compute_ghost, cut_angle, is_path_clear,
    pocket_approach_ok, estimate_cue_position
)
from pool_sim import simulate_shot

NUM_BALLS = 15
NUM_CANDIDATES = 4                     # safe / breaker / position / key ball
N_SHOT_ACTIONS = NUM_CANDIDATES + 1   # 4 candidates + pass
N_SPIN_ACTIONS = 3                     # stop / follow / draw
N_SPEED_ACTIONS = 3                    # soft / medium / hard
SPEED_LEVELS = [18, 40, 70]
PER_CAND_FEATS = 12
GLOBAL_FEATS = 4                       # cue x/y, run_length, balls_on_table
INPUT_DIM = GLOBAL_FEATS + NUM_CANDIDATES * PER_CAND_FEATS   # 52
MAX_SHOTS = 50                         # max shots per run attempt
RACK_CENTER = (80.0, 25.0)

# PPO hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
K_EPOCHS = 4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02       # shot + speed entropy
SPIN_ENTROPY_COEF = 0.10  # higher for spin — prevents "always stop" collapse
LR = 3e-4
MAX_GRAD_NORM = 0.5


class RunNet(nn.Module):
    """Actor-Critic for single-player run practice.
    Three action heads: shot (which ball), spin (stop/follow/draw), speed (soft/med/hard).
    """
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.shot_head = nn.Linear(64, N_SHOT_ACTIONS)
        self.spin_head = nn.Linear(64, N_SPIN_ACTIONS)
        self.speed_head = nn.Linear(64, N_SPEED_ACTIONS)
        self.value_head = nn.Linear(64, 1)
        with torch.no_grad():
            self.shot_head.weight.zero_()
            self.shot_head.bias.zero_()
            self.spin_head.weight.zero_()
            self.spin_head.bias.zero_()
            self.speed_head.weight.zero_()
            self.speed_head.bias.zero_()

    def forward(self, x):
        h = self.trunk(x)
        return (self.shot_head(h), self.spin_head(h),
                self.speed_head(h), self.value_head(h).squeeze(-1))


# For probe compatibility
StrategyNet = RunNet
ActorCritic = RunNet


# ─── Table ─────────────────────────────────────────────────────────────────

class RunTable:
    """Single-player table for run practice."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cue = [R * 3 + random.random() * (TL - 6 * R),
                    R * 3 + random.random() * (TW - 6 * R)]
        self.balls = {}
        self.pocketed = set()
        self.run_length = 0
        # Mixed environment: 70% scattered (learn position control with soft/medium),
        # 30% partial rack (learn rack-breaking with hard speed).
        if random.random() < 0.7:
            self._scatter_balls()
        else:
            self._partial_rack()

    def _partial_rack(self):
        """Start with a partially broken rack: 10-12 balls in tight triangle,
        3-5 balls scattered loose. Simulates a real 14.1 table mid-game where
        only a few balls are pocketable and position play is essential."""
        # First, place all 15 in a tight rack
        comp = 0.998
        rs = R * math.sqrt(3) * comp
        D = 2 * R * comp
        positions = []
        for row in range(5):
            for col in range(row + 1):
                positions.append((75 + row * rs, 25 + (col - row / 2) * D))
        ids = list(range(1, 16))
        random.shuffle(ids)
        for i, bid in enumerate(ids):
            self.balls[bid] = list(positions[i])

        # Now scatter 3-5 random balls away from the rack
        n_loose = random.randint(3, 5)
        loose_ids = random.sample(ids, n_loose)
        placed = [self.cue]
        # Collect rack ball positions as obstacles for placement
        for bid in ids:
            if bid not in loose_ids:
                placed.append(self.balls[bid])
        for bid in loose_ids:
            for _ in range(400):
                x = R * 2 + random.random() * (TL - 4 * R)
                y = R * 2 + random.random() * (TW - 4 * R)
                # Keep away from the rack area AND from other placed balls
                in_rack = (73 < x < 84 and 19 < y < 31)
                if not in_rack and all(math.sqrt((x - px) ** 2 + (y - py) ** 2) > 2.5 * R
                                       for px, py in placed):
                    self.balls[bid] = [x, y]
                    placed.append((x, y))
                    break

    def _scatter_balls(self):
        placed = [self.cue]
        for bid in range(1, 16):
            for _ in range(400):
                x = R * 2 + random.random() * (TL - 4 * R)
                y = R * 2 + random.random() * (TW - 4 * R)
                if all(math.sqrt((x - px) ** 2 + (y - py) ** 2) > 2.2 * R
                       for px, py in placed):
                    self.balls[bid] = [x, y]
                    placed.append((x, y))
                    break

    def get_active_balls(self):
        return [(bid, pos) for bid, pos in self.balls.items()
                if bid not in self.pocketed]

    def execute_shot(self, ball_id, spin, speed_level, ghost, aim_dir):
        """Physics-based shot. Returns (pocketed_target, all_pocketed, scratched,
        hit_ball, hit_rail)."""
        active = {bid: (pos[0], pos[1]) for bid, pos in self.balls.items()
                  if bid not in self.pocketed}
        force = SPEED_LEVELS[speed_level]
        result = simulate_shot(
            tuple(self.cue), active,
            aim_dir[0] * force, aim_dir[1] * force,
            spin, aim_dir[0], aim_dir[1]
        )
        for bid, (fx, fy) in result.final_positions.items():
            if bid == 0:
                self.cue = [fx, fy]
            elif bid in self.balls:
                self.balls[bid] = [fx, fy]
        if result.cue_scratched:
            self.cue = [R * 3 + random.random() * (TL / 4),
                        R * 3 + random.random() * (TW - 6 * R)]
        obj_pocketed = []
        for bid in result.pocketed_ids:
            if bid != 0 and bid not in self.pocketed:
                self.pocketed.add(bid)
                obj_pocketed.append(bid)
        target_pocketed = ball_id in result.pocketed_ids
        # Re-rack when 1 or fewer balls remain (14.1 continuous)
        on_table = sum(1 for b in self.balls if b not in self.pocketed)
        if on_table <= 1:
            remain_id = None
            for b in self.balls:
                if b not in self.pocketed:
                    remain_id = b
                    break
            self._rerack(exclude_id=remain_id)
        return (target_pocketed, obj_pocketed, result.cue_scratched,
                result.hit_ball, result.hit_rail)

    def _rerack(self, exclude_id=None):
        """Re-rack as a partial rack: 14 balls in tight triangle, then knock
        3-5 loose. Same challenge every rack cycle — navigate with few balls."""
        # Place all 14 in tight triangle
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
        # Now scatter 3-5 balls loose (same as initial _partial_rack)
        racked_ids = [bid for bid in ids if bid != exclude_id]
        n_loose = random.randint(3, 5)
        loose_ids = random.sample(racked_ids[:len(racked_ids)], min(n_loose, len(racked_ids)))
        placed = [self.cue]
        if exclude_id and exclude_id in self.balls:
            placed.append(self.balls[exclude_id])
        for bid in racked_ids:
            if bid not in loose_ids:
                placed.append(self.balls[bid])
        for bid in loose_ids:
            for _ in range(400):
                x = R * 2 + random.random() * (TL - 4 * R)
                y = R * 2 + random.random() * (TW - 4 * R)
                in_rack = (73 < x < 84 and 19 < y < 31)
                if not in_rack and all(math.sqrt((x - px) ** 2 + (y - py) ** 2) > 2.5 * R
                                       for px, py in placed):
                    self.balls[bid] = [x, y]
                    placed.append((x, y))
                    break


# ─── Helpers ───────────────────────────────────────────────────────────────

def cluster_count(ball_pos, active, radius=5 * R):
    cnt = 0
    for bid, (bx, by) in active:
        d = math.sqrt((bx - ball_pos[0]) ** 2 + (by - ball_pos[1]) ** 2)
        if 0.1 < d < radius:
            cnt += 1
    return cnt


def best_next_difficulty(cue_pos, active):
    best_diff = 999.0
    for bid, bpos in active:
        if bid > 15:
            continue
        result = best_pocket_for_ball(cue_pos, bpos, bid, active)
        if result is None:
            continue
        pidx, ca, ghost, pdir = result
        dx, dy = ghost[0] - cue_pos[0], ghost[1] - cue_pos[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 0.1:
            continue
        px, py = POCKETS[pidx][0].item(), POCKETS[pidx][1].item()
        pd = math.sqrt((px - bpos[0]) ** 2 + (py - bpos[1]) ** 2)
        diff = ca / 75.0 + (d + pd) / (TL * 1.5)
        if diff < best_diff:
            best_diff = diff
    return best_diff


def build_candidates(table):
    """Build 3 diversity-aware candidates + observation vector.
    Spin is NOT pre-selected — the network chooses spin.
    Features describe each candidate WITHOUT a spin commitment."""
    active = table.get_active_balls()
    raw = []
    for bid, bpos in active:
        if bid > 15:
            continue
        result = best_pocket_for_ball(table.cue, bpos, bid, active)
        if result is None:
            continue
        pidx, ca, ghost, pdir = result
        dx, dy = ghost[0] - table.cue[0], ghost[1] - table.cue[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 0.1:
            continue
        aim_dir = (dx / d, dy / d)
        px, py = POCKETS[pidx][0].item(), POCKETS[pidx][1].item()
        pd = math.sqrt((px - bpos[0]) ** 2 + (py - bpos[1]) ** 2)
        difficulty = ca / 75.0 + (d + pd) / (TL * 1.5)
        raw.append({
            'bid': bid, 'bpos': bpos, 'pidx': pidx, 'ca': ca,
            'ghost': ghost, 'pdir': pdir, 'aim_dir': aim_dir,
            'difficulty': difficulty, 'cue_to_ball': d, 'ball_to_pocket': pd,
        })

    makeable = [c for c in raw if c['difficulty'] <= 2.0]
    # Pre-compute cluster + lookahead for each makeable ball
    for c in makeable:
        c['cluster'] = cluster_count(c['bpos'], active)
        # Compute average reach across all 3 spins (since network picks spin)
        total_reach = 0
        min_next = 2.0
        active_minus = [(bid, bp) for bid, bp in active if bid != c['bid']]
        for spin in range(3):
            cue_after = estimate_cue_position(c['ghost'], c['aim_dir'], c['pdir'], spin)
            cue_after = (max(R, min(TL - R, cue_after[0])),
                         max(R, min(TW - R, cue_after[1])))
            for bid2, bpos2 in active_minus:
                res = best_pocket_for_ball(cue_after, bpos2, bid2, active_minus)
                if res is not None:
                    total_reach += 1
                    pidx2, ca2, ghost2, _ = res
                    dx2 = ghost2[0] - cue_after[0]
                    dy2 = ghost2[1] - cue_after[1]
                    d2 = math.sqrt(dx2*dx2 + dy2*dy2)
                    px2, py2 = POCKETS[pidx2][0].item(), POCKETS[pidx2][1].item()
                    pd2 = math.sqrt((px2-bpos2[0])**2 + (py2-bpos2[1])**2)
                    diff2 = ca2/75.0 + (d2+pd2)/(TL*1.5)
                    if diff2 < min_next:
                        min_next = diff2
        c['avg_reach'] = total_reach / 3.0
        c['best_next_diff'] = min_next

    # Diversity-aware slot selection
    candidates = []
    used_ids = set()
    if makeable:
        safe_sorted = sorted(makeable, key=lambda c: c['difficulty'])
        for c in safe_sorted:
            if c['bid'] not in used_ids:
                candidates.append(c); used_ids.add(c['bid']); break
        breaker_sorted = sorted(makeable, key=lambda c: (-c['cluster'], c['difficulty']))
        for c in breaker_sorted:
            if c['bid'] not in used_ids:
                candidates.append(c); used_ids.add(c['bid']); break
        position_sorted = sorted(makeable, key=lambda c: (c['best_next_diff'], c['difficulty']))
        for c in position_sorted:
            if c['bid'] not in used_ids:
                candidates.append(c); used_ids.add(c['bid']); break
        # Slot 3 "key ball": ball near the rack but NOT inside it.
        # When pocketed with the right spin/speed, cue ball can break the rack.
        RACK_X_MIN, RACK_X_MAX = 73, 84
        RACK_Y_MIN, RACK_Y_MAX = 19, 31
        key_candidates = []
        for c in makeable:
            if c['bid'] in used_ids:
                continue
            bx, by = c['bpos']
            in_rack = (RACK_X_MIN < bx < RACK_X_MAX and RACK_Y_MIN < by < RACK_Y_MAX)
            if in_rack:
                continue  # skip balls inside the tight pack
            rack_dist = math.sqrt((bx - RACK_CENTER[0]) ** 2 + (by - RACK_CENTER[1]) ** 2)
            if rack_dist < 25:  # within ~25 inches of rack center
                key_candidates.append((rack_dist, c))
        if key_candidates:
            key_candidates.sort(key=lambda x: x[0])  # closest to rack first
            candidates.append(key_candidates[0][1])
            used_ids.add(key_candidates[0][1]['bid'])

    # Build observation (52 dims)
    feats = np.zeros(NUM_CANDIDATES * PER_CAND_FEATS, dtype=np.float32)
    detailed = []
    for slot in range(NUM_CANDIDATES):
        base = slot * PER_CAND_FEATS
        if slot >= len(candidates):
            detailed.append(None)
            continue
        c = candidates[slot]
        # Estimate cue-after for STOP spin as a baseline reference position
        cue_after_stop = estimate_cue_position(c['ghost'], c['aim_dir'], c['pdir'], 0)
        cue_after_stop = (max(R, min(TL-R, cue_after_stop[0])),
                          max(R, min(TW-R, cue_after_stop[1])))
        rack_dist = math.sqrt((cue_after_stop[0] - RACK_CENTER[0]) ** 2 +
                              (cue_after_stop[1] - RACK_CENTER[1]) ** 2)
        feats[base + 0] = c['bpos'][0] / TL
        feats[base + 1] = c['bpos'][1] / TW
        feats[base + 2] = cue_after_stop[0] / TL
        feats[base + 3] = cue_after_stop[1] / TW
        feats[base + 4] = min(c['difficulty'], 2.0) / 2.0
        feats[base + 5] = c['cluster'] / 15.0
        feats[base + 6] = min(c['best_next_diff'], 2.0) / 2.0
        feats[base + 7] = c['avg_reach'] / 15.0
        feats[base + 8] = min(rack_dist, TL) / TL
        feats[base + 9] = c['cue_to_ball'] / TL
        feats[base + 10] = c['ball_to_pocket'] / TL
        feats[base + 11] = 1.0
        detailed.append(c)

    on_table = sum(1 for b in table.balls if b not in table.pocketed)
    global_feats = np.array([
        table.cue[0] / TL,
        table.cue[1] / TW,
        min(table.run_length, 15) / 15.0,
        on_table / 15.0,
    ], dtype=np.float32)
    obs = np.concatenate([global_feats, feats])
    return detailed, obs


# ─── PPO trajectory and GAE ───────────────────────────────────────────────

class Trajectory:
    __slots__ = ['obs', 'shot_actions', 'spin_actions', 'speed_actions',
                 'shot_log_probs', 'spin_log_probs', 'speed_log_probs',
                 'values', 'rewards', 'shot_masks']

    def __init__(self):
        self.obs = []
        self.shot_actions = []
        self.spin_actions = []
        self.speed_actions = []
        self.shot_log_probs = []
        self.spin_log_probs = []
        self.speed_log_probs = []
        self.values = []
        self.rewards = []
        self.shot_masks = []

    def add(self, obs, shot_a, spin_a, speed_a,
            shot_lp, spin_lp, speed_lp, value, reward, shot_mask):
        self.obs.append(obs)
        self.shot_actions.append(shot_a)
        self.spin_actions.append(spin_a)
        self.speed_actions.append(speed_a)
        self.shot_log_probs.append(shot_lp)
        self.spin_log_probs.append(spin_lp)
        self.speed_log_probs.append(speed_lp)
        self.values.append(value)
        self.rewards.append(reward)
        self.shot_masks.append(shot_mask)

    def __len__(self):
        return len(self.obs)


def masked_probs(logits, mask):
    stable = logits - logits.max()
    exp = torch.exp(stable) * mask
    p = exp / exp.sum().clamp(min=1e-8)
    if torch.isnan(p).any():
        p = mask / mask.sum()
    return p


def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.array(values, dtype=np.float32)


# ─── Run episode ───────────────────────────────────────────────────────────

def play_run(net, device):
    """Single-player run: pocket balls until miss/foul/stuck. Returns trajectory."""
    table = RunTable()
    traj = Trajectory()

    with torch.no_grad():
        for shot_num in range(MAX_SHOTS):
            detailed, obs = build_candidates(table)
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            shot_logits, spin_logits, speed_logits, value = net(obs_t)
            shot_logits = shot_logits.squeeze(0)
            spin_logits = spin_logits.squeeze(0)
            speed_logits = speed_logits.squeeze(0)
            value = value.item()

            # Shot (masked)
            shot_mask = torch.ones(N_SHOT_ACTIONS, device=device)
            for i in range(NUM_CANDIDATES):
                if detailed[i] is None:
                    shot_mask[i] = 0.0
            if shot_mask[:NUM_CANDIDATES].sum().item() < 1:
                break  # no valid shots, run over

            shot_probs = masked_probs(shot_logits, shot_mask)
            shot_dist = torch.distributions.Categorical(shot_probs)
            shot_a = shot_dist.sample()
            shot_lp = shot_dist.log_prob(shot_a).item()
            si = shot_a.item()

            # Spin (no mask)
            spin_probs = torch.softmax(spin_logits, dim=0)
            spin_dist = torch.distributions.Categorical(spin_probs)
            spin_a = spin_dist.sample()
            spin_lp = spin_dist.log_prob(spin_a).item()
            spi = spin_a.item()

            # Speed (no mask)
            speed_probs = torch.softmax(speed_logits, dim=0)
            speed_dist = torch.distributions.Categorical(speed_probs)
            speed_a = speed_dist.sample()
            speed_lp = speed_dist.log_prob(speed_a).item()
            spd = speed_a.item()

            if si == NUM_CANDIDATES:
                # Pass — voluntarily end the run (e.g., no good shot left)
                traj.add(obs, si, spi, spd, shot_lp, spin_lp, speed_lp,
                         value, -1.0, shot_mask.numpy().copy())
                break

            c = detailed[si]
            # Execute with network's spin and speed choices
            (target_pocketed, all_pocketed, scratched,
             hit_ball, hit_rail) = table.execute_shot(
                c['bid'], spi, spd, c['ghost'], c['aim_dir']
            )

            foul = scratched or not hit_ball
            if foul:
                traj.add(obs, si, spi, spd, shot_lp, spin_lp, speed_lp,
                         value, -2.0, shot_mask.numpy().copy())
                break

            if target_pocketed or len(all_pocketed) > 0:
                n_pocketed = len(all_pocketed)
                table.run_length += n_pocketed
                # Escalating run reward
                reward = n_pocketed * (1.0 + (table.run_length - 1) * 0.5)
                # Shape: how easy is the next shot?
                active_now = table.get_active_balls()
                if len(active_now) == 0:
                    reward += 3.0  # cleared the table!
                else:
                    nd = best_next_difficulty(table.cue, active_now)
                    if nd < 0.8:
                        reward += 1.0   # great shape
                    elif nd < 1.2:
                        reward += 0.3   # decent shape
                    elif nd > 1.5:
                        reward -= 0.5   # bad shape
                traj.add(obs, si, spi, spd, shot_lp, spin_lp, speed_lp,
                         value, reward, shot_mask.numpy().copy())
            else:
                # Missed — run ends
                traj.add(obs, si, spi, spd, shot_lp, spin_lp, speed_lp,
                         value, -2.0, shot_mask.numpy().copy())
                break

    return traj, table.run_length


# ─── PPO training ──────────────────────────────────────────────────────────

def train():
    device = torch.device('cpu')
    net = RunNet().to(device)
    opt = optim.Adam(net.parameters(), lr=LR)
    # Fresh start — mixed environment (scattered + partial rack)
    print(f'{sum(p.numel() for p in net.parameters()):,} params, {device}', flush=True)

    best_avg_run = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    runs_per_batch = 64

    for iteration in range(50000):
        all_trajs = []
        run_lengths = []

        for _ in range(runs_per_batch):
            traj, run_len = play_run(net, device)
            run_lengths.append(run_len)
            if len(traj) > 0:
                all_trajs.append(traj)

        # GAE + flatten
        b_obs, b_shot_a, b_spin_a, b_speed_a = [], [], [], []
        b_shot_lp, b_spin_lp, b_speed_lp = [], [], []
        b_adv, b_ret, b_masks = [], [], []

        for traj in all_trajs:
            adv, ret = compute_gae(traj.rewards, traj.values)
            b_obs.extend(traj.obs)
            b_shot_a.extend(traj.shot_actions)
            b_spin_a.extend(traj.spin_actions)
            b_speed_a.extend(traj.speed_actions)
            b_shot_lp.extend(traj.shot_log_probs)
            b_spin_lp.extend(traj.spin_log_probs)
            b_speed_lp.extend(traj.speed_log_probs)
            b_adv.extend(adv.tolist())
            b_ret.extend(ret.tolist())
            b_masks.extend(traj.shot_masks)

        if len(b_obs) == 0:
            continue

        obs_t = torch.tensor(np.array(b_obs), device=device)
        shot_a_t = torch.tensor(b_shot_a, device=device, dtype=torch.long)
        spin_a_t = torch.tensor(b_spin_a, device=device, dtype=torch.long)
        speed_a_t = torch.tensor(b_speed_a, device=device, dtype=torch.long)
        old_shot_lp = torch.tensor(b_shot_lp, device=device)
        old_spin_lp = torch.tensor(b_spin_lp, device=device)
        old_speed_lp = torch.tensor(b_speed_lp, device=device)
        adv_t = torch.tensor(b_adv, device=device)
        ret_t = torch.tensor(b_ret, device=device)
        masks_t = torch.tensor(np.array(b_masks), device=device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO epochs
        t_pg, t_vl, t_ent, t_clip = 0.0, 0.0, 0.0, 0.0

        for epoch in range(K_EPOCHS):
            shot_lg, spin_lg, speed_lg, vals = net(obs_t)

            # Shot probs (masked)
            sp = torch.zeros_like(shot_lg)
            for i in range(shot_lg.size(0)):
                sp[i] = masked_probs(shot_lg[i], masks_t[i])
            sd = torch.distributions.Categorical(sp)
            new_shot_lp = sd.log_prob(shot_a_t)
            shot_ent = sd.entropy()

            # Spin probs
            spin_p = torch.softmax(spin_lg, dim=-1)
            spin_d = torch.distributions.Categorical(spin_p)
            new_spin_lp = spin_d.log_prob(spin_a_t)
            spin_ent = spin_d.entropy()

            # Speed probs
            speed_p = torch.softmax(speed_lg, dim=-1)
            speed_d = torch.distributions.Categorical(speed_p)
            new_speed_lp = speed_d.log_prob(speed_a_t)
            speed_ent = speed_d.entropy()

            # Joint ratio
            new_joint = new_shot_lp + new_spin_lp + new_speed_lp
            old_joint = old_shot_lp + old_spin_lp + old_speed_lp
            ratio = torch.exp(new_joint - old_joint)

            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            pg_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(vals, ret_t)
            ent_loss = -(ENTROPY_COEF * (shot_ent.mean() + speed_ent.mean())
                        + SPIN_ENTROPY_COEF * spin_ent.mean())
            loss = pg_loss + VALUE_COEF * v_loss + ent_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()

            with torch.no_grad():
                cf = ((ratio - 1).abs() > CLIP_EPS).float().mean().item()
            t_pg += pg_loss.item()
            t_vl += v_loss.item()
            t_ent += (shot_ent.mean().item() + spin_ent.mean().item() + speed_ent.mean().item())
            t_clip += cf

        avg_run = np.mean(run_lengths)
        max_run = np.max(run_lengths)

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'Iter {iteration+1:6d} | '
                  f'AvgRun={avg_run:.1f} MaxRun={max_run} | '
                  f'PG={t_pg/K_EPOCHS:.3f} VL={t_vl/K_EPOCHS:.2f} '
                  f'Ent={t_ent/K_EPOCHS:.3f} Clip={t_clip/K_EPOCHS:.2f} | '
                  f'{elapsed:.0f}s', flush=True)

            if avg_run > best_avg_run:
                best_avg_run = avg_run
                torch.save(net.state_dict(), 'checkpoints/best_run_net.pt')
                print(f'  -> Best avg run: {avg_run:.1f}', flush=True)

        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), 'checkpoints/latest_run_net.pt')

    print(f'Done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    train()
