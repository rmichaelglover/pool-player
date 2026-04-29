"""
14.1 Continuous hybrid self-play training with PPO + physics.

Physics-based shot execution replaces geometric estimates. The C physics
engine (pool_sim.c) simulates ball collisions, friction, cushion bounces,
and spin effects. The network now also picks shot speed (soft/medium/hard).

Candidate building still uses geometric lookahead (for speed), but the
actual shot outcome is physics-determined. This trains the network to
learn real position play, not a fake geometric model.
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

TARGET_SCORE = 10
NUM_BALLS = 15
NUM_CANDIDATES = 3
N_SHOT_ACTIONS = NUM_CANDIDATES + 1   # 3 candidates + safety
N_SPEED_ACTIONS = 3                    # soft / medium / hard
SPEED_LEVELS = [30, 60, 90]
PER_CAND_FEATS = 12
GLOBAL_FEATS = 6
INPUT_DIM = GLOBAL_FEATS + NUM_CANDIDATES * PER_CAND_FEATS   # 42
MAX_TURNS = 100
RACK_CENTER = (80.0, 25.0)
RACK_ZONE_RADIUS = 20.0

# PPO hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
K_EPOCHS = 4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.05
LR = 3e-4
MAX_GRAD_NORM = 0.5


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.shot_head = nn.Linear(64, N_SHOT_ACTIONS)    # which candidate or safety
        self.speed_head = nn.Linear(64, N_SPEED_ACTIONS)  # soft / medium / hard
        self.value_head = nn.Linear(64, 1)
        with torch.no_grad():
            self.shot_head.weight.zero_()
            self.shot_head.bias.zero_()
            self.speed_head.weight.zero_()
            self.speed_head.bias.zero_()

    def forward(self, x):
        h = self.trunk(x)
        return self.shot_head(h), self.speed_head(h), self.value_head(h).squeeze(-1)


# Alias for probe_strategy.py compatibility
StrategyNet = ActorCritic


# ─── Environment ───────────────────────────────────────────────────────────

class Table141:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cue = [R * 3 + random.random() * (TL - 6 * R),
                    R * 3 + random.random() * (TW - 6 * R)]
        self.balls = {}
        self.pocketed = set()
        self.scores = [0, 0]
        self.current_player = 0
        self.consec_fouls = [0, 0]
        self.run_length = 0
        self._scatter_balls()

    def _scatter_balls(self, exclude_id=None):
        placed = [self.cue]
        ids = list(range(1, 16))
        if exclude_id is not None and exclude_id in ids:
            ids.remove(exclude_id)
            placed.append(self.balls[exclude_id])
        for bid in ids:
            for _ in range(400):
                x = R * 2 + random.random() * (TL - 4 * R)
                y = R * 2 + random.random() * (TW - 4 * R)
                if all(math.sqrt((x - px) ** 2 + (y - py) ** 2) > 2.2 * R for px, py in placed):
                    self.balls[bid] = [x, y]
                    placed.append((x, y))
                    if bid in self.pocketed:
                        self.pocketed.remove(bid)
                    break

    def _setup_rack(self, exclude_id=None):
        """Place balls in a tight 14.1 triangle at (75, 25)."""
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

    def get_active_balls(self):
        return [(bid, pos) for bid, pos in self.balls.items() if bid not in self.pocketed]

    def execute_shot_physics(self, ball_id, spin, speed_level, ghost, aim_dir):
        """Execute a shot using the C physics engine.

        Returns: (target_pocketed, all_pocketed_ids, scratched,
                  hit_ball, hit_rail, rerack_bonus)
        """
        active = {}
        for bid, pos in self.balls.items():
            if bid not in self.pocketed:
                active[bid] = (pos[0], pos[1])

        force = SPEED_LEVELS[speed_level]
        cue_vx = aim_dir[0] * force
        cue_vy = aim_dir[1] * force

        result = simulate_shot(
            tuple(self.cue), active, cue_vx, cue_vy,
            spin, aim_dir[0], aim_dir[1]
        )

        # Update positions from physics
        for bid, (fx, fy) in result.final_positions.items():
            if bid == 0:
                self.cue = [fx, fy]
            elif bid in self.balls:
                self.balls[bid] = [fx, fy]

        # Handle scratch: place cue in kitchen
        if result.cue_scratched:
            self.cue = [R * 3 + random.random() * (TL / 4),
                        R * 3 + random.random() * (TW - 6 * R)]

        # Process pocketed object balls
        obj_pocketed = []
        for bid in result.pocketed_ids:
            if bid == 0:
                continue
            if bid not in self.pocketed:
                self.pocketed.add(bid)
                obj_pocketed.append(bid)

        target_pocketed = ball_id in result.pocketed_ids

        # Re-rack check
        rerack_bonus = 0.0
        on_table = sum(1 for b in self.balls if b not in self.pocketed)
        if on_table <= 1:
            remain_id = None
            for b in self.balls:
                if b not in self.pocketed:
                    remain_id = b
                    break
            if remain_id is not None:
                bx, by = self.balls[remain_id]
                rack_dist = math.sqrt((bx - RACK_CENTER[0]) ** 2 + (by - RACK_CENTER[1]) ** 2)
                rerack_bonus = 2.0 if rack_dist < RACK_ZONE_RADIUS else 0.5
            # Cue position bonus: reward being near the rack to break it open
            cue_rack_dist = math.sqrt((self.cue[0] - RACK_CENTER[0]) ** 2 +
                                      (self.cue[1] - RACK_CENTER[1]) ** 2)
            if cue_rack_dist < 15:
                rerack_bonus += 1.5  # great break position
            elif cue_rack_dist < 30:
                rerack_bonus += 0.5  # decent
            # Tight triangle re-rack (real 14.1 rack, not scattered)
            self._setup_rack(exclude_id=remain_id)

        return (target_pocketed, obj_pocketed, result.cue_scratched,
                result.hit_ball, result.hit_rail, rerack_bonus)

    def execute_safety_physics(self):
        """Play a safety: softly hit the nearest ball. Physics determines
        final positions, contact, and rail. Returns (hit_ball, hit_rail, scratched)."""
        active = self.get_active_balls()
        if not active:
            return False, False, False

        # Pick the nearest ball and tap it softly
        best_bid, best_dist = None, 999.0
        for bid, (bx, by) in active:
            d = math.sqrt((bx - self.cue[0]) ** 2 + (by - self.cue[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_bid = bid

        bx, by = self.balls[best_bid]
        dx, dy = bx - self.cue[0], by - self.cue[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 0.1:
            return False, False, False
        aim_dir = (dx / d, dy / d)

        active_dict = {bid: (pos[0], pos[1]) for bid, pos in self.balls.items()
                       if bid not in self.pocketed}
        result = simulate_shot(
            tuple(self.cue), active_dict,
            aim_dir[0] * 20, aim_dir[1] * 20,  # very soft
            0, aim_dir[0], aim_dir[1]           # stop spin
        )

        # Update positions
        for bid, (fx, fy) in result.final_positions.items():
            if bid == 0:
                self.cue = [fx, fy]
            elif bid in self.balls:
                self.balls[bid] = [fx, fy]

        if result.cue_scratched:
            self.cue = [R * 3 + random.random() * (TL / 4),
                        R * 3 + random.random() * (TW - 6 * R)]

        # Process any accidentally pocketed balls
        for bid in result.pocketed_ids:
            if bid != 0 and bid not in self.pocketed:
                self.pocketed.add(bid)

        return result.hit_ball, result.hit_rail, result.cue_scratched


# ─── Helpers ───────────────────────────────────────────────────────────────

def cluster_count(ball_pos, active, radius=5 * R):
    cnt = 0
    for bid, (bx, by) in active:
        d = math.sqrt((bx - ball_pos[0]) ** 2 + (by - ball_pos[1]) ** 2)
        if 0.1 < d < radius:
            cnt += 1
    return cnt


def lookahead_for_cue_after(cue_after, active_minus_target):
    reach = 0
    min_next_diff = 2.0
    for bid2, bpos2 in active_minus_target:
        res = best_pocket_for_ball(cue_after, bpos2, bid2, active_minus_target)
        if res is None:
            continue
        reach += 1
        pidx2, ca2, ghost2, _ = res
        dx2, dy2 = ghost2[0] - cue_after[0], ghost2[1] - cue_after[1]
        d2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        px2, py2 = POCKETS[pidx2][0].item(), POCKETS[pidx2][1].item()
        pd2 = math.sqrt((px2 - bpos2[0]) ** 2 + (py2 - bpos2[1]) ** 2)
        diff2 = ca2 / 75.0 + (d2 + pd2) / (TL * 1.5)
        if diff2 < min_next_diff:
            min_next_diff = diff2
    return reach, min_next_diff


def pick_best_spin(ghost, aim_dir, pocket_dir, active_minus_target):
    best = None
    for spin in range(3):
        cue_after = estimate_cue_position(ghost, aim_dir, pocket_dir, spin)
        cue_after = (max(R, min(TL - R, cue_after[0])),
                     max(R, min(TW - R, cue_after[1])))
        reach, next_diff = lookahead_for_cue_after(cue_after, active_minus_target)
        key = (-reach, next_diff)
        if best is None or key < best[0]:
            best = (key, spin, cue_after, reach, next_diff)
    _, spin, cue_after, reach, next_diff = best
    return spin, cue_after, reach, next_diff


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
    for c in makeable:
        active_minus = [(bid, bp) for bid, bp in active if bid != c['bid']]
        spin, cue_after, reach, next_diff = pick_best_spin(
            c['ghost'], c['aim_dir'], c['pdir'], active_minus
        )
        c['best_spin'] = spin
        c['cue_after'] = cue_after
        c['reach'] = reach
        c['next_diff'] = next_diff
        c['cluster'] = cluster_count(c['bpos'], active)

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
        # Slot 2: ball whose best spin leaves the easiest follow-up
        position_sorted = sorted(makeable, key=lambda c: (c['next_diff'], c['difficulty']))
        for c in position_sorted:
            if c['bid'] not in used_ids:
                candidates.append(c); used_ids.add(c['bid']); break

    feats = np.zeros(NUM_CANDIDATES * PER_CAND_FEATS, dtype=np.float32)
    detailed = []
    for slot in range(NUM_CANDIDATES):
        base = slot * PER_CAND_FEATS
        if slot >= len(candidates):
            detailed.append(None)
            continue
        c = candidates[slot]
        cue_after = c['cue_after']
        reach = c['reach']
        next_diff = c['next_diff']
        rack_dist = math.sqrt((cue_after[0] - RACK_CENTER[0]) ** 2 +
                              (cue_after[1] - RACK_CENTER[1]) ** 2)
        feats[base + 0] = c['bpos'][0] / TL
        feats[base + 1] = c['bpos'][1] / TW
        feats[base + 2] = cue_after[0] / TL
        feats[base + 3] = cue_after[1] / TW
        feats[base + 4] = min(c['difficulty'], 2.0) / 2.0
        feats[base + 5] = c['cluster'] / 15.0
        feats[base + 6] = min(next_diff, 2.0) / 2.0
        feats[base + 7] = reach / 15.0
        feats[base + 8] = min(rack_dist, TL) / TL
        feats[base + 9] = c['cue_to_ball'] / TL
        feats[base + 10] = c['ball_to_pocket'] / TL
        feats[base + 11] = 1.0
        detailed.append(c)

    p = table.current_player
    on_table = sum(1 for b in table.balls if b not in table.pocketed)
    global_feats = np.array([
        table.cue[0] / TL,
        table.cue[1] / TW,
        table.scores[p] / TARGET_SCORE,
        table.scores[1 - p] / TARGET_SCORE,
        table.consec_fouls[p] / 3.0,
        on_table / 15.0,
    ], dtype=np.float32)
    obs = np.concatenate([global_feats, feats])
    return detailed, obs


# ─── PPO trajectory and GAE ───────────────────────────────────────────────

class Trajectory:
    __slots__ = ['obs', 'shot_actions', 'speed_actions',
                 'shot_log_probs', 'speed_log_probs',
                 'values', 'rewards', 'shot_masks']

    def __init__(self):
        self.obs = []
        self.shot_actions = []
        self.speed_actions = []
        self.shot_log_probs = []
        self.speed_log_probs = []
        self.values = []
        self.rewards = []
        self.shot_masks = []

    def add(self, obs, shot_a, speed_a, shot_lp, speed_lp, value, reward, shot_mask):
        self.obs.append(obs)
        self.shot_actions.append(shot_a)
        self.speed_actions.append(speed_a)
        self.shot_log_probs.append(shot_lp)
        self.speed_log_probs.append(speed_lp)
        self.values.append(value)
        self.rewards.append(reward)
        self.shot_masks.append(shot_mask)

    def __len__(self):
        return len(self.obs)


def masked_probs_from_logits(logits, mask):
    stable = logits - logits.max()
    exp = torch.exp(stable) * mask
    return exp / exp.sum().clamp(min=1e-8)


def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def play_game(net, device):
    """Play one game with physics-based shot execution."""
    table = Table141()
    trajs = [Trajectory(), Trajectory()]
    pending_safety = [None, None]

    def resolve_pending(opp, outcome_reward):
        if pending_safety[opp] is None:
            return
        idx = pending_safety[opp]
        trajs[opp].rewards[idx] = outcome_reward
        pending_safety[opp] = None

    with torch.no_grad():
        for turn in range(MAX_TURNS):
            p = table.current_player
            detailed, obs = build_candidates(table)
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            shot_logits, speed_logits, value = net(obs_t)
            shot_logits = shot_logits.squeeze(0)
            speed_logits = speed_logits.squeeze(0)
            value = value.item()

            # Shot action (masked)
            shot_mask = torch.ones(N_SHOT_ACTIONS, device=device)
            for i in range(NUM_CANDIDATES):
                if detailed[i] is None:
                    shot_mask[i] = 0.0
            if shot_mask.sum().item() < 1:
                break

            shot_probs = masked_probs_from_logits(shot_logits, shot_mask)
            if torch.isnan(shot_probs).any():
                shot_probs = shot_mask / shot_mask.sum()  # NaN guard: fall back to uniform
            shot_dist = torch.distributions.Categorical(shot_probs)
            shot_a = shot_dist.sample()
            shot_lp = shot_dist.log_prob(shot_a).item()
            si = shot_a.item()

            # Speed action (uniform softmax, no masking)
            speed_probs = torch.softmax(speed_logits, dim=0)
            speed_dist = torch.distributions.Categorical(speed_probs)
            speed_a = speed_dist.sample()
            speed_lp = speed_dist.log_prob(speed_a).item()
            spi = speed_a.item()

            if si == NUM_CANDIDATES:
                # Safety — physics-based: softly hit nearest ball
                hit_ball, hit_rail, scratched = table.execute_safety_physics()
                safety_foul = scratched or not hit_ball or (not hit_rail and hit_ball)
                if safety_foul:
                    # Safety caused a foul
                    resolve_pending(1 - p, 0.5)
                    table.scores[p] -= 1
                    table.consec_fouls[p] += 1
                    if table.consec_fouls[p] >= 3:
                        table.scores[p] -= 15
                        table.consec_fouls[p] = 0
                    table.run_length = 0
                    table.current_player = 1 - p
                    trajs[p].add(obs, si, spi, shot_lp, speed_lp, value,
                                 -1.5, shot_mask.numpy().copy())
                else:
                    # Legal safety — reward pending until opponent acts
                    resolve_pending(1 - p, 0.5)
                    table.consec_fouls[p] = 0
                    table.run_length = 0
                    table.current_player = 1 - p
                    pending_safety[p] = len(trajs[p])
                    trajs[p].add(obs, si, spi, shot_lp, speed_lp, value,
                                 0.0, shot_mask.numpy().copy())
            else:
                c = detailed[si]
                # Physics-based shot execution
                (target_pocketed, all_pocketed, scratched,
                 hit_ball, hit_rail, rerack_bonus) = table.execute_shot_physics(
                    c['bid'], c['best_spin'], spi, c['ghost'], c['aim_dir']
                )

                # Foul detection
                foul = False
                if scratched:
                    foul = True
                elif not hit_ball:
                    foul = True
                elif not hit_rail and len(all_pocketed) == 0:
                    foul = True

                if foul:
                    # Resolve opponent's pending safety
                    resolve_pending(1 - p, 0.5)
                    table.scores[p] -= 1
                    table.consec_fouls[p] += 1
                    if table.consec_fouls[p] >= 3:
                        table.scores[p] -= 15
                        table.consec_fouls[p] = 0
                    table.run_length = 0
                    table.current_player = 1 - p
                    trajs[p].add(obs, si, spi, shot_lp, speed_lp, value,
                                 -1.5, shot_mask.numpy().copy())
                elif target_pocketed or len(all_pocketed) > 0:
                    # Pocketed at least one ball
                    if target_pocketed:
                        resolve_pending(1 - p, -1.0)
                    else:
                        resolve_pending(1 - p, 0.5)

                    n_pocketed = len(all_pocketed)
                    table.scores[p] += n_pocketed
                    table.consec_fouls[p] = 0
                    table.run_length += n_pocketed
                    reward = n_pocketed * (1.0 + (table.run_length - 1) * 0.5) + rerack_bonus
                    reward += 0.15 * c['cluster']
                    # Shape bonus: how good is our position after the shot?
                    active_now = table.get_active_balls()
                    nd = best_next_difficulty(table.cue, active_now)
                    if nd < 1.0:
                        reward += 0.5  # good shape bonus
                    elif nd > 1.5:
                        reward -= 0.5  # bad shape penalty
                    # Break ball positioning: when few balls left, reward cue
                    # being near the rack area (needed to break the re-rack)
                    on_now = len(active_now)
                    if on_now <= 3:
                        cue_rack = math.sqrt((table.cue[0] - RACK_CENTER[0]) ** 2 +
                                             (table.cue[1] - RACK_CENTER[1]) ** 2)
                        if cue_rack < 15:
                            reward += 1.0
                        elif cue_rack < 30:
                            reward += 0.3
                    trajs[p].add(obs, si, spi, shot_lp, speed_lp, value,
                                 reward, shot_mask.numpy().copy())
                    if table.scores[p] >= TARGET_SCORE:
                        trajs[p].rewards[-1] += 5.0
                        if len(trajs[1 - p]) > 0:
                            trajs[1 - p].rewards[-1] -= 3.0
                        break
                else:
                    # Missed (no foul, no pocket)
                    resolve_pending(1 - p, 0.5)
                    table.run_length = 0
                    table.current_player = 1 - p
                    # Opponent-shape penalty
                    miss_penalty = -0.5
                    opp_nd = best_next_difficulty(table.cue, table.get_active_balls())
                    if opp_nd < 0.5:
                        miss_penalty -= 0.3
                    trajs[p].add(obs, si, spi, shot_lp, speed_lp, value,
                                 miss_penalty, shot_mask.numpy().copy())

    for i in range(2):
        resolve_pending(i, 0.5)

    return trajs, table.scores


# ─── PPO training ──────────────────────────────────────────────────────────

def train():
    device = torch.device('cpu')
    net = ActorCritic().to(device)
    opt = optim.Adam(net.parameters(), lr=LR)
    # Start fresh — physics world model is different from geometric
    print(f'{sum(p.numel() for p in net.parameters()):,} params, {device}', flush=True)

    best_loser = 999.0
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()
    games_per_batch = 32

    for iteration in range(50000):
        all_trajs = []
        game_scores = []
        for _ in range(games_per_batch):
            trajs, scores = play_game(net, device)
            game_scores.append(scores)
            for t in trajs:
                if len(t) > 0:
                    all_trajs.append(t)

        # Compute GAE per trajectory, flatten into batch
        batch_obs = []
        batch_shot_a = []
        batch_speed_a = []
        batch_old_shot_lp = []
        batch_old_speed_lp = []
        batch_advantages = []
        batch_returns = []
        batch_shot_masks = []

        for traj in all_trajs:
            advantages, returns = compute_gae(traj.rewards, traj.values)
            batch_obs.extend(traj.obs)
            batch_shot_a.extend(traj.shot_actions)
            batch_speed_a.extend(traj.speed_actions)
            batch_old_shot_lp.extend(traj.shot_log_probs)
            batch_old_speed_lp.extend(traj.speed_log_probs)
            batch_advantages.extend(advantages.tolist())
            batch_returns.extend(returns.tolist())
            batch_shot_masks.extend(traj.shot_masks)

        if len(batch_obs) == 0:
            continue

        obs_t = torch.tensor(np.array(batch_obs), device=device)
        shot_a_t = torch.tensor(batch_shot_a, device=device, dtype=torch.long)
        speed_a_t = torch.tensor(batch_speed_a, device=device, dtype=torch.long)
        old_shot_lp_t = torch.tensor(batch_old_shot_lp, device=device)
        old_speed_lp_t = torch.tensor(batch_old_speed_lp, device=device)
        adv_t = torch.tensor(batch_advantages, device=device)
        ret_t = torch.tensor(batch_returns, device=device)
        masks_t = torch.tensor(np.array(batch_shot_masks), device=device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO epochs
        total_pg = 0.0
        total_vl = 0.0
        total_ent = 0.0
        total_clip = 0.0

        for epoch in range(K_EPOCHS):
            shot_logits, speed_logits, values = net(obs_t)

            # Shot probabilities with masking
            shot_probs = torch.zeros_like(shot_logits)
            for i in range(shot_logits.size(0)):
                shot_probs[i] = masked_probs_from_logits(shot_logits[i], masks_t[i])
            shot_dist = torch.distributions.Categorical(shot_probs)
            new_shot_lp = shot_dist.log_prob(shot_a_t)
            shot_entropy = shot_dist.entropy()

            # Speed probabilities (no mask)
            speed_probs = torch.softmax(speed_logits, dim=-1)
            speed_dist = torch.distributions.Categorical(speed_probs)
            new_speed_lp = speed_dist.log_prob(speed_a_t)
            speed_entropy = speed_dist.entropy()

            # Joint log prob and ratio
            new_joint_lp = new_shot_lp + new_speed_lp
            old_joint_lp = old_shot_lp_t + old_speed_lp_t
            ratio = torch.exp(new_joint_lp - old_joint_lp)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            pg_loss = -torch.min(surr1, surr2).mean()

            v_loss = F.mse_loss(values, ret_t)
            ent_loss = -(shot_entropy.mean() + speed_entropy.mean())

            loss = pg_loss + VALUE_COEF * v_loss + ENTROPY_COEF * ent_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1).abs() > CLIP_EPS).float().mean().item()
            total_pg += pg_loss.item()
            total_vl += v_loss.item()
            total_ent += (shot_entropy.mean().item() + speed_entropy.mean().item())
            total_clip += clip_frac

        avg_pg = total_pg / K_EPOCHS
        avg_vl = total_vl / K_EPOCHS
        avg_ent = total_ent / K_EPOCHS
        avg_clip = total_clip / K_EPOCHS

        if (iteration + 1) % 10 == 0:
            sa = np.array(game_scores)
            avg_p1 = sa[:, 0].mean()
            avg_p2 = sa[:, 1].mean()
            avg_loser = sa.min(axis=1).mean()
            wins = sum(1 for s in game_scores if s[0] >= TARGET_SCORE or s[1] >= TARGET_SCORE)
            elapsed = time.time() - t0
            print(f'Iter {iteration+1:6d} | '
                  f'P1={avg_p1:.1f} P2={avg_p2:.1f} | '
                  f'AvgLoser={avg_loser:.2f} | '
                  f'Decided={wins}/{games_per_batch} | '
                  f'PG={avg_pg:.3f} VL={avg_vl:.2f} '
                  f'Ent={avg_ent:.3f} Clip={avg_clip:.2f} | '
                  f'{elapsed:.0f}s', flush=True)

            if avg_loser < best_loser:
                best_loser = avg_loser
                torch.save(net.state_dict(), 'checkpoints/best_strategy_ppo.pt')
                print(f'  -> Best avg loser: {avg_loser:.2f}', flush=True)

        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), 'checkpoints/latest_strategy_ppo.pt')

    print(f'Done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    train()
