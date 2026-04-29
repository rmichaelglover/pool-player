"""
Phase 3: Shot selection and sequencing with multiple balls.

The network sees all ball positions and outputs the full decision:
  - Which ball to shoot (1 of N)
  - Which pocket to target (1 of 6)
  - What spin type (draw / stop / follow)

Everything is geometric (no physics). The aim angle comes from the
already-trained aiming network. This system learns STRATEGY:
which sequence of balls + spin types runs the most balls.

Metric: balls pocketed in a row before getting stuck.
Reward: escalating (1x, 1.5x, 2x...) for consecutive pockets.
"""
import torch, torch.nn as nn, torch.optim as optim, math, time, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import BALL_RADIUS as R, TABLE_LENGTH as TL, TABLE_WIDTH as TW

# Pocket aim points (matching training/demo)
_mo = 2.5 * 0.45
_smo = 2.75 * 0.15
POCKETS = torch.tensor([
    [_mo, _mo], [TL/2, _smo], [TL-_mo, _mo],
    [_mo, TW-_mo], [TL/2, TW-_smo], [TL-_mo, TW-_mo],
], dtype=torch.float32)
POCKET_RADII = torch.tensor([2.5, 2.75, 2.5, 2.5, 2.75, 2.5])

IDEAL_DIR = torch.tensor([[-1,-1],[0,-1],[1,-1],[-1,1],[0,1],[1,1]], dtype=torch.float32)
IDEAL_DIR = IDEAL_DIR / IDEAL_DIR.norm(dim=1, keepdim=True)
MAX_ANGLE = torch.tensor([65,50,65,65,50,65], dtype=torch.float32) * math.pi / 180

NUM_BALLS = 15  # object balls (no cue ball in the count)
NUM_POCKETS = 6
NUM_SPIN = 3    # 0=stop, 1=follow, 2=draw

# Spin effect on cue ball position after collision (geometric estimate)
# Returns (direction_blend, distance_multiplier)
SPIN_PARAMS = {
    0: ('stop', 0.0, 3.0),     # stop: stay near contact, short drift
    1: ('follow', 1.0, 20.0),  # follow: continue along pocket direction
    2: ('draw', -1.0, 12.0),   # draw: reverse along aim direction
}


def best_pocket_for_ball(cue_pos, ball_pos, ball_id, active_balls):
    """Pick the pocket with the lowest cut angle (easiest shot) for this ball.
    Only considers pockets with clear paths (cue->ghost AND ball->pocket).
    Returns (pocket_idx, cut_angle_deg, ghost, pocket_dir) or None if no valid pocket."""
    best = None
    best_ca = 999
    for p in range(NUM_POCKETS):
        px, py = POCKETS[p][0].item(), POCKETS[p][1].item()
        if not pocket_approach_ok(ball_pos, p):
            continue
        ghost = compute_ghost(ball_pos, (px, py))
        if ghost is None:
            continue
        dx, dy = px - ball_pos[0], py - ball_pos[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d < 0.1:
            continue
        pocket_dir = (dx/d, dy/d)
        ca = cut_angle(cue_pos, ghost, pocket_dir)
        if ca > 75:
            continue
        # Both paths must be clear (exclude the target ball itself)
        if not is_path_clear(cue_pos, ghost, active_balls, {ball_id}):
            continue
        if not is_path_clear(ball_pos, (px, py), active_balls, {ball_id}):
            continue
        # Combined difficulty: cut angle + shot length
        # Normalize both to 0-1 range, weight equally
        cue_to_ghost = math.sqrt((ghost[0]-cue_pos[0])**2 + (ghost[1]-cue_pos[1])**2)
        ball_to_pocket = math.sqrt((px-ball_pos[0])**2 + (py-ball_pos[1])**2)
        total_dist = cue_to_ghost + ball_to_pocket

        difficulty = ca / 75.0 + total_dist / (TL * 1.5)

        if difficulty < best_ca:
            best_ca = difficulty
            best = (p, ca, ghost, pocket_dir)
    return best


class SequenceNet(nn.Module):
    """
    Input: cue ball pos (2) + up to 15 ball positions (30) = 32 dims
    Output: score for each (ball, spin) combination
            = 15 balls * 3 spins = 45 scores

    Pocket is selected geometrically (lowest cut angle), not by the network.
    """
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, NUM_BALLS * NUM_SPIN),  # 45 outputs
        )

    def forward(self, x):
        scores = self.net(x)
        return scores.view(-1, NUM_BALLS, NUM_SPIN)


def is_path_clear(start, end, balls, exclude_ids, clearance=2*R):
    """Check if a straight path between two points is clear of other balls.
    Checks perpendicular distance along the segment body AND distance to
    endpoints — a ball near either endpoint would block even if its
    projection falls outside [0, length]."""
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length < 0.1:
        return True
    nx, ny = dx/length, dy/length
    cl2 = clearance * clearance
    for bid, (bx, by) in balls:
        if bid in exclude_ids:
            continue
        bx2, by2 = bx - start[0], by - start[1]
        proj = bx2*nx + by2*ny
        if proj < 0:
            # Past start: check distance to start point
            if bx2*bx2 + by2*by2 < cl2:
                return False
        elif proj > length:
            # Past end: check distance to end point
            ex, ey = bx2 - dx, by2 - dy
            if ex*ex + ey*ey < cl2:
                return False
        else:
            perp = abs(-bx2*ny + by2*nx)
            if perp < clearance:
                return False
    return True


def compute_ghost(ball_pos, pocket_pos):
    """Ghost ball position for aiming."""
    dx, dy = pocket_pos[0]-ball_pos[0], pocket_pos[1]-ball_pos[1]
    d = math.sqrt(dx*dx + dy*dy)
    if d < 0.01:
        return None
    return (ball_pos[0] - dx/d * 2*R, ball_pos[1] - dy/d * 2*R)


def cut_angle(cue_pos, ghost_pos, pocket_dir):
    """Cut angle in degrees."""
    dx, dy = ghost_pos[0]-cue_pos[0], ghost_pos[1]-cue_pos[1]
    d = math.sqrt(dx*dx + dy*dy)
    if d < 0.01:
        return 90
    aim_nx, aim_ny = dx/d, dy/d
    dot = aim_nx*pocket_dir[0] + aim_ny*pocket_dir[1]
    return math.acos(max(-1, min(1, dot))) * 180/math.pi


def estimate_cue_position(ghost, aim_dir, pocket_dir, spin_type):
    """Estimate where the cue ball ends up after pocketing with given spin."""
    _, blend, dist = SPIN_PARAMS[spin_type]
    if spin_type == 0:  # stop
        # Cue stays near ghost with small perpendicular drift
        perp_x, perp_y = -pocket_dir[1], pocket_dir[0]
        return (ghost[0] + perp_x * dist, ghost[1] + perp_y * dist)
    elif spin_type == 1:  # follow
        # Cue continues along pocket direction
        return (ghost[0] + pocket_dir[0] * dist, ghost[1] + pocket_dir[1] * dist)
    else:  # draw
        # Cue reverses along aim direction
        return (ghost[0] - aim_dir[0] * dist, ghost[1] - aim_dir[1] * dist)


def pocket_approach_ok(ball_pos, pocket_idx):
    """Check if the pocket approach angle is valid.
    Closer balls get a more lenient angle limit — when the ball is
    near the pocket, it doesn't have to travel far between the cushion
    noses so steeper approaches work."""
    px, py = POCKETS[pocket_idx]
    dx, dy = px - ball_pos[0], py - ball_pos[1]
    d = math.sqrt(dx*dx + dy*dy)
    if d < 0.01:
        return True
    nx, ny = dx/d, dy/d
    dot = nx * IDEAL_DIR[pocket_idx][0].item() + ny * IDEAL_DIR[pocket_idx][1].item()
    angle = math.acos(max(-1, min(1, dot)))
    # Base limit from table geometry
    base_limit = MAX_ANGLE[pocket_idx].item()
    # Add up to 20 degrees for close balls (within 20 inches)
    distance_bonus = max(0, (20 - d) / 20) * (20 * math.pi / 180)
    return angle < (base_limit + distance_bonus)


class TableState:
    """Represents the current table for one episode."""
    def __init__(self, num_balls=3):
        self.num_balls = num_balls
        self.reset()

    def reset(self):
        # Place cue ball
        self.cue = [R*3 + np.random.random()*(TL-6*R), R*3 + np.random.random()*(TW-6*R)]
        # Place object balls (not overlapping)
        self.balls = {}  # {id: (x, y)}
        placed = [self.cue]
        for i in range(1, self.num_balls + 1):
            for _ in range(200):
                x = R*3 + np.random.random()*(TL-6*R)
                y = R*3 + np.random.random()*(TW-6*R)
                ok = all(math.sqrt((x-px)**2+(y-py)**2) > 3*R for px, py in placed)
                if ok:
                    self.balls[i] = (x, y)
                    placed.append((x, y))
                    break
        self.pocketed = set()

    def get_obs(self):
        """32-dim observation: cue(2) + 15 ball positions (30, -1 if pocketed/absent)."""
        obs = np.full(32, -1.0, dtype=np.float32)
        obs[0] = self.cue[0] / TL
        obs[1] = self.cue[1] / TW
        for bid, (bx, by) in self.balls.items():
            if bid not in self.pocketed and bid <= 15:
                idx = 2 + (bid-1)*2
                obs[idx] = bx / TL
                obs[idx+1] = by / TW
        return obs

    def get_active_balls(self):
        """List of (id, (x,y)) for non-pocketed balls."""
        return [(bid, pos) for bid, pos in self.balls.items() if bid not in self.pocketed]

    def best_shot_for_ball(self, ball_id, spin_type):
        """Find the best pocket for this ball (lowest cut angle, clear paths).
        Returns: (valid, pocket_idx, difficulty, cue_final) or (False, ...)"""
        if ball_id in self.pocketed or ball_id not in self.balls:
            return False, None, None, None

        ball_pos = self.balls[ball_id]
        active = self.get_active_balls()
        result = best_pocket_for_ball(self.cue, ball_pos, ball_id, active)
        if result is None:
            return False, None, None, None

        pocket_idx, ca, ghost, pocket_dir = result

        # Aim direction
        dx, dy = ghost[0]-self.cue[0], ghost[1]-self.cue[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d < 0.1:
            return False, None, None, None
        aim_dir = (dx/d, dy/d)

        # Estimate cue ball position
        cue_final = estimate_cue_position(ghost, aim_dir, pocket_dir, spin_type)
        cue_final = (max(R, min(TL-R, cue_final[0])), max(R, min(TW-R, cue_final[1])))

        difficulty = ca / 75.0 + d / TL
        return True, pocket_idx, difficulty, cue_final

    def execute_shot(self, ball_id, spin_type):
        """Execute a shot using the best pocket.
        Success probability decreases with difficulty (steeper cuts and
        longer shots miss more often, just like real pool).
        Returns (pocketed, pocket_idx)."""
        valid, pocket_idx, difficulty, cue_final = self.best_shot_for_ball(ball_id, spin_type)
        if not valid:
            return False, None

        # Miss probability based on difficulty
        # difficulty = cut_angle/75 + total_dist/(TL*1.5)
        # Easy shot (difficulty ~0.2): 98% success
        # Medium shot (difficulty ~0.6): 85% success
        # Hard shot (difficulty ~1.0): 60% success
        # Very hard (difficulty ~1.5): 30% success
        success_prob = max(0.1, 1.0 - difficulty * 0.5)
        if np.random.random() > success_prob:
            # Missed — ball stays, cue ball moves to approximate position
            self.cue = list(cue_final)
            return False, pocket_idx

        self.pocketed.add(ball_id)
        self.cue = list(cue_final)
        return True, pocket_idx


def play_episode(net, table, device='cpu'):
    """Play one episode using the network's decisions. Returns total reward."""
    table.reset()
    total_reward = 0
    run_length = 0
    max_shots = 30

    for shot_num in range(max_shots):
        if len(table.get_active_balls()) == 0:
            break

        obs = torch.tensor(table.get_obs(), device=device).unsqueeze(0)
        with torch.no_grad():
            scores = net(obs)  # (1, 15, 3)

        scores_np = scores.squeeze(0).cpu().numpy()
        best_score = -1e9
        best_choice = None

        for bid, _ in table.get_active_balls():
            if bid > 15: continue
            for s in range(3):
                valid, pidx, _, _ = table.best_shot_for_ball(bid, s)
                if valid:
                    sc = scores_np[bid-1, s]
                    if sc > best_score:
                        best_score = sc
                        best_choice = (bid, s)

        if best_choice is None:
            break

        bid, sid = best_choice
        pocketed, pocket_idx = table.execute_shot(bid, sid)

        if pocketed:
            run_length += 1
            reward = 2.0 * (1.0 + (run_length - 1) * 0.5)
            total_reward += reward
        else:
            run_length = 0
            break

    return total_reward, run_length


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SequenceNet(512).to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    print(f'{sum(p.numel() for p in net.parameters()):,} params, {device}')

    # Training: use REINFORCE (policy gradient) since episodes are sequential
    # Each episode: play a game, collect rewards, update policy
    batch_size = 64
    num_balls = 3
    best_avg = 0
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    for iteration in range(50000):
        # Curriculum
        if iteration > 5000: num_balls = 5
        if iteration > 15000: num_balls = 8
        if iteration > 30000: num_balls = 12

        batch_rewards = []
        batch_log_probs = []

        for _ in range(batch_size):
            table = TableState(num_balls)
            table.reset()
            episode_log_probs = []
            episode_rewards = []
            run_length = 0

            for shot_num in range(30):
                active = table.get_active_balls()
                if len(active) == 0:
                    break

                obs = torch.tensor(table.get_obs(), device=device).unsqueeze(0)
                scores = net(obs).squeeze(0)  # (15, 3)

                # Build valid actions: (ball, spin) pairs where ball has a viable pocket
                valid_actions = []
                valid_scores = []
                for bid, _ in active:
                    if bid > 15: continue
                    for s in range(3):
                        valid, pidx, _, _ = table.best_shot_for_ball(bid, s)
                        if valid:
                            valid_actions.append((bid, s, pidx))
                            valid_scores.append(scores[bid-1, s])

                if len(valid_actions) == 0:
                    break

                # Softmax over valid actions (with stability)
                valid_scores_t = torch.stack(valid_scores)
                valid_scores_t = valid_scores_t - valid_scores_t.max()
                probs = torch.softmax(valid_scores_t, dim=0)
                probs = probs.clamp(min=1e-8)
                probs = probs / probs.sum()
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

                bid, sid, pid = valid_actions[action_idx.item()]
                pocketed, pocket_idx = table.execute_shot(bid, sid)

                if pocketed:
                    run_length += 1
                    reward = 2.0 * (1.0 + (run_length - 1) * 0.5)
                else:
                    reward = -0.5
                    run_length = 0

                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)

                if not pocketed:
                    break

            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + 0.99 * G
                returns.insert(0, G)

            if len(returns) > 0:
                returns_t = torch.tensor(returns, device=device)
                if len(returns) > 1:
                    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
                else:
                    returns_t = returns_t - returns_t.mean()
                for lp, ret in zip(episode_log_probs, returns_t):
                    batch_log_probs.append(-lp * ret)
                batch_rewards.append(sum(episode_rewards))

        if len(batch_log_probs) > 0:
            loss = torch.stack(batch_log_probs).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        if (iteration + 1) % 100 == 0:
            avg_reward = np.mean(batch_rewards) if batch_rewards else 0
            avg_run = np.mean([r for r in batch_rewards if r > 0]) if any(r > 0 for r in batch_rewards) else 0
            max_run = max(batch_rewards) if batch_rewards else 0
            elapsed = time.time() - t0
            print(f'Iter {iteration+1:6d} | Balls:{num_balls} | '
                  f'AvgR={avg_reward:6.2f} | MaxR={max_run:5.1f} | '
                  f'Loss={loss.item():.4f} | {elapsed:.0f}s')
            if avg_reward > best_avg:
                best_avg = avg_reward
                torch.save(net.state_dict(), 'checkpoints/best_sequence.pt')
                print(f'  -> Best {avg_reward:.2f}')

    print(f'Done in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    train()
