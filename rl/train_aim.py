"""
Direct backprop training for pool aiming.
No RL -- pure supervised regression with differentiable geometry.

Setup: cue ball + one object ball + designated pocket
Network learns: given (cue, ball, pocket) -> aim angle
Loss: distance from cue ball ray to ghost ball + distance from
      object ball trajectory to pocket

The network must discover the ghost ball concept on its own.
"""
import os
import sys
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

# Table constants
TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0
BALL_RADIUS = 1.125
POCKET_RADIUS = 2.5
POCKET_RADIUS_SIDE = 2.75

# Pocket positions -- use AIM POINTS (mouth centers, not back corners)
# These match the browser demo's POCKET_AIM positions
_pr = POCKET_RADIUS
_prs = POCKET_RADIUS_SIDE
_mo = _pr * 0.45   # corner mouth offset
_smo = _prs * 0.15  # side mouth offset
POCKETS = torch.tensor([
    [_mo, _mo],                         # top-left
    [TABLE_LENGTH/2, _smo],             # top-side
    [TABLE_LENGTH - _mo, _mo],          # top-right
    [_mo, TABLE_WIDTH - _mo],           # bottom-left
    [TABLE_LENGTH/2, TABLE_WIDTH - _smo], # bottom-side
    [TABLE_LENGTH - _mo, TABLE_WIDTH - _mo], # bottom-right
], dtype=torch.float32)

POCKET_RADII = torch.tensor([
    POCKET_RADIUS, POCKET_RADIUS_SIDE, POCKET_RADIUS,
    POCKET_RADIUS, POCKET_RADIUS_SIDE, POCKET_RADIUS,
], dtype=torch.float32)

# Pocket approach angle limits (from the AI code)
# Side pockets: 50 deg from perpendicular, Corner: 65 deg from bisector
POCKET_MAX_ANGLE = torch.tensor([65, 50, 65, 65, 50, 65], dtype=torch.float32) * math.pi / 180

# Ideal approach directions for each pocket (ball-to-pocket direction)
POCKET_IDEAL_DIR = torch.tensor([
    [-1, -1],   # top-left
    [0, -1],    # top-side
    [1, -1],    # top-right
    [-1, 1],    # bottom-left
    [0, 1],     # bottom-side
    [1, 1],     # bottom-right
], dtype=torch.float32)
POCKET_IDEAL_DIR = POCKET_IDEAL_DIR / POCKET_IDEAL_DIR.norm(dim=1, keepdim=True)


class AimNetwork(nn.Module):
    """
    Network that learns to aim.
    Input: cue ball pos (2) + object ball pos (2) + pocket pos (2) = 6
    Output: aim angle (1)

    Must discover the ghost ball concept: the correct aim point is NOT
    the object ball center, but offset based on the pocket direction.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),  # aim angle output
        )

    def forward(self, x):
        """x: (batch, 6) -> aim_angle: (batch, 1)"""
        # Output is unconstrained, interpret as angle via 2*pi*sigmoid or just raw
        raw = self.net(x)
        # Use tanh * pi so output is in (-pi, pi), then we can shift to (0, 2pi)
        angle = torch.tanh(raw) * math.pi  # (-pi, pi)
        return angle.squeeze(-1)


def generate_batch(batch_size, device='cpu'):
    """Generate random cue ball + object ball + pocket layouts.
    Fully vectorized -- no Python loops. Fast on GPU.
    Returns: cue_pos, ball_pos, pocket_pos, pocket_idx"""
    R = BALL_RADIUS
    pockets = POCKETS.to(device)
    ideal_dirs = POCKET_IDEAL_DIR.to(device)
    max_angles = POCKET_MAX_ANGLE.to(device)

    # Random positions
    cue = torch.zeros(batch_size, 2, device=device)
    cue[:, 0] = torch.rand(batch_size, device=device) * (TABLE_LENGTH - 4*R) + 2*R
    cue[:, 1] = torch.rand(batch_size, device=device) * (TABLE_WIDTH - 4*R) + 2*R

    ball = torch.zeros(batch_size, 2, device=device)
    ball[:, 0] = torch.rand(batch_size, device=device) * (TABLE_LENGTH - 4*R) + 2*R
    ball[:, 1] = torch.rand(batch_size, device=device) * (TABLE_WIDTH - 4*R) + 2*R

    # Ensure balls aren't too close (regenerate close ones)
    dist = (cue - ball).norm(dim=1)
    too_close = dist < 5 * R
    while too_close.any():
        n = too_close.sum().item()
        ball[too_close, 0] = torch.rand(n, device=device) * (TABLE_LENGTH - 4*R) + 2*R
        ball[too_close, 1] = torch.rand(n, device=device) * (TABLE_WIDTH - 4*R) + 2*R
        dist = (cue - ball).norm(dim=1)
        too_close = dist < 5 * R

    # Vectorized pocket selection: score all 6 pockets for all samples at once
    # ball: (B, 2), pockets: (6, 2)
    # tp: direction from ball to each pocket: (B, 6, 2)
    tp = pockets.unsqueeze(0) - ball.unsqueeze(1)  # (B, 6, 2)
    tp_dist = tp.norm(dim=2).clamp(min=0.01)  # (B, 6)
    tp_n = tp / tp_dist.unsqueeze(2)  # (B, 6, 2)

    # Pocket approach angle check
    # dot product of tp_n with ideal direction for each pocket
    approach_dot = (tp_n * ideal_dirs.unsqueeze(0)).sum(dim=2)  # (B, 6)
    approach_angle = torch.acos(approach_dot.clamp(-1, 1))  # (B, 6)
    angle_ok = approach_angle < max_angles.unsqueeze(0)  # (B, 6)

    # Ghost ball positions: (B, 6, 2)
    ghost = ball.unsqueeze(1) - tp_n * (2 * R)  # (B, 6, 2)

    # Cut angle from cue ball to ghost
    cg = ghost - cue.unsqueeze(1)  # (B, 6, 2)
    cg_dist = cg.norm(dim=2).clamp(min=0.01)  # (B, 6)
    cg_n = cg / cg_dist.unsqueeze(2)  # (B, 6, 2)
    cut_dot = (cg_n * tp_n).sum(dim=2)  # (B, 6)
    cut_angle = torch.acos(cut_dot.clamp(-1, 1))  # (B, 6)
    cut_ok = cut_angle < math.radians(75)  # (B, 6)

    # Score: prefer short distance, small cut angle
    score = 50 - tp_dist * 0.3 - cg_dist * 0.2 - cut_angle * (180/math.pi) * 0.5
    # Mask out invalid pockets
    score = score.where(angle_ok & cut_ok, torch.tensor(-1e9, device=device))

    # Pick best pocket per sample
    best_p = score.argmax(dim=1)  # (B,)
    pocket_pos = pockets[best_p]  # (B, 2)

    return cue, ball, pocket_pos, best_p


def compute_loss(aim_angle, cue, ball, pocket_pos):
    """
    Differentiable loss measuring how close the object ball gets to the pocket.

    Approach: compute the actual collision normal from the aim angle,
    trace the object ball's post-collision trajectory, and measure
    how far it passes from the pocket center. Add a sharp pocketing
    bonus so there's a big jump from "near miss" to "in the pocket."
    """
    R = BALL_RADIUS
    ray_dx = torch.cos(aim_angle)
    ray_dy = torch.sin(aim_angle)

    # Direction from ball to pocket (ideal direction)
    tp = pocket_pos - ball
    tp_dist = tp.norm(dim=1, keepdim=True).clamp(min=0.01)
    tp_n = tp / tp_dist

    # Ghost ball position
    ghost = ball - tp_n * (2 * R)

    # Perpendicular distance of cue ball ray from ghost ball center
    to_ghost = ghost - cue
    ghost_perp = to_ghost[:, 0] * ray_dy - to_ghost[:, 1] * ray_dx
    ghost_proj = to_ghost[:, 0] * ray_dx + to_ghost[:, 1] * ray_dy

    # Perpendicular distance of cue ball ray from object ball center
    to_ball = ball - cue
    ball_perp = to_ball[:, 0] * ray_dy - to_ball[:, 1] * ray_dx
    ball_proj = to_ball[:, 0] * ray_dx + to_ball[:, 1] * ray_dy

    # Does the cue ball hit the object ball? (perp < 2R and proj > 0)
    hits_ball = (ball_perp.abs() < 2 * R) & (ball_proj > 0)

    # Collision normal: determined by the offset at contact
    # The object ball deflects in the direction from contact point to ball center.
    # contact offset = ball_perp (how far off-center the hit is)
    # Normal angle relative to aim: sin(theta) = ball_perp / (2R)
    # Object ball direction = rotated from aim direction by theta
    sin_theta = (ball_perp / (2 * R)).clamp(-0.999, 0.999)
    cos_theta = torch.sqrt(1 - sin_theta**2)

    # Object ball direction after collision (rotate aim direction)
    obj_dx = ray_dx * cos_theta - ray_dy * sin_theta
    obj_dy = ray_dx * sin_theta + ray_dy * cos_theta

    # How far does the object ball's trajectory pass from the pocket?
    to_pocket = pocket_pos - ball
    pocket_perp = to_pocket[:, 0] * obj_dy - to_pocket[:, 1] * obj_dx
    pocket_proj = to_pocket[:, 0] * obj_dx + to_pocket[:, 1] * obj_dy

    # --- Loss components ---

    # L1: Ghost miss (smooth gradient for rough aiming)
    L1 = ghost_perp ** 2

    # L2: Backward penalty
    L2 = torch.relu(-ghost_proj) * 5.0

    # L3: Pocket miss distance (the key metric)
    # Only applies when the cue ball actually hits the object ball
    L3 = torch.where(hits_ball, pocket_perp ** 2, ghost_perp ** 2 + 50.0)

    # L4: Object ball heading away from pocket
    L4 = torch.where(hits_ball, torch.relu(-pocket_proj) * 2.0, torch.zeros_like(pocket_proj))

    # L5: POCKETING BONUS -- sharp step function
    # When pocket_perp < pocket_radius: big reward (-20)
    # When pocket_perp > pocket_radius: nothing
    # Use a steep sigmoid for differentiability
    pocket_r = 2.0
    pocket_miss = pocket_perp.abs()
    pocketing_score = torch.sigmoid(8.0 * (pocket_r - pocket_miss))
    L5 = torch.where(hits_ball, -20.0 * pocketing_score, torch.zeros_like(pocketing_score))

    loss = L1 + L2 + L3 + L4 + L5

    # Clamp loss per sample to prevent NaN
    loss = loss.clamp(-30, 200)

    # Stats
    pocketed = hits_ball & (pocket_miss < pocket_r)
    pocketed_frac = pocketed.float().mean().item()

    return (loss.mean(),
            L1.mean().item(),
            L3.mean().item(),
            ghost_perp.abs().mean().item(),
            pocketed_frac)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    net = AimNetwork(hidden=128).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    num_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {num_params:,}")
    print()

    batch_size = 2048
    num_iters = 50000
    log_interval = 200

    best_perp = float('inf')
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()

    for i in range(num_iters):
        # Generate random layouts
        cue, ball, pocket_pos, pocket_idx = generate_batch(batch_size, device)

        # Network input: (cue_x, cue_y, ball_x, ball_y, pocket_x, pocket_y) normalized
        inp = torch.cat([
            cue / torch.tensor([TABLE_LENGTH, TABLE_WIDTH], device=device),
            ball / torch.tensor([TABLE_LENGTH, TABLE_WIDTH], device=device),
            pocket_pos / torch.tensor([TABLE_LENGTH, TABLE_WIDTH], device=device),
        ], dim=1)  # (batch, 6)

        # Forward
        aim_angle = net(inp)

        # Compute differentiable loss
        loss, l1_val, l2_val, avg_perp, pocket_rate = compute_loss(aim_angle, cue, ball, pocket_pos)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"Iter {i+1:6d} | Loss {loss.item():8.4f} | "
                  f"Ghost {l1_val:7.4f} | Pocket {l2_val:7.4f} | "
                  f"Perp {avg_perp:5.3f}in | "
                  f"Pocketed {pocket_rate*100:5.1f}% | "
                  f"LR {lr:.1e} | {elapsed:.0f}s")

            if avg_perp < best_perp:
                best_perp = avg_perp
                torch.save(net.state_dict(), os.path.join(save_dir, 'best_aim.pt'))
                print(f"  -> New best: perp={avg_perp:.3f}in, pocket={pocket_rate*100:.1f}%")

    # Final save
    torch.save(net.state_dict(), os.path.join(save_dir, 'final_aim.pt'))
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best perpendicular miss: {best_perp:.3f} inches")
    print(f"(Ball radius = {BALL_RADIUS} inches, so < {BALL_RADIUS:.3f} = contact)")


def test():
    """Test a trained model."""
    device = torch.device('cpu')
    net = AimNetwork(hidden=128)

    ckpt = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_aim.pt')
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print("Loaded checkpoint")
    else:
        print("No checkpoint found, using random weights")

    net.eval()

    # Test on 1000 random layouts
    hits = 0
    pockets = 0
    total = 1000
    perps = []

    with torch.no_grad():
        cue, ball, pocket_pos, pocket_idx = generate_batch(total)
        inp = torch.cat([
            cue / torch.tensor([TABLE_LENGTH, TABLE_WIDTH]),
            ball / torch.tensor([TABLE_LENGTH, TABLE_WIDTH]),
            pocket_pos / torch.tensor([TABLE_LENGTH, TABLE_WIDTH]),
        ], dim=1)

        aim_angle = net(inp)

        for i in range(total):
            angle = aim_angle[i].item()
            dx, dy = math.cos(angle), math.sin(angle)
            cx, cy = cue[i, 0].item(), cue[i, 1].item()
            bx, by = ball[i, 0].item(), ball[i, 1].item()

            # Ghost ball position
            px, py = pocket_pos[i, 0].item(), pocket_pos[i, 1].item()
            tpx, tpy = px - bx, py - by
            tp_dist = math.sqrt(tpx*tpx + tpy*tpy)
            if tp_dist < 0.01: continue
            tp_nx, tp_ny = tpx/tp_dist, tpy/tp_dist
            gx = bx - tp_nx * 2 * BALL_RADIUS
            gy = by - tp_ny * 2 * BALL_RADIUS

            # Perpendicular distance from ray to ghost
            to_gx, to_gy = gx - cx, gy - cy
            perp = abs(to_gx * dy - to_gy * dx)
            perps.append(perp)

            if perp < 2 * BALL_RADIUS:
                hits += 1
                # Check if object ball direction leads to pocket
                # (simplified: if perp < 0.5 * R, it's close enough to pocket)
                if perp < BALL_RADIUS * 0.5:
                    pockets += 1

    avg_perp = np.mean(perps)
    print(f"\nTest results on {total} layouts:")
    print(f"  Avg perpendicular miss: {avg_perp:.2f} inches")
    print(f"  Hit rate (perp < 2R): {hits/total*100:.1f}%")
    print(f"  Pocket rate (perp < 0.5R): {pockets/total*100:.1f}%")
    print(f"  (Random would be ~25% hit, ~2% pocket)")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        train()
