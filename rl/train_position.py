"""
Phase 2: Learn position play (speed + english).
Network inputs: cue pos, ball pos, pocket pos, target cue ball position = 8 dims
Network outputs: cos(aim), sin(aim), force, contact_y (draw/follow), contact_x (english) = 5 dims

Loss:
  L1: angular aim error (must pocket the ball)
  L2: distance from cue ball final position to target (only when pocketed)

Uses simplified geometric model for cue ball position after collision:
  - Stun shot (center hit): cue deflects ~90 deg from aim, travels based on force
  - Follow (top spin): deflection < 90 deg, cue follows through
  - Draw (back spin): cue reverses along aim line
  - English: slight lateral shift of deflection angle
"""
import torch, torch.nn as nn, torch.optim as optim, math, time, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import generate_batch as gen_batch_base, BALL_RADIUS as R
from train_aim import TABLE_LENGTH as TL, TABLE_WIDTH as TW

class PositionNet(nn.Module):
    """Outputs (cos, sin, force, contact_y, contact_x)."""
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden), nn.ReLU(),    # 8 inputs (added target pos)
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 5),                   # 5 outputs
        )

    def forward(self, x):
        out = self.net(x)
        # cos, sin: normalize to unit circle
        cs = out[:, :2]
        norm = cs.norm(dim=1, keepdim=True).clamp(min=0.001)
        cs = cs / norm
        # force: sigmoid -> 0 to 1 (mapped to min-max speed later)
        force = torch.sigmoid(out[:, 2])
        # contact_y: tanh -> -1 to 1 (draw to follow)
        contact_y = torch.tanh(out[:, 3])
        # contact_x: tanh -> -1 to 1 (left to right english)
        contact_x = torch.tanh(out[:, 4])
        return cs, force, contact_y, contact_x


def generate_batch(batch_size, device='cpu'):
    """Generate layouts with a random target position for the cue ball."""
    cue, ball, pocket_pos, pocket_idx = gen_batch_base(batch_size, device)
    # Random target position (where cue ball should end up)
    target = torch.zeros(batch_size, 2, device=device)
    target[:, 0] = torch.rand(batch_size, device=device) * (TL - 6*R) + 3*R
    target[:, 1] = torch.rand(batch_size, device=device) * (TW - 6*R) + 3*R
    return cue, ball, pocket_pos, pocket_idx, target


def simulate_cue_position(ghost, aim_cs, force, contact_y, contact_x,
                          ball, pocket_pos):
    """
    Simplified geometric model of where the cue ball ends up after collision.

    After hitting the object ball at the ghost position:
    - Stun (center): cue deflects 90 deg perpendicular to aim, travels short distance
    - Follow (contact_y > 0): deflection < 90 deg, cue follows through toward pocket
    - Draw (contact_y < 0): cue reverses back along aim line
    - English (contact_x): slight lateral shift of deflection
    - Force: controls total travel distance after collision

    Returns: (batch, 2) predicted cue ball final position
    """
    batch = ghost.shape[0]
    aim_dx = aim_cs[:, 0]  # cos
    aim_dy = aim_cs[:, 1]  # sin

    # Pocket direction from ball
    tp = pocket_pos - ball
    tpd = tp.norm(dim=1, keepdim=True).clamp(min=0.01)
    tp_n = (tp / tpd).squeeze(1)  # (batch, 2) unit direction ball->pocket

    # Perpendicular to pocket direction (the natural 90-deg deflection direction)
    perp_x = -tp_n[:, 1]
    perp_y = tp_n[:, 0]

    # Choose which side of perpendicular (away from pocket line)
    cross = aim_dx * tp_n[:, 1] - aim_dy * tp_n[:, 0]
    sign = torch.sign(cross)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    perp_x = perp_x * sign
    perp_y = perp_y * sign

    # Base deflection direction depends on contact_y:
    # contact_y = 0 (stun): pure perpendicular (90 deg deflection)
    # contact_y > 0 (follow): blend toward pocket direction (< 90 deg)
    # contact_y < 0 (draw): reverse along aim line
    follow_blend = contact_y.clamp(0, 1)  # 0 to 1
    draw_blend = (-contact_y).clamp(0, 1)  # 0 to 1
    stun_blend = 1 - follow_blend - draw_blend

    # Stun direction: perpendicular
    # Follow direction: toward pocket (same as ball went)
    # Draw direction: opposite of aim
    dir_x = (stun_blend * perp_x +
             follow_blend * tp_n[:, 0] +
             draw_blend * (-aim_dx))
    dir_y = (stun_blend * perp_y +
             follow_blend * tp_n[:, 1] +
             draw_blend * (-aim_dy))

    # Add english effect: slight lateral shift
    dir_x = dir_x + contact_x * perp_x * 0.3
    dir_y = dir_y + contact_x * perp_y * 0.3

    # Normalize direction
    dir_len = torch.sqrt(dir_x**2 + dir_y**2).clamp(min=0.001)
    dir_x = dir_x / dir_len
    dir_y = dir_y / dir_len

    # Travel distance depends on force and spin type
    # Stun: moderate travel
    # Follow: longer travel (more energy retained)
    # Draw: shorter travel (energy used to reverse)
    base_dist = force * 40  # 0 to 40 inches
    dist = base_dist * (1 + follow_blend * 0.5 - draw_blend * 0.3)

    # Final position
    final_x = (ghost[:, 0] + dir_x * dist).clamp(R, TL - R)
    final_y = (ghost[:, 1] + dir_y * dist).clamp(R, TW - R)

    return torch.stack([final_x, final_y], dim=1)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PositionNet(512).to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200000)
    print(f'{sum(p.numel() for p in net.parameters()):,} params, {device}')

    best = 999
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    for i in range(200000):
        cue, ball, pp, _, target = generate_batch(2048, device)

        # Ghost ball and correct aim
        tp = pp - ball; tpd = tp.norm(dim=1, keepdim=True).clamp(min=0.01)
        tp_n = tp / tpd
        ghost = ball - tp_n * (2*R)
        dg = ghost - cue; dg_dist = dg.norm(dim=1, keepdim=True).clamp(min=0.001)
        target_cs = dg / dg_dist  # correct (cos, sin)

        # Cut angle for weighting
        cg_n = dg / dg_dist
        cut_dot = (cg_n * tp_n.squeeze()).sum(dim=1)
        cut_deg = torch.acos(cut_dot.clamp(-1,1)) * 180/math.pi

        # Network input: cue + ball + pocket + target = 8 dims
        scale = torch.tensor([TL, TW], device=device)
        inp = torch.cat([cue/scale, ball/scale, pp/scale, target/scale], dim=1)

        pred_cs, force, contact_y, contact_x = net(inp)

        # L1: Aim accuracy (must pocket the ball)
        aim_dot = (pred_cs * target_cs).sum(dim=1)
        L1 = (1 - aim_dot)
        weight = 1 + (cut_deg > 20).float()*0.5 + (cut_deg > 40).float()*1.5 + (cut_deg > 60).float()*2
        L1 = L1 * weight * 200  # heavy weight: pocketing is non-negotiable

        # L2: Cue ball position (only matters when aim is good enough to pocket)
        cue_final = simulate_cue_position(ghost, pred_cs, force, contact_y, contact_x,
                                          ball, pp)
        position_error = (cue_final - target).norm(dim=1)
        # Only count position error when aim is good (would pocket)
        aim_good = (aim_dot > 0.9999)  # < ~0.8 degrees
        L2 = torch.where(aim_good, position_error, torch.zeros_like(position_error))

        loss = L1.mean() + L2.mean()

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if (i+1) % 2000 == 0:
            with torch.no_grad():
                pred_angle = torch.atan2(pred_cs[:,1], pred_cs[:,0])
                correct_angle = torch.atan2(target_cs[:,1], target_cs[:,0])
                aim_err = torch.atan2(torch.sin(pred_angle-correct_angle),
                                      torch.cos(pred_angle-correct_angle)).abs() * 180/math.pi
                good_aim = aim_err < 1
                pos_err = position_error[good_aim].mean().item() if good_aim.sum() > 0 else 99
                avg_force = force.mean().item()
                avg_cy = contact_y.mean().item()
                avg_cx = contact_x.mean().item()
            elapsed = time.time() - t0
            aim_w1 = (aim_err < 1).float().mean().item()
            aim_w2 = (aim_err < 2).float().mean().item()
            print(f'Iter {i+1:6d} | Aim: med={aim_err.median().item():.2f} <1d={aim_w1*100:.0f}% <2d={aim_w2*100:.0f}% | '
                  f'Pos: {pos_err:.1f}in (when aimed) | '
                  f'F={avg_force:.2f} CY={avg_cy:.2f} CX={avg_cx:.2f} | {elapsed:.0f}s')
            combined = aim_err.mean().item() + pos_err * 0.1
            if combined < best:
                best = combined
                torch.save(net.state_dict(), 'checkpoints/best_position.pt')
                print(f'  -> Best')

    # Export
    export_model(net)
    print(f'Done in {time.time()-t0:.0f}s')

def export_model(net=None):
    if net is None:
        net = PositionNet(512)
        net.load_state_dict(torch.load('checkpoints/best_position.pt', map_location='cpu'))
    net.cpu().eval()
    model = {'layers': [], 'activation': 'relu', 'output': 'position'}
    for name, param in net.net.named_parameters():
        model['layers'].append({'name': name, 'shape': list(param.shape),
                                'data': param.detach().numpy().tolist()})
    with open('../js/aim_model_position.json', 'w') as f:
        json.dump(model, f)
    print(f'Exported ({os.path.getsize("../js/aim_model_position.json")/1024:.0f} KB)')

if __name__ == '__main__':
    train()
