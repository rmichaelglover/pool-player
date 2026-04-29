"""
Aim training with (cos, sin) output to eliminate angle discontinuity.
The network outputs 2 values: (cos theta, sin theta).
Angle is recovered as atan2(sin, cos) -- no wrapping issues at +/-180.
"""
import torch, torch.nn as nn, torch.optim as optim, math, time, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import generate_batch, BALL_RADIUS as R, TABLE_LENGTH as TL, TABLE_WIDTH as TW

class AimNetCosSin(nn.Module):
    """Outputs (cos theta, sin theta) instead of theta directly."""
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2),  # output: (cos, sin)
        )

    def forward(self, x):
        out = self.net(x)
        # Normalize to unit circle to ensure valid (cos, sin)
        norm = out.norm(dim=1, keepdim=True).clamp(min=0.001)
        return out / norm  # (batch, 2) = (cos theta, sin theta)

    def get_angle(self, x):
        cs = self.forward(x)
        return torch.atan2(cs[:, 1], cs[:, 0])

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AimNetCosSin(512).to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200000)
    print(f'{sum(p.numel() for p in net.parameters()):,} params, {device}')

    best = 999
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    for i in range(200000):
        cue, ball, pp, _ = generate_batch(2048, device)
        tp = pp - ball; tpd = tp.norm(dim=1, keepdim=True).clamp(min=0.01)
        tp_n = tp / tpd
        ghost = ball - tp_n * (2*R)

        # Target: (cos, sin) of the correct angle
        dg = ghost - cue  # (batch, 2)
        dg_dist = dg.norm(dim=1, keepdim=True).clamp(min=0.001)
        target_cs = dg / dg_dist  # (batch, 2) = (cos theta, sin theta)

        # Cut angle for weighting
        cg_n = dg / dg_dist
        cut_dot = (cg_n * tp_n.squeeze()).sum(dim=1)
        cut_deg = torch.acos(cut_dot.clamp(-1,1)) * 180/math.pi

        inp = torch.cat([cue/torch.tensor([TL,TW],device=device),
                         ball/torch.tensor([TL,TW],device=device),
                         pp/torch.tensor([TL,TW],device=device)], dim=1)
        pred_cs = net(inp)

        # Loss: 1 - cos(angle between predicted and target direction)
        # = 1 - dot(pred, target) since both are unit vectors
        dot_product = (pred_cs * target_cs).sum(dim=1)
        angle_loss = 1 - dot_product  # 0 when perfect, 2 when opposite

        # Weight steep cuts more
        weight = 1 + (cut_deg > 20).float()*0.5 + (cut_deg > 40).float()*1.5 + (cut_deg > 60).float()*2
        loss = (angle_loss * weight).mean() * 100

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if (i+1) % 2000 == 0:
            with torch.no_grad():
                pred_angle = torch.atan2(pred_cs[:, 1], pred_cs[:, 0])
                correct_angle = torch.atan2(target_cs[:, 1], target_cs[:, 0])
                # Angular error (handles wrapping correctly)
                err_rad = torch.atan2(torch.sin(pred_angle - correct_angle),
                                      torch.cos(pred_angle - correct_angle)).abs()
                err_deg = err_rad * 180 / math.pi
                w1 = (err_deg < 1).float().mean().item()
                w2 = (err_deg < 2).float().mean().item()
                outlier = (err_deg > 10).float().mean().item()
            me = err_deg.mean().item(); md = err_deg.median().item()
            print(f'Iter {i+1:6d} | Mean {me:5.1f} | Med {md:4.2f} | <1d {w1*100:.0f}% | <2d {w2*100:.0f}% | >10d {outlier*100:.1f}% | {time.time()-t0:.0f}s')
            if me < best:
                best = me; torch.save(net.state_dict(), 'checkpoints/best_aim_cossin.pt')
                print(f'  -> Best {me:.1f}')

    # Export
    export_model(net)
    print(f'Done in {time.time()-t0:.0f}s, best={best:.1f}deg')

def export_model(net=None):
    if net is None:
        net = AimNetCosSin(512)
        net.load_state_dict(torch.load('checkpoints/best_aim_cossin.pt', map_location='cpu'))
    net.cpu().eval()
    model = {'layers': [], 'activation': 'relu', 'output': 'cossin'}
    for name, param in net.net.named_parameters():
        model['layers'].append({'name': name, 'shape': list(param.shape),
                                'data': param.detach().numpy().tolist()})
    with open('../js/aim_model.json', 'w') as f:
        json.dump(model, f)
    print(f'Exported ({os.path.getsize("../js/aim_model.json")/1024:.0f} KB)')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        export_model()
    else:
        train()
