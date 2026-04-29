"""Bigger network (4 layers, 512 hidden) to eliminate the 6% outliers."""
import torch, torch.nn as nn, torch.optim as optim, math, time, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import generate_batch, BALL_RADIUS as R, TABLE_LENGTH as TL, TABLE_WIDTH as TW

class AimNetBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return (torch.tanh(self.net(x)) * math.pi).squeeze(-1)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AimNetBig().to(device)
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
        correct = torch.atan2(ghost[:,1]-cue[:,1], ghost[:,0]-cue[:,0])

        # Cut angle for weighting
        cg = ghost - cue; cgd = cg.norm(dim=1).clamp(min=0.01)
        cut_dot = (cg/cgd.unsqueeze(1) * tp_n.squeeze()).sum(dim=1)
        cut_deg = torch.acos(cut_dot.clamp(-1,1)) * 180/math.pi

        inp = torch.cat([cue/torch.tensor([TL,TW],device=device),
                         ball/torch.tensor([TL,TW],device=device),
                         pp/torch.tensor([TL,TW],device=device)], dim=1)
        pred = net(inp)

        # Weighted angular loss
        angle_loss = 1 - torch.cos(pred - correct)
        weight = 1 + (cut_deg > 20).float()*0.5 + (cut_deg > 40).float()*1.5 + (cut_deg > 60).float()*2
        loss = (angle_loss * weight).mean() * 100

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if (i+1) % 2000 == 0:
            with torch.no_grad():
                err = (pred-correct).abs()*180/math.pi
                w1 = (err<1).float().mean().item()
                w2 = (err<2).float().mean().item()
                outlier = (err>10).float().mean().item()
            me = err.mean().item(); md = err.median().item()
            print(f'Iter {i+1:6d} | Mean {me:5.1f} | Med {md:4.2f} | <1d {w1*100:.0f}% | <2d {w2*100:.0f}% | >10d {outlier*100:.1f}% | {time.time()-t0:.0f}s')
            if me < best:
                best = me; torch.save(net.state_dict(), 'checkpoints/best_aim_big.pt')
                print(f'  -> Best {me:.1f}')

    # Export
    net.cpu().eval()
    model = {'layers': [], 'activation': 'relu'}
    for name, param in net.net.named_parameters():
        model['layers'].append({'name': name, 'shape': list(param.shape), 'data': param.detach().numpy().tolist()})
    with open('../js/aim_model.json', 'w') as f: json.dump(model, f)
    print(f'Exported. Done in {time.time()-t0:.0f}s, best={best:.1f}deg')

if __name__ == '__main__':
    train()
