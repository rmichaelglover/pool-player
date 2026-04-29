"""Train with ReLU for exact JS inference (no GELU approximation error)."""
import torch, torch.nn as nn, torch.optim as optim, math, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import generate_batch, BALL_RADIUS, TABLE_LENGTH as TL, TABLE_WIDTH as TW

class AimNetReLU(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return (torch.tanh(self.net(x)) * math.pi).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = AimNetReLU(256).to(device)
opt = optim.Adam(net.parameters(), lr=5e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100000)
R = BALL_RADIUS
print(f'ReLU, {sum(p.numel() for p in net.parameters()):,} params, {device}')

best = 999; os.makedirs('checkpoints', exist_ok=True); t0 = time.time()
for i in range(100000):
    cue, ball, pp, _ = generate_batch(2048, device)
    tp = pp - ball; tpd = tp.norm(dim=1, keepdim=True).clamp(min=0.01)
    ghost = ball - (tp/tpd)*(2*R)
    correct = torch.atan2(ghost[:,1]-cue[:,1], ghost[:,0]-cue[:,0])
    inp = torch.cat([cue/torch.tensor([TL,TW],device=device),
                     ball/torch.tensor([TL,TW],device=device),
                     pp/torch.tensor([TL,TW],device=device)], dim=1)
    pred = net(inp)
    loss = (1 - torch.cos(pred - correct)).mean() * 100
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(); sched.step()
    if (i+1) % 2000 == 0:
        with torch.no_grad():
            err = (pred-correct).abs()*180/math.pi
        me=err.mean().item(); md=err.median().item()
        w1=(err<1).float().mean().item(); w2=(err<2).float().mean().item()
        print(f'Iter {i+1:6d} | Loss {loss.item():6.3f} | Mean {me:5.1f} | Med {md:4.2f} | <1d {w1*100:.0f}% | <2d {w2*100:.0f}% | {time.time()-t0:.0f}s')
        if me < best: best=me; torch.save(net.state_dict(),'checkpoints/best_aim_relu.pt'); print(f'  -> Best {me:.1f}')
print(f'Done in {time.time()-t0:.0f}s, best={best:.1f}')
