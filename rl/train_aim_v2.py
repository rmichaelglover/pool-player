"""
Aim training v2: Direct angle loss.
Trains to predict the correct aim angle (cue -> ghost ball direction).
Loss = 1 - cos(predicted - correct), which is 0 when perfect, 2 when opposite.
"""
import os, sys, math, time
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import AimNetwork, generate_batch, BALL_RADIUS, TABLE_LENGTH, TABLE_WIDTH

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AimNetwork(hidden=256).to(device)
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)
    R = BALL_RADIUS; TL = TABLE_LENGTH; TW = TABLE_WIDTH
    print(f'Device: {device}, Params: {sum(p.numel() for p in net.parameters()):,}')

    best_err = 999
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    for i in range(100000):
        cue, ball, pocket_pos, _ = generate_batch(2048, device)
        tp = pocket_pos - ball
        tp_dist = tp.norm(dim=1, keepdim=True).clamp(min=0.01)
        tp_n = tp / tp_dist
        ghost = ball - tp_n * (2*R)
        correct = torch.atan2(ghost[:,1]-cue[:,1], ghost[:,0]-cue[:,0])

        inp = torch.cat([cue/torch.tensor([TL,TW],device=device),
                         ball/torch.tensor([TL,TW],device=device),
                         pocket_pos/torch.tensor([TL,TW],device=device)], dim=1)
        pred = net(inp)

        loss = (1 - torch.cos(pred - correct)).mean() * 100
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        if (i+1) % 500 == 0:
            with torch.no_grad():
                err = (pred - correct).abs() * 180/math.pi
            mean_e = err.mean().item()
            med_e = err.median().item()
            w2 = (err<2).float().mean().item()
            w5 = (err<5).float().mean().item()
            elapsed = time.time() - t0
            print(f'Iter {i+1:6d} | Loss {loss.item():7.3f} | '
                  f'Mean {mean_e:5.1f}deg | Med {med_e:5.1f}deg | '
                  f'<2deg {w2*100:4.0f}% | <5deg {w5*100:4.0f}% | {elapsed:.0f}s')
            if mean_e < best_err:
                best_err = mean_e
                torch.save(net.state_dict(), 'checkpoints/best_aim.pt')
                print(f'  -> Best: {mean_e:.1f}deg')

    print(f'Done. Best: {best_err:.1f}deg in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    train()
