"""
Train with ReLU, weighted loss for steep cuts.
Steep cuts (40-75 deg) get 3x the loss weight so the network
works harder to get them right.
"""
import torch, torch.nn as nn, torch.optim as optim, math, time, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import generate_batch, BALL_RADIUS, TABLE_LENGTH as TL, TABLE_WIDTH as TW
from train_relu import AimNetReLU

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R = BALL_RADIUS

    # Load the previous best as starting point (fine-tune)
    net = AimNetReLU(256).to(device)
    ckpt = 'checkpoints/best_aim_relu.pt'
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device))
        print(f'Loaded checkpoint, fine-tuning')
    else:
        print(f'Training from scratch')

    opt = optim.Adam(net.parameters(), lr=1e-4)  # lower LR for fine-tuning
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50000)
    print(f'Device: {device}, Params: {sum(p.numel() for p in net.parameters()):,}')

    best = 999
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    for i in range(50000):
        cue, ball, pp, _ = generate_batch(2048, device)

        # Compute ghost ball and correct angle
        tp = pp - ball
        tpd = tp.norm(dim=1, keepdim=True).clamp(min=0.01)
        tp_n = tp / tpd
        ghost = ball - tp_n * (2*R)
        correct = torch.atan2(ghost[:,1]-cue[:,1], ghost[:,0]-cue[:,0])

        # Compute cut angle for each sample
        cg = ghost - cue
        cgd = cg.norm(dim=1).clamp(min=0.01)
        cg_n = cg / cgd.unsqueeze(1)
        cut_dot = (cg_n * tp_n.squeeze()).sum(dim=1)
        cut_angle_deg = torch.acos(cut_dot.clamp(-1, 1)) * 180 / math.pi

        # Forward
        inp = torch.cat([cue/torch.tensor([TL,TW],device=device),
                         ball/torch.tensor([TL,TW],device=device),
                         pp/torch.tensor([TL,TW],device=device)], dim=1)
        pred = net(inp)

        # Angular loss per sample
        angle_loss = 1 - torch.cos(pred - correct)

        # Weight steep cuts 3x more heavily
        # Straight (0-20 deg): weight 1.0
        # Medium (20-40 deg): weight 1.5
        # Steep (40-60 deg): weight 3.0
        # Very steep (60-75 deg): weight 5.0
        weight = torch.ones_like(cut_angle_deg)
        weight = weight + (cut_angle_deg > 20).float() * 0.5
        weight = weight + (cut_angle_deg > 40).float() * 1.5
        weight = weight + (cut_angle_deg > 60).float() * 2.0

        loss = (angle_loss * weight).mean() * 100

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if (i+1) % 1000 == 0:
            with torch.no_grad():
                err = (pred - correct).abs() * 180 / math.pi

                # Break down by cut angle range
                straight = cut_angle_deg < 20
                medium = (cut_angle_deg >= 20) & (cut_angle_deg < 40)
                steep = (cut_angle_deg >= 40) & (cut_angle_deg < 60)
                very_steep = cut_angle_deg >= 60

                def stats(mask):
                    if mask.sum() == 0: return 0, 0, 0
                    e = err[mask]
                    return e.mean().item(), e.median().item(), (e < 2).float().mean().item() * 100

                s_mean, s_med, s_pct = stats(straight)
                m_mean, m_med, m_pct = stats(medium)
                st_mean, st_med, st_pct = stats(steep)
                vs_mean, vs_med, vs_pct = stats(very_steep)

            elapsed = time.time() - t0
            overall_mean = err.mean().item()
            overall_w2 = (err < 2).float().mean().item() * 100

            print(f'Iter {i+1:6d} | {elapsed:5.0f}s | Overall: mean={overall_mean:.1f} <2d={overall_w2:.0f}%')
            print(f'  Straight(<20): med={s_med:.2f} <2d={s_pct:.0f}% | '
                  f'Medium(20-40): med={m_med:.2f} <2d={m_pct:.0f}% | '
                  f'Steep(40-60): med={st_med:.2f} <2d={st_pct:.0f}% | '
                  f'VySteep(60+): med={vs_med:.2f} <2d={vs_pct:.0f}%')

            if overall_mean < best:
                best = overall_mean
                torch.save(net.state_dict(), 'checkpoints/best_aim_relu.pt')
                print(f'  -> Best: {overall_mean:.1f}')

    # Export
    net.cpu().eval()
    model = {'layers': [], 'activation': 'relu'}
    for name, param in net.net.named_parameters():
        model['layers'].append({'name': name, 'shape': list(param.shape),
                                'data': param.detach().numpy().tolist()})
    with open('../js/aim_model.json', 'w') as f:
        json.dump(model, f)
    print(f'\nExported. Done in {time.time()-t0:.0f}s, best={best:.1f}deg')

if __name__ == '__main__':
    train()
