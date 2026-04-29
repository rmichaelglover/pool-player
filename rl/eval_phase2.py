"""Deterministic evaluation of a Phase 2 checkpoint."""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from train_phase2 import Phase2Env


def evaluate(ckpt_path, num_shots=1000, device_name='cpu', deterministic=True):
    device = torch.device(device_name)
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2).to(device)
    net.log_std = torch.nn.Parameter(torch.full((2,), -0.5).to(device))
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()

    env = Phase2Env()
    obs = env.reset()
    hits = pockets = 0
    with torch.no_grad():
        for _ in range(num_shots):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _ = net.get_action(obs_t, deterministic=deterministic)
            a = action[0].cpu().numpy()
            aim_angle = math.atan2(float(a[0]), float(a[1]))
            _, _, info = env.step(aim_angle)
            if info['hit']: hits += 1
            if info['pocketed']: pockets += 1
            obs = env.reset()
    return hits / num_shots, pockets / num_shots


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/phase2_best.pt')
    parser.add_argument('--shots', type=int, default=1000)
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    hr, pr = evaluate(args.ckpt, args.shots, deterministic=not args.stochastic)
    mode = 'stochastic' if args.stochastic else 'deterministic'
    print(f'Checkpoint: {args.ckpt}')
    print(f'Mode: {mode}, {args.shots} shots')
    print(f'  Hit rate:    {hr:.1%}')
    print(f'  Pocket rate: {pr:.1%}')
