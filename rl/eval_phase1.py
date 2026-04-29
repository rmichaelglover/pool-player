"""
Deterministic evaluation of a Phase 1 checkpoint.

Stochastic training hit rate inflates/deflates with exploration noise. To
measure what the policy has actually learned, sample actions from the MEAN
(deterministic=True) and run on a fresh distribution of 1-ball setups.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_attention_net import PoolAttentionNet
from train_curriculum import Phase1Env


def evaluate(ckpt_path, num_shots=1000, device_name='cpu', deterministic=True, net_kwargs=None):
    device = torch.device(device_name)
    kwargs = dict(embed_dim=96, num_heads=6, num_layers=4, act_dim=2)
    if net_kwargs:
        kwargs.update(net_kwargs)
    net = PoolAttentionNet(**kwargs).to(device)
    # Match phase1 log_std shape
    net.log_std = torch.nn.Parameter(torch.full((2,), -0.5).to(device))

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()

    env = Phase1Env()
    obs = env.reset()
    hits = 0
    pockets = 0
    angle_errors = []

    with torch.no_grad():
        for _ in range(num_shots):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _ = net.get_action(obs_t, deterministic=deterministic)
            a = action[0].cpu().numpy()
            aim_angle = math.atan2(float(a[0]), float(a[1]))

            # Compute angle-to-ball for comparison
            dx = env.ball_pos[0] - env.cue[0]
            dy = env.ball_pos[1] - env.cue[1]
            optimal_angle = math.atan2(dy, dx)
            err = abs(aim_angle - optimal_angle)
            if err > math.pi:
                err = 2 * math.pi - err
            angle_errors.append(math.degrees(err))

            _, done, info = env.step(aim_angle, 30.0, 0.0)
            if info['hit']:
                hits += 1
            if info['pocketed']:
                pockets += 1
            # Reset every shot for clean measurement
            obs = env.reset()

    hr = hits / num_shots
    pr = pockets / num_shots
    mean_err = np.mean(angle_errors)
    median_err = np.median(angle_errors)
    return {
        'hit_rate': hr,
        'pocket_rate': pr,
        'mean_angle_err_deg': mean_err,
        'median_angle_err_deg': median_err,
        'n_shots': num_shots,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/phase1_best.pt')
    parser.add_argument('--shots', type=int, default=1000)
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    r = evaluate(args.ckpt, num_shots=args.shots,
                 deterministic=not args.stochastic)
    mode = 'stochastic' if args.stochastic else 'deterministic'
    print(f'Checkpoint: {args.ckpt}')
    print(f'Mode: {mode}, {r["n_shots"]} shots')
    print(f'  Hit rate:     {r["hit_rate"]:.1%}')
    print(f'  Pocket rate:  {r["pocket_rate"]:.1%}')
    print(f'  Mean aim err: {r["mean_angle_err_deg"]:.1f} deg')
    print(f'  Median err:   {r["median_angle_err_deg"]:.1f} deg')
