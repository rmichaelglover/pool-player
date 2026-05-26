"""
Head-to-head evaluation for 8-ball checkpoints.

Plays N games between two models (or a model vs a heuristic baseline),
reports win rates, average game length, foul rate, and Elo difference.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eight_ball_net import EightBallNet, MAX_SHOTS
from eight_ball_env import EightBallEnv


def load_model(path, device='cpu', embed_dim=128, num_heads=8, num_layers=4):
    net = EightBallNet(embed_dim=embed_dim, num_heads=num_heads,
                       num_layers=num_layers).to(device)
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(state)
    return net


def obs_to_batch(obs, device):
    return {
        'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(device),
        'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(device),
        'ball_group': torch.from_numpy(obs.ball_group).unsqueeze(0).to(device),
        'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(device),
        'game_state': torch.from_numpy(obs.game_state).unsqueeze(0).to(device),
        'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(device),
        'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(device),
    }


def play_game(net_a, net_b, device='cpu', env_kwargs=None):
    """Play one game. net_a is player 0, net_b is player 1.
    Returns (winner, total_shots, fouls_p0, fouls_p1)."""
    kw = env_kwargs or {}
    env = EightBallEnv(**kw)
    obs = env.reset()
    nets = {0: net_a, 1: net_b}
    fouls = {0: 0, 1: 0}

    for _ in range(200):
        player = env.current_player
        net = nets[player]
        obs_batch = obs_to_batch(obs, device)

        with torch.no_grad():
            action_idx, force_raw, spin_raw, _, _ = net.get_action(
                obs_batch, deterministic=True)

        obs, reward, done, info = env.step(
            action_idx.item(), force_raw.item(), spin_raw.item(), obs)

        if 'foul' in info:
            fouls[info.get('player', player)] += 1

        if done:
            break

    return env.winner, env.total_shots, fouls[0], fouls[1]


def evaluate(model_a_path, model_b_path=None, num_games=100,
             device='cpu', env_kwargs=None):
    """Run head-to-head evaluation. If model_b_path is None, model_b is
    a fresh (random) network."""
    device = torch.device(device)
    net_a = load_model(model_a_path, device)
    net_b = load_model(model_b_path, device)
    net_a.eval()
    net_b.eval()

    wins = {0: 0, 1: 0, None: 0}
    lengths = []
    fouls_a = []
    fouls_b = []

    for g in range(num_games):
        # Alternate who goes first to remove first-mover advantage
        if g % 2 == 0:
            winner, shots, fa, fb = play_game(net_a, net_b, device, env_kwargs)
            wins[winner] += 1
        else:
            winner, shots, fb, fa = play_game(net_b, net_a, device, env_kwargs)
            # Flip winner
            if winner == 0:
                wins[1] += 1
            elif winner == 1:
                wins[0] += 1
            else:
                wins[None] += 1
        lengths.append(shots)
        fouls_a.append(fa)
        fouls_b.append(fb)

    # Elo difference (assume model_b is 1000 baseline)
    w_a = wins[0] + 0.5 * wins.get(None, 0)
    total = num_games
    if w_a > 0 and w_a < total:
        elo_diff = -400 * math.log10(total / w_a - 1)
    elif w_a == total:
        elo_diff = 400.0
    elif w_a == 0:
        elo_diff = -400.0
    else:
        elo_diff = 0.0

    print(f'\n=== Evaluation: {num_games} games ===')
    print(f'Model A: {model_a_path or "random"}')
    print(f'Model B: {model_b_path or "random"}')
    print(f'Wins: A={wins[0]}  B={wins[1]}  Draw={wins.get(None, 0)}')
    print(f'Win rate A: {wins[0]/num_games:.1%}')
    print(f'Elo diff (A - B): {elo_diff:+.0f}')
    print(f'Avg game length: {np.mean(lengths):.1f} shots')
    print(f'Avg fouls: A={np.mean(fouls_a):.1f}  B={np.mean(fouls_b):.1f}')

    return {
        'wins_a': wins[0], 'wins_b': wins[1], 'draws': wins.get(None, 0),
        'elo_diff': elo_diff,
        'avg_length': float(np.mean(lengths)),
        'avg_fouls_a': float(np.mean(fouls_a)),
        'avg_fouls_b': float(np.mean(fouls_b)),
    }


def main():
    p = argparse.ArgumentParser(description='8-ball head-to-head evaluation')
    p.add_argument('model_a', help='Path to model A checkpoint')
    p.add_argument('--model_b', default=None,
                   help='Path to model B checkpoint (default: random)')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    evaluate(args.model_a, args.model_b, args.games, args.device)


if __name__ == '__main__':
    main()
