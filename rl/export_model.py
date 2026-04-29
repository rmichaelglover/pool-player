"""
Export trained PyTorch policy to JSON for browser inference.
Usage: python3 export_model.py checkpoints/best_policy.pt
"""
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from policy_network import ActorCritic
from config import Config


def export_to_json(model_path, output_path=None):
    cfg = Config()
    policy = ActorCritic(obs_dim=cfg.obs_dim, act_dim=cfg.act_dim, hidden_dims=cfg.hidden_dims)
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()

    model_data = {
        'obs_dim': cfg.obs_dim,
        'act_dim': cfg.act_dim,
        'hidden_dims': list(cfg.hidden_dims),
        'obs_mean': policy.obs_mean.numpy().tolist(),
        'obs_var': policy.obs_var.numpy().tolist(),
        'actor': [],
        'log_std': policy.log_std.detach().numpy().tolist(),
    }

    # Extract actor weights
    for name, param in policy.actor.named_parameters():
        model_data['actor'].append({
            'name': name,
            'shape': list(param.shape),
            'data': param.detach().numpy().tolist(),
        })

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), '..', 'js', 'rl_policy.json')

    with open(output_path, 'w') as f:
        json.dump(model_data, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported to {output_path} ({size_kb:.0f} KB)")
    print(f"Actor layers: {len(model_data['actor'])}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 export_model.py <model_path> [output_path]")
        sys.exit(1)
    export_to_json(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
