"""Export RunNet weights to JSON for browser inference."""
import torch, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_runs import RunNet

def export(ckpt_path, out_path):
    net = RunNet()
    net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    net.eval()
    weights = {
        'trunk_0_weight': net.trunk[0].weight.data.tolist(),
        'trunk_0_bias': net.trunk[0].bias.data.tolist(),
        'trunk_2_weight': net.trunk[2].weight.data.tolist(),
        'trunk_2_bias': net.trunk[2].bias.data.tolist(),
        'shot_weight': net.shot_head.weight.data.tolist(),
        'shot_bias': net.shot_head.bias.data.tolist(),
        'spin_weight': net.spin_head.weight.data.tolist(),
        'spin_bias': net.spin_head.bias.data.tolist(),
        'speed_weight': net.speed_head.weight.data.tolist(),
        'speed_bias': net.speed_head.bias.data.tolist(),
        'input_dim': 40,  # metadata for the demo
    }
    with open(out_path, 'w') as f:
        json.dump(weights, f)
    print(f'Exported {ckpt_path} -> {out_path}')
    print(f'  trunk[0]: {net.trunk[0].weight.shape}')
    print(f'  trunk[2]: {net.trunk[2].weight.shape}')
    print(f'  shot:     {net.shot_head.weight.shape}')
    print(f'  spin:     {net.spin_head.weight.shape}')
    print(f'  speed:    {net.speed_head.weight.shape}')

if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best_run_net.pt'
    out = sys.argv[2] if len(sys.argv) > 2 else '../strategy_weights.json'
    export(ckpt, out)
