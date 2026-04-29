"""Export trained aim model weights to JSON for browser."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aim import AimNetwork

net = AimNetwork(hidden=128)
net.load_state_dict(torch.load('checkpoints/best_aim.pt', map_location='cpu'))
net.eval()

model = {'layers': []}
for name, param in net.net.named_parameters():
    model['layers'].append({
        'name': name,
        'shape': list(param.shape),
        'data': param.detach().numpy().tolist()
    })

out_path = os.path.join(os.path.dirname(__file__), '..', 'js', 'aim_model.json')
with open(out_path, 'w') as f:
    json.dump(model, f)
print(f"Exported to {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
