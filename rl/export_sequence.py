"""Export sequence model for browser demo."""
import torch, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sequence import SequenceNet

net = SequenceNet(512)
net.load_state_dict(torch.load('checkpoints/best_sequence.pt', map_location='cpu'))
net.eval()
model = {'layers': [], 'activation': 'relu', 'output': 'sequence'}
for name, param in net.net.named_parameters():
    model['layers'].append({'name': name, 'shape': list(param.shape), 'data': param.detach().numpy().tolist()})
with open('../js/sequence_model.json', 'w') as f:
    json.dump(model, f)
print(f'Exported ({os.path.getsize("../js/sequence_model.json")/1024:.0f} KB)')
