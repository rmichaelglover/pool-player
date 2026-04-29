#!/bin/bash
set -e
echo "=== Pool AI Training Setup ==="

# Compile C physics sim
echo "Compiling C physics engine..."
gcc -O2 -shared -fPIC -o libpool_sim.so pool_sim.c -lm
echo "  -> libpool_sim.so built (0.056ms/shot)"

# Check PyTorch + CUDA
echo "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Quick sanity test
echo "Running sanity test..."
python3 -c "
from pool_attention_net import PoolAttentionNet
import torch
net = PoolAttentionNet()
print(f'Network: {sum(p.numel() for p in net.parameters()):,} params')
obs = torch.randn(4, 38)
a, v = net(obs)
print(f'Forward pass OK: action={a.shape}, value={v.shape}')
if torch.cuda.is_available():
    net = net.cuda()
    obs = obs.cuda()
    a, v = net(obs)
    print(f'CUDA forward pass OK')
"

echo ""
echo "=== Ready! Launch training with: ==="
echo "  python3 train_attention.py --envs 1024 --device cuda --iters 10000"
echo ""
echo "For quick test first:"
echo "  python3 train_attention.py --envs 64 --device cuda --iters 100"
