"""
State-diversity probe for the trained strategy network.

Questions we want to answer:
  1. Mode-collapsed? (Does the policy always pick the same slot regardless of state?)
  2. State-conditional? (Do different table states lead to different choices?)
  3. Are all 4 actions ever used, or does the policy ignore some slots?
  4. Does behavior change with game phase (many balls vs few balls)?
"""
import torch, random, math, sys, os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_selfplay import (
    StrategyNet, Table141, build_candidates, N_ACTIONS, NUM_CANDIDATES
)

SLOT_NAMES = ['safe', 'breaker', 'position', 'SAFETY']


def sample_game_states(n_states=1000, device='cpu'):
    """Play self-games with a random policy; collect diverse states."""
    states = []
    while len(states) < n_states:
        table = Table141()
        for _ in range(80):
            detailed, obs = build_candidates(table)
            if any(d is not None for d in detailed):
                states.append((obs.copy(), [d is not None for d in detailed],
                              sum(1 for b in table.balls if b not in table.pocketed)))
            # Random move
            active = table.get_active_balls()
            if not active:
                break
            bid, _ = random.choice(active)
            table.execute_shot(bid, 0, 1.0)
            # Occasionally switch cue to mix states
            if random.random() < 0.1:
                from train_sequence import R, TL, TW
                table.cue = [R * 2 + random.random() * (TL - 4 * R),
                             R * 2 + random.random() * (TW - 4 * R)]
    return states[:n_states]


def probe():
    device = torch.device('cpu')
    net = StrategyNet().to(device)
    ckpt = 'checkpoints/best_strategy_ppo.pt'
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    print(f'Loaded {ckpt}')

    # --- Sanity: what does the untrained network output on the same states? ---
    net_random = StrategyNet().to(device)  # zero-init → uniform

    print('Sampling 1000 states...')
    states = sample_game_states(1000, device=device)
    print(f'Got {len(states)} states')

    # --- Analyze trained policy ---
    trained_picks = Counter()
    trained_entropies = []
    random_picks = Counter()
    per_phase_picks = {'early(>10)': Counter(), 'mid(5-10)': Counter(), 'late(<5)': Counter()}

    with torch.no_grad():
        for obs, exists, nballs in states:
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            logits = net(obs_t)[0].squeeze(0)  # shot_logits from (shot, speed, value)
            mask = torch.tensor([float(exists[i]) for i in range(NUM_CANDIDATES)] + [1.0])
            stable = logits - logits.max()
            probs = torch.exp(stable) * mask
            probs = probs / probs.sum()
            pick = torch.argmax(probs).item()
            ent = -(probs * (probs.clamp(min=1e-12)).log()).sum().item()
            trained_picks[SLOT_NAMES[pick]] += 1
            trained_entropies.append(ent)
            if nballs > 10:
                per_phase_picks['early(>10)'][SLOT_NAMES[pick]] += 1
            elif nballs >= 5:
                per_phase_picks['mid(5-10)'][SLOT_NAMES[pick]] += 1
            else:
                per_phase_picks['late(<5)'][SLOT_NAMES[pick]] += 1

            rlogits = net_random(obs_t)[0].squeeze(0)
            rprobs = torch.softmax(rlogits, dim=-1) * mask
            rprobs = rprobs / rprobs.sum()
            rpick = torch.argmax(rprobs).item()
            random_picks[SLOT_NAMES[rpick]] += 1

    print()
    print('=== Trained policy action distribution (argmax) ===')
    total = sum(trained_picks.values())
    for name in SLOT_NAMES:
        c = trained_picks[name]
        print(f'  {name:10s}: {c:4d} ({100*c/total:5.1f}%)')

    print()
    print('=== Random (zero-init) policy for comparison ===')
    for name in SLOT_NAMES:
        c = random_picks[name]
        print(f'  {name:10s}: {c:4d} ({100*c/total:5.1f}%)')

    print()
    print('=== Per-state entropy (trained) ===')
    arr = np.array(trained_entropies)
    print(f'  mean={arr.mean():.3f}, median={np.median(arr):.3f}, '
          f'min={arr.min():.3f}, max={arr.max():.3f}')
    print(f'  frac states with entropy > 0.5 (truly mixed): {(arr > 0.5).mean()*100:.1f}%')
    print(f'  frac states with entropy < 0.1 (confident):    {(arr < 0.1).mean()*100:.1f}%')

    print()
    print('=== Action distribution by game phase ===')
    for phase, c in per_phase_picks.items():
        tot = sum(c.values()) or 1
        parts = ' '.join(f'{name}={c[name]}({100*c[name]/tot:.0f}%)' for name in SLOT_NAMES)
        print(f'  {phase:12s} (n={tot}): {parts}')


if __name__ == '__main__':
    probe()
