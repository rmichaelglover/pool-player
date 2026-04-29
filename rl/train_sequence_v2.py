"""
Sequence training v2: Direct backprop instead of REINFORCE.

Instead of running episodes and getting noisy reward signals,
directly compute the difficulty of each (ball, spin) choice
and train the network to assign the highest score to the
easiest valid shot.

This is supervised learning: for each table layout, we know
the "correct" answer (the easiest shot), and we train the
network to predict it.

The key insight: shot difficulty is a deterministic function of
the geometry. We don't need RL to learn it — we can compute
the label directly and backprop.

But the SEQUENCING (which ball to shoot FIRST for best shape)
is learned through multi-step rollouts with the geometric model.
"""
import torch, torch.nn as nn, torch.optim as optim, math, time, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sequence import (
    SequenceNet, TableState, best_pocket_for_ball, estimate_cue_position,
    compute_ghost, cut_angle, is_path_clear,
    R, TL, TW, POCKETS, NUM_BALLS, NUM_POCKETS, NUM_SPIN
)


def compute_shot_difficulties(table):
    """For each (ball, spin) combo, compute the difficulty.
    Returns dict: {(ball_id, spin): difficulty} for valid shots only.
    Lower difficulty = easier shot."""
    results = {}
    active = table.get_active_balls()

    for bid, bpos in active:
        if bid > 15:
            continue
        result = best_pocket_for_ball(table.cue, bpos, bid, active)
        if result is None:
            continue
        pidx, ca, ghost, pocket_dir = result

        # Aim direction
        dx, dy = ghost[0]-table.cue[0], ghost[1]-table.cue[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d < 0.1:
            continue
        aim_dir = (dx/d, dy/d)

        # Pocket distance
        px, py = POCKETS[pidx][0].item(), POCKETS[pidx][1].item()
        pd = math.sqrt((px-bpos[0])**2 + (py-bpos[1])**2)
        total_dist = d + pd

        # Base difficulty (same formula as pocket selection)
        difficulty = ca / 75.0 + total_dist / (TL * 1.5)

        for spin in range(3):
            # Estimate cue ball position after shot
            cue_final = estimate_cue_position(ghost, aim_dir, pocket_dir, spin)
            cue_final = (max(R, min(TL-R, cue_final[0])), max(R, min(TW-R, cue_final[1])))

            # Check if cue ball position leads to good follow-up shots
            # Score the resulting position by the best available next shot
            follow_up_diff = 999
            for bid2, bpos2 in active:
                if bid2 == bid or bid2 > 15:
                    continue
                # What's the best shot from cue_final on bid2?
                remaining = [(b, p) for b, p in active if b != bid]
                result2 = best_pocket_for_ball(cue_final, bpos2, bid2, remaining)
                if result2 is not None:
                    _, ca2, ghost2, _ = result2
                    d2 = math.sqrt((ghost2[0]-cue_final[0])**2 + (ghost2[1]-cue_final[1])**2)
                    pd2 = math.sqrt((POCKETS[result2[0]][0].item()-bpos2[0])**2 +
                                    (POCKETS[result2[0]][1].item()-bpos2[1])**2)
                    diff2 = ca2 / 75.0 + (d2 + pd2) / (TL * 1.5)
                    follow_up_diff = min(follow_up_diff, diff2)

            # Combined: current shot difficulty + follow-up difficulty
            # Weight follow-up less (bird in hand > bird in bush)
            combined = difficulty + follow_up_diff * 0.3
            results[(bid, spin)] = combined

    return results


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SequenceNet(512).to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50000)
    print(f'{sum(p.numel() for p in net.parameters()):,} params, {device}')

    num_balls = 3
    best_loss = 999
    os.makedirs('checkpoints', exist_ok=True)
    t0 = time.time()

    for iteration in range(50000):
        # Curriculum
        if iteration > 5000: num_balls = 5
        if iteration > 15000: num_balls = 8
        if iteration > 30000: num_balls = 12

        batch_loss = []

        # Generate a batch of random layouts
        for _ in range(64):
            table = TableState(num_balls)
            table.reset()

            obs = torch.tensor(table.get_obs(), device=device).unsqueeze(0)
            scores = net(obs).squeeze(0)  # (15, 3)

            # Compute difficulty for each valid (ball, spin) choice
            difficulties = compute_shot_difficulties(table)
            if len(difficulties) == 0:
                continue

            # Find the best (lowest difficulty) choice
            best_action = min(difficulties, key=difficulties.get)
            best_diff = difficulties[best_action]

            # Build target: the network should give the highest score
            # to the easiest shot, lowest to the hardest.
            # Use cross-entropy: softmax over valid actions, target = easiest
            valid_scores = []
            valid_diffs = []
            valid_indices = []
            for (bid, spin), diff in difficulties.items():
                valid_scores.append(scores[bid-1, spin])
                valid_diffs.append(diff)
                valid_indices.append((bid, spin))

            if len(valid_scores) < 2:
                continue

            valid_scores_t = torch.stack(valid_scores)
            valid_diffs_t = torch.tensor(valid_diffs, device=device)

            # Target distribution: softmax of NEGATIVE difficulty
            # (lower difficulty = higher probability)
            target_logits = -valid_diffs_t * 3.0  # scale factor for sharpness
            target_probs = torch.softmax(target_logits, dim=0)

            # Loss: KL divergence between network's softmax and target
            pred_log_probs = torch.log_softmax(valid_scores_t, dim=0)
            loss = -(target_probs * pred_log_probs).sum()
            batch_loss.append(loss)

        if len(batch_loss) == 0:
            continue

        total_loss = torch.stack(batch_loss).mean()
        opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if (iteration + 1) % 200 == 0:
            # Evaluate: on 100 random layouts, does the network pick the easiest shot?
            correct = 0
            top3 = 0
            total = 0
            for _ in range(100):
                table = TableState(num_balls)
                table.reset()
                diffs = compute_shot_difficulties(table)
                if len(diffs) < 2:
                    continue
                total += 1

                obs = torch.tensor(table.get_obs(), device=device).unsqueeze(0)
                with torch.no_grad():
                    sc = net(obs).squeeze(0)

                # Network's pick
                best_net_score = -1e9
                net_pick = None
                for (bid, spin), diff in diffs.items():
                    s = sc[bid-1, spin].item()
                    if s > best_net_score:
                        best_net_score = s
                        net_pick = (bid, spin)

                # Correct pick (lowest difficulty)
                sorted_actions = sorted(diffs.items(), key=lambda x: x[1])
                correct_pick = sorted_actions[0][0]
                top3_picks = [a[0] for a in sorted_actions[:3]]

                if net_pick == correct_pick:
                    correct += 1
                if net_pick in top3_picks:
                    top3 += 1

            elapsed = time.time() - t0
            acc = correct/max(total,1)*100
            t3 = top3/max(total,1)*100
            print(f'Iter {iteration+1:6d} | Balls:{num_balls} | '
                  f'Loss={total_loss.item():.4f} | '
                  f'Picks easiest: {acc:.0f}% | Top-3: {t3:.0f}% | {elapsed:.0f}s')
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save(net.state_dict(), 'checkpoints/best_sequence.pt')
                print(f'  -> Best')

    # Export
    net.cpu().eval()
    model = {'layers':[],'activation':'relu','output':'sequence'}
    for n,p in net.net.named_parameters():
        model['layers'].append({'name':n,'shape':list(p.shape),'data':p.detach().numpy().tolist()})
    with open('../js/sequence_model.json','w') as f: json.dump(model,f)
    print(f'Exported. Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    train()
