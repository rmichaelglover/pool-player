"""
Render a static demo page showing Phase 3 policy behavior.

For each of N sample shots:
  - draw table + cue ball + object ball + target pocket + ghost ball
  - overlay the policy's aim line, the cue ball's straight-line path, and
    the object ball's analytic straight-line exit path after contact
  - mark the final ball position from the sim
  - tag outcome (POCKET, HIT, MISS) and aim error in degrees

Output: ../phase3_demo.html (self-contained, open in a browser)
"""
import argparse
import html
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pool_attention_net import PoolAttentionNet
from pool_sim import simulate_shot
from train_phase3 import (
    Phase3Env, ghost_ball, object_ball_exit_direction,
    POCKETS, R, TABLE_LENGTH, TABLE_WIDTH,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/phase3approach1_best.pt')
    parser.add_argument('--cut', type=float, default=60.0)
    parser.add_argument('--shots', type=int, default=16,
                        help='Number of sample shots to display.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', default='../phase3_demo.html')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    device = torch.device('cpu')
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2).to(device)
    net.log_std = torch.nn.Parameter(torch.full((2,), -0.5).to(device))
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    env = Phase3Env(max_cut_deg=args.cut)
    shots = []
    with torch.no_grad():
        for _ in range(args.shots):
            obs = env.reset()
            t = torch.FloatTensor(obs).unsqueeze(0)
            action, _, _ = net.get_action(t, deterministic=True)
            a = action[0].cpu().numpy()
            aim_angle = math.atan2(float(a[0]), float(a[1]))
            cue0 = tuple(env.cue)
            ball0 = tuple(env.ball_pos)
            target = env.target_pocket
            g = ghost_ball(ball0, target)
            ghost_angle = math.atan2(g[1] - cue0[1], g[0] - cue0[0])
            err_deg = math.degrees(abs(aim_angle - ghost_angle))
            if err_deg > 180:
                err_deg = 360 - err_deg

            # Run sim (same as Phase3Env.step but we want final positions for plotting)
            force = 60.0
            aim_dx = math.cos(aim_angle)
            aim_dy = math.sin(aim_angle)
            result = simulate_shot(
                cue0, {1: ball0},
                aim_dx * force, aim_dy * force,
                0, aim_dx, aim_dy,
            )
            cue_final = result.final_positions.get(0, cue0)
            ball_final = result.final_positions.get(1, ball0)
            hit = result.hit_ball
            pocketed = 1 in result.pocketed_ids

            # Analytic object-ball exit direction (if contact happened)
            exit_angle = object_ball_exit_direction(cue0, ball0, aim_angle)

            shots.append(dict(
                cue0=cue0, ball0=ball0, target=target, ghost=g,
                aim_angle=aim_angle, err_deg=err_deg,
                cue_final=cue_final, ball_final=ball_final,
                exit_angle=exit_angle,
                hit=hit, pocketed=pocketed,
            ))

    # ── Render to HTML/SVG ──────────────────────────────────────────────
    margin = 3.0  # inches of padding around the table
    W = TABLE_LENGTH + 2 * margin
    H = TABLE_WIDTH + 2 * margin
    scale = 6  # svg px per inch

    def x(val): return (val + margin) * scale
    def y(val): return (val + margin) * scale

    def aim_endpoint(p0, angle, length=80):
        return (p0[0] + math.cos(angle) * length,
                p0[1] + math.sin(angle) * length)

    def clip_to_table(p0, angle, max_len=200):
        """Return the point where the ray exits the table rectangle."""
        v = (math.cos(angle), math.sin(angle))
        t_candidates = []
        if v[0] > 0: t_candidates.append((TABLE_LENGTH - p0[0]) / v[0])
        elif v[0] < 0: t_candidates.append((-p0[0]) / v[0])
        if v[1] > 0: t_candidates.append((TABLE_WIDTH - p0[1]) / v[1])
        elif v[1] < 0: t_candidates.append((-p0[1]) / v[1])
        t = min(t for t in t_candidates if t > 0) if t_candidates else max_len
        t = min(t, max_len)
        return (p0[0] + v[0] * t, p0[1] + v[1] * t)

    def shot_svg(s, idx):
        parts = [f'<svg viewBox="0 0 {int(W*scale)} {int(H*scale)}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">']
        # Table felt
        parts.append(f'<rect x="{x(0)}" y="{y(0)}" width="{TABLE_LENGTH*scale}" height="{TABLE_WIDTH*scale}" fill="#0e6b3b" stroke="#3a2817" stroke-width="10"/>')
        # Pockets
        for p in POCKETS:
            radius = 2.5 if (p[0] in (0, TABLE_LENGTH)) else 2.75
            fill = '#111'
            if p == s['target']:
                parts.append(f'<circle cx="{x(p[0])}" cy="{y(p[1])}" r="{(radius+0.5)*scale}" fill="none" stroke="#ffd84a" stroke-width="3"/>')
            parts.append(f'<circle cx="{x(p[0])}" cy="{y(p[1])}" r="{radius*scale}" fill="{fill}"/>')

        # Aim line from cue (red dashed, clipped to table)
        aim_end = clip_to_table(s['cue0'], s['aim_angle'])
        parts.append(f'<line x1="{x(s["cue0"][0])}" y1="{y(s["cue0"][1])}" '
                     f'x2="{x(aim_end[0])}" y2="{y(aim_end[1])}" '
                     f'stroke="#ff4040" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.85"/>')

        # Object ball analytic exit trajectory (green dashed, clipped to table)
        if s['exit_angle'] is not None:
            exit_end = clip_to_table(s['ball0'], s['exit_angle'])
            parts.append(f'<line x1="{x(s["ball0"][0])}" y1="{y(s["ball0"][1])}" '
                         f'x2="{x(exit_end[0])}" y2="{y(exit_end[1])}" '
                         f'stroke="#40ff80" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.85"/>')

        # Ghost ball (light green outline, dashed)
        parts.append(f'<circle cx="{x(s["ghost"][0])}" cy="{y(s["ghost"][1])}" '
                     f'r="{R*scale}" fill="none" stroke="#b6ffc6" stroke-width="1.5" '
                     f'stroke-dasharray="3 2" opacity="0.7"/>')

        # Ball final position (ghosted)
        if not s['pocketed']:
            parts.append(f'<circle cx="{x(s["ball_final"][0])}" cy="{y(s["ball_final"][1])}" '
                         f'r="{R*scale}" fill="#4a9a5a" opacity="0.4" stroke="#226030" stroke-width="1"/>')

        # Cue final position (ghosted)
        parts.append(f'<circle cx="{x(s["cue_final"][0])}" cy="{y(s["cue_final"][1])}" '
                     f'r="{R*scale}" fill="#dde" opacity="0.35" stroke="#667" stroke-width="1"/>')

        # Cue ball start (white)
        parts.append(f'<circle cx="{x(s["cue0"][0])}" cy="{y(s["cue0"][1])}" '
                     f'r="{R*scale}" fill="#fdfdfd" stroke="#222" stroke-width="1.5"/>')
        # Object ball start (solid green)
        parts.append(f'<circle cx="{x(s["ball0"][0])}" cy="{y(s["ball0"][1])}" '
                     f'r="{R*scale}" fill="#44c268" stroke="#226030" stroke-width="1.5"/>')

        parts.append('</svg>')

        outcome = 'POCKET' if s['pocketed'] else ('HIT' if s['hit'] else 'MISS')
        color = '#6ce47f' if s['pocketed'] else ('#e2c44c' if s['hit'] else '#ff6666')
        return f'''
  <div class="shot">
    <div class="svg">{''.join(parts)}</div>
    <div class="meta">
      <span class="idx">#{idx+1}</span>
      <span class="outcome" style="color:{color}">{outcome}</span>
      <span class="err">aim err: {s['err_deg']:.2f}°</span>
    </div>
  </div>'''

    n_pocket = sum(1 for s in shots if s['pocketed'])
    n_hit = sum(1 for s in shots if s['hit'] and not s['pocketed'])
    n_miss = sum(1 for s in shots if not s['hit'])

    body = ''.join(shot_svg(s, i) for i, s in enumerate(shots))

    doc = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Phase 3 policy demo</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background:#101418; color:#eee; font-family:Menlo,Consolas,monospace; margin:0; padding:16px; }}
  h1 {{ margin:0 0 4px; font-size:18px; color:#ffd84a; }}
  p.sub {{ margin:0 0 12px; color:#999; font-size:12px; }}
  .legend {{ font-size:11px; color:#bbb; margin-bottom:14px; display:flex; gap:20px; flex-wrap:wrap; }}
  .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
  .sw {{ display:inline-block; width:14px; height:14px; border-radius:50%; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }}
  .shot {{ background:#1a1e24; border-radius:6px; padding:8px; }}
  .svg {{ width:100%; }}
  .meta {{ display:flex; justify-content:space-between; font-size:11px; margin-top:4px; color:#aaa; }}
  .idx {{ color:#888; }}
  .outcome {{ font-weight:bold; }}
</style>
</head>
<body>
<h1>Phase 3 policy demo — {html.escape(args.ckpt)}</h1>
<p class="sub">{args.shots} shots on Phase3Env(max_cut={args.cut}°). Policy is deterministic.
  Pocketed {n_pocket}/{args.shots} ({n_pocket/args.shots:.0%}) · Hit-no-pocket {n_hit} · Missed {n_miss}.</p>
<div class="legend">
  <span><span class="sw" style="background:#fdfdfd;border:1.5px solid #222"></span> cue ball (start)</span>
  <span><span class="sw" style="background:#44c268;border:1.5px solid #226030"></span> object ball (start)</span>
  <span><span class="sw" style="background:transparent;border:1.5px dashed #b6ffc6"></span> ghost ball</span>
  <span><span class="sw" style="background:transparent;border:2px solid #ffd84a"></span> target pocket</span>
  <span style="color:#ff4040">── aim direction</span>
  <span style="color:#40ff80">── object ball exit (straight-line)</span>
  <span><span class="sw" style="background:#4a9a5a;opacity:0.4"></span> ball final (after physics)</span>
  <span><span class="sw" style="background:#dde;opacity:0.35"></span> cue final</span>
</div>
<div class="grid">{body}</div>
</body>
</html>'''

    out_path = (HERE / args.out).resolve()
    out_path.write_text(doc)
    print(f'wrote {out_path}')
    print(f'  {args.shots} shots  pocketed={n_pocket}  hit={n_hit}  miss={n_miss}')


if __name__ == '__main__':
    main()
