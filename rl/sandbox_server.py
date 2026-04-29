"""
Tiny stdlib HTTP server for the one-ball interactive sandbox.

Serves:
  GET  /                 -> sandbox.html (from the parent project dir)
  GET  /<any static>     -> same, relative to parent dir
  POST /shoot            -> JSON {cue, ball, force, ex, ey, aim_mode,
                                  manual_aim_deg, ckpt}
                            returns JSON {trajectory, hit, pocketed,
                                          aim_deg, cue_final, ball_final}

The Phase 3 network is lazy-loaded the first time 'network' aim is
requested and cached. Any checkpoint under rl/checkpoints/ is allowed.

Usage:
    python sandbox_server.py [--port 8000]
    open http://localhost:8000/ in a browser
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from pool_sim import simulate_shot
from pool_attention_net import PoolAttentionNet
from train_phase3 import TABLE_LENGTH, TABLE_WIDTH, R

_nets: dict[str, PoolAttentionNet] = {}


def load_net(ckpt_relpath: str) -> PoolAttentionNet:
    if ckpt_relpath in _nets:
        return _nets[ckpt_relpath]
    path = (HERE / ckpt_relpath).resolve()
    if not str(path).startswith(str(HERE)):
        raise ValueError('ckpt must be inside rl/')
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2)
    net.log_std = torch.nn.Parameter(torch.full((2,), -0.5))
    net.load_state_dict(torch.load(str(path), map_location='cpu', weights_only=True))
    net.eval()
    _nets[ckpt_relpath] = net
    print(f'[load] {ckpt_relpath}  {sum(p.numel() for p in net.parameters()):,} params')
    return net


def build_observation(cue, ball):
    """38-dim obs that matches Phase3Env.get_obs()."""
    obs = np.full(38, -1.0, dtype=np.float32)
    obs[0] = cue[0] / TABLE_LENGTH
    obs[1] = cue[1] / TABLE_WIDTH
    obs[2] = ball[0] / TABLE_LENGTH
    obs[3] = ball[1] / TABLE_WIDTH
    obs[32] = 0.0
    obs[33] = 0.0
    obs[34] = 1.0 / 15.0
    obs[35] = 0.0
    obs[36] = 0.0
    obs[37] = 0.0
    return obs


def network_aim(ckpt_relpath: str, cue, ball) -> float:
    net = load_net(ckpt_relpath)
    obs = build_observation(cue, ball)
    t = torch.from_numpy(obs).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = net.get_action(t, deterministic=True)
    a = action[0].cpu().numpy()
    return math.atan2(float(a[0]), float(a[1]))


def clip_pos(xy, min_m=R*1.1, max_margin_x=TABLE_LENGTH, max_margin_y=TABLE_WIDTH):
    x = max(min_m, min(max_margin_x - min_m, float(xy[0])))
    y = max(min_m, min(max_margin_y - min_m, float(xy[1])))
    return [x, y]


def shoot(payload: dict) -> dict:
    cue = clip_pos(payload['cue'])
    ball = clip_pos(payload['ball'])
    force = float(payload.get('force', 40.0))
    ex = float(payload.get('ex', 0.0))
    ey = float(payload.get('ey', 0.0))
    aim_mode = payload.get('aim_mode', 'manual')
    ckpt = payload.get('ckpt', 'checkpoints/phase3_slowlr_92pct.pt')

    if aim_mode == 'network':
        aim_rad = network_aim(ckpt, cue, ball)
    else:
        aim_rad = math.radians(float(payload.get('manual_aim_deg', 0.0)))

    # Continuous spin from tip-offset: spin_factor = 2.5 * ey (derived from
    # tip-contact physics — ey is tip offset as fraction of ball radius).
    # contact_x (english) is not modeled yet.
    spin_factor = 2.5 * ey

    aim_dx = math.cos(aim_rad)
    aim_dy = math.sin(aim_rad)

    result = simulate_shot(
        tuple(cue), {1: tuple(ball)},
        aim_dx * force, aim_dy * force,
        spin_factor, aim_dx, aim_dy,
        record_trajectory=True,
        traj_max_frames=600,
    )

    traj = result.trajectory.tolist() if result.trajectory is not None else []
    return {
        'trajectory': traj,
        'aim_deg': math.degrees(aim_rad),
        'hit': bool(result.hit_ball),
        'pocketed': 1 in result.pocketed_ids,
        'cue_scratched': bool(result.cue_scratched),
        'cue_final': list(result.final_positions[0]),
        'ball_final': list(result.final_positions[1]),
        'spin_factor': spin_factor,
        'english_applied': False,  # honest flag: sim ignores contact_x
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quieter

    def _send_json(self, code, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/' or path == '/index.html':
            self._send_file(PROJECT_ROOT / 'sandbox.html', 'text/html; charset=utf-8')
            return
        if path == '/list_checkpoints':
            ckpts = sorted(p.name for p in (HERE / 'checkpoints').glob('*.pt'))
            self._send_json(200, {'checkpoints': ckpts})
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/shoot':
            self.send_error(404)
            return
        length = int(self.headers.get('Content-Length', '0'))
        try:
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            result = shoot(payload)
            self._send_json(200, result)
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {'error': str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--default-ckpt',
                        default='checkpoints/phase3_slowlr_92pct.pt')
    args = parser.parse_args()

    # Warm up the default checkpoint so first shot is fast
    try:
        load_net(args.default_ckpt)
    except Exception as e:
        print(f'[warn] failed to preload {args.default_ckpt}: {e}')

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Sandbox server on http://{args.host}:{args.port}/')
    print(f'Default checkpoint: {args.default_ckpt}')
    print('Ctrl-C to stop')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopping')


if __name__ == '__main__':
    main()
