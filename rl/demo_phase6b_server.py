"""
Demo server for watching the Phase 6b AI play 14.1 continuous with search.

Serves:
  GET  /                  -> demo_phase6b.html
  POST /start             -> initialize new episode, returns initial state
  POST /next_shot         -> run one shot with search, returns trajectory + state

Episode state is stored server-side keyed by session_id (the client generates one).

Usage:
    python demo_phase6b_server.py --ckpt checkpoints/phase6_p6_500_best.pt [--port 8001]
    open http://localhost:8001/ in a browser
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pool_attention_net import PoolAttentionNet
from pool_sim import simulate_shot
from train_phase4 import decode_action, ACT_DIM
from train_phase6b import (Phase6bEnv, TABLE_LENGTH, TABLE_WIDTH, R,
                            RACK_APEX, RACK_POSITIONS)
from shot_search_phase6b import shot_search_phase6b
from heuristic_policy import heuristic_action, HeuristicAction
from pool_game_net import PoolGameNet
from train_phase7 import Phase7Env
from train_phase8 import Phase8Env
from shot_enumerator import POCKETS as P7_POCKETS, POCKET_NAMES
from shot_search_phase7 import shot_search_phase7

# ── Global state ───────────────────────────────────────────────────────────
_net = None
_p7_net = None
_device = None
_ckpt_path = None
_policy_mode = 'heuristic'  # 'heuristic' | 'network' | 'phase7'
_p7_env_mode = 'standard'   # 'standard' = Phase7Env (auto-break), 'breakdrill' = Phase8Env
_p7_noise = {'aim_noise_deg': 0.0, 'force_noise_pct': 0.0, 'spin_noise': 0.0}
_p7_search_mc = 1
_sessions: dict = {}


def load_net(ckpt_path: str, embed_dim=96, num_heads=6, num_layers=4,
             ff_dim=192, device='cpu'):
    global _net, _device, _ckpt_path
    _device = torch.device(device)
    _net = PoolAttentionNet(embed_dim=embed_dim, num_heads=num_heads,
                            num_layers=num_layers, ff_dim=ff_dim,
                            act_dim=ACT_DIM).to(_device)
    _net.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5).to(_device))
    _net.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
    _net.eval()
    _ckpt_path = ckpt_path
    print(f'Loaded net: {ckpt_path} on {_device}')


def serialize_balls(env: Phase6bEnv):
    return [
        {'id': bid, 'x': pos[0], 'y': pos[1]}
        for bid, pos in sorted(env.balls.items())
    ]


def run_trajectory_shot(env: Phase6bEnv, aim_angle, force, spin_factor,
                         max_traj_frames=600):
    """Run one shot via env.step (so call-shot rule is applied) and return a
    client-friendly bundle including trajectory frames."""
    rerack_before = env.rerack_count
    _, reward, done, info = env.step(
        aim_angle, force, spin_factor,
        record_trajectory=True, traj_max_frames=max_traj_frames,
    )
    rerack_happened = env.rerack_count > rerack_before

    return {
        'trajectory': info.get('trajectory', []),
        'ball_ids_order': info.get('trajectory_ball_ids', []),
        'pocketed_ids': info.get('pocketed_ids', []),
        'scratch': info.get('scratch', False),
        'hit_ball': info.get('hit_ball', False),
        'called_id': info.get('called_id'),
        'called_pocket': info.get('called_pocket', -1),
        'called_actual_pocket': info.get('called_actual_pocket', -1),
        'called_shot_valid': info.get('called_shot_valid', False),
        'reward': reward,
        'rerack_happened': rerack_happened,
        'cue_after': list(env.cue),
        'balls_after': serialize_balls(env),
        'shot_idx': env.shot_idx,
        'rerack_count': env.rerack_count,
        'total_pocketed': env.total_pocketed,
        'done': env.done,
    }


def handle_start(payload):
    session_id = payload.get('session_id', 'default')
    max_shots = int(payload.get('max_shots', 50))
    if _policy_mode == 'phase7':
        if _p7_env_mode == 'breakdrill':
            env = Phase8Env(max_shots=max_shots, **_p7_noise)
        else:
            env = Phase7Env(max_shots=max_shots, **_p7_noise)
    else:
        env = Phase6bEnv(max_shots=max_shots, lenient_break=True)
    _sessions[session_id] = env
    return {
        'session_id': session_id,
        'cue': list(env.cue),
        'balls': serialize_balls(env),
        'table_length': TABLE_LENGTH,
        'table_width': TABLE_WIDTH,
        'ball_radius': R,
        'rack_apex': list(RACK_APEX),
        'shot_idx': getattr(env, 'shot_idx', 0),
        'total_pocketed': getattr(env, 'total_pocketed', 0),
        'rerack_count': getattr(env, 'rerack_count', 0),
        'done': env.done,
        'policy_mode': _policy_mode,
    }


def handle_next_shot(payload):
    session_id = payload.get('session_id', 'default')
    K1 = int(payload.get('K1', 24))
    K2 = int(payload.get('K2', 16))
    if session_id not in _sessions:
        return {'error': 'no such session'}
    env = _sessions[session_id]
    if env.done:
        return {'error': 'episode already done'}

    t0 = time.time()

    if _policy_mode == 'phase7':
        return _handle_phase7_shot(env, t0)

    extra = {}
    if _policy_mode == 'heuristic':
        ha = heuristic_action(env)
        aim, force, spin = ha.aim_angle, ha.force, ha.spin
        extra['reason'] = ha.reason
    else:
        raw_action = shot_search_phase6b(_net, env, K_per_depth=(K1, K2), device=_device)
        aim, force, spin = decode_action(raw_action)
    decision_ms = (time.time() - t0) * 1000

    out = run_trajectory_shot(env, aim, force, spin)
    out['aim_deg'] = math.degrees(aim)
    out['force'] = force
    out['spin_factor'] = spin
    out['search_ms'] = decision_ms
    out['policy_mode'] = _policy_mode
    out.update(extra)
    return out


def _handle_phase7_shot(env: Phase7Env, t0):
    """Run one shot using the Phase 7 network in deterministic mode."""
    obs = env.get_obs()
    batch = {
        'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(_device),
        'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(_device),
        'ball_is_cue': torch.from_numpy(obs.ball_is_cue).unsqueeze(0).to(_device),
        'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(_device),
        'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(_device),
        'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(_device),
    }
    # If there are no legal shots, end.
    if not obs.shot_meta:
        env.done = True
        return {
            'trajectory': [], 'ball_ids_order': [],
            'pocketed_ids': [], 'scratch': False, 'hit_ball': False,
            'called_id': None, 'called_pocket': -1, 'called_actual_pocket': -1,
            'called_shot_valid': False,
            'aim_deg': 0.0, 'force': 0.0, 'spin_factor': 0.0,
            'reward': 0.0, 'rerack_happened': False,
            'cue_after': list(env.cue), 'balls_after': serialize_balls(env),
            'shot_idx': env.shot_idx, 'rerack_count': env.rerack_count,
            'total_pocketed': env.total_pocketed, 'done': True,
            'search_ms': (time.time() - t0) * 1000,
            'policy_mode': 'phase7',
            'reason': 'no legal shots',
        }

    # Depth-1 shot search at inference (K=8, M=8). With execution noise
    # active, MC averaging over `_p7_search_mc` rollouts per candidate makes
    # search prefer shots that succeed *robustly* under noise rather than
    # deterministic-frontier shots. Without noise the search code collapses
    # MC back to 1 sample automatically.
    action = shot_search_phase7(_p7_net, env, obs, K_shots=8, M_per_shot=8,
                                 device=_device,
                                 noise_samples=_p7_search_mc)
    if action is None:
        env.done = True
        return {
            'trajectory': [], 'ball_ids_order': [],
            'pocketed_ids': [], 'scratch': False, 'hit_ball': False,
            'called_id': None, 'called_pocket': -1, 'called_actual_pocket': -1,
            'called_shot_valid': False,
            'aim_deg': 0.0, 'force': 0.0, 'spin_factor': 0.0,
            'reward': 0.0, 'rerack_happened': False,
            'cue_after': list(env.cue), 'balls_after': serialize_balls(env),
            'shot_idx': env.shot_idx, 'rerack_count': env.rerack_count,
            'total_pocketed': env.total_pocketed, 'done': True,
            'search_ms': (time.time() - t0) * 1000,
            'policy_mode': 'phase7',
            'reason': 'search returned no action (no legal shots)',
        }
    chosen_idx, f_raw, s_raw = action
    chosen_shot = obs.shot_meta[chosen_idx]

    rerack_before = env.rerack_count
    _, reward, done, info = env.step(
        chosen_idx, f_raw, s_raw, obs,
        record_trajectory=True, traj_max_frames=600,
    )
    rerack_happened = env.rerack_count > rerack_before
    decision_ms = (time.time() - t0) * 1000

    aim = info.get('aim_angle', chosen_shot.aim_angle)
    force = info.get('force', 0.0)
    spin = info.get('spin', 0.0)

    return {
        'trajectory': info.get('trajectory', []),
        'ball_ids_order': info.get('trajectory_ball_ids', []),
        'pocketed_ids': info.get('pocketed_ids', []),
        'scratch': info.get('scratch', False),
        'hit_ball': True,
        'called_id': chosen_shot.ball_id,
        'called_pocket': chosen_shot.pocket_idx,
        'called_actual_pocket': chosen_shot.pocket_idx if info.get('called_ok') else -1,
        'called_shot_valid': info.get('called_ok', False),
        'reward': reward,
        'rerack_happened': rerack_happened,
        'cue_after': list(env.cue),
        'balls_after': serialize_balls(env),
        'shot_idx': env.shot_idx,
        'rerack_count': env.rerack_count,
        'total_pocketed': env.total_pocketed,
        'done': env.done,
        'aim_deg': math.degrees(aim),
        'force': force,
        'spin_factor': spin,
        'search_ms': decision_ms,
        'policy_mode': 'phase7',
        'reason': f'ball {chosen_shot.ball_id} → {POCKET_NAMES[chosen_shot.pocket_idx]} '
                  f'(cut {chosen_shot.cut_angle_deg:.1f}°)',
    }


# ── HTTP plumbing ──────────────────────────────────────────────────────────

HTML_PATH = HERE.parent / 'demo_phase6b.html'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, code, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            if HTML_PATH.exists():
                body = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()
                self.wfile.write(b'demo_phase6b.html not found')
        elif path == '/geometry':
            from table_geometry import to_dict
            self._send_json(200, to_dict())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else '{}'
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            self._send_json(400, {'error': 'bad json'}); return
        try:
            if path == '/start':
                self._send_json(200, handle_start(payload))
            elif path == '/next_shot':
                self._send_json(200, handle_next_shot(payload))
            else:
                self._send_json(404, {'error': 'unknown endpoint'})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {'error': str(e)})


def main():
    global _policy_mode, _p7_env_mode, _p7_net, _device, _p7_noise, _p7_search_mc
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints/phase6_p6_500_best.pt')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--device', default='cpu')
    p.add_argument('--embed_dim', type=int, default=96)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--ff_dim', type=int, default=192)
    p.add_argument('--policy', default='heuristic',
                   choices=['heuristic', 'network', 'phase7'])
    p.add_argument('--p7_ckpt', default='checkpoints/phase7_p7_first_best.pt',
                   help='Phase 7 network checkpoint')
    p.add_argument('--p7_embed_dim', type=int, default=128)
    p.add_argument('--p7_num_heads', type=int, default=8)
    p.add_argument('--p7_num_layers', type=int, default=4)
    p.add_argument('--p7_env', default='standard',
                   choices=['standard', 'breakdrill'],
                   help='standard = full 15-ball rack with auto-break; '
                        'breakdrill = 14-ball rack + break ball + cue (agent picks break)')
    p.add_argument('--p7_aim_noise_deg', type=float, default=0.0,
                   help='Per-shot Gaussian aim noise σ (degrees). 0.10 ≈ good amateur.')
    p.add_argument('--p7_force_noise_pct', type=float, default=0.0,
                   help='Per-shot Gaussian force noise (fraction of force).')
    p.add_argument('--p7_spin_noise', type=float, default=0.0,
                   help='Per-shot Gaussian spin noise σ (units of spin_factor).')
    p.add_argument('--p7_search_mc', type=int, default=8,
                   help='MC samples per candidate in shot search. Effective '
                        'only when noise > 0. 8 ≈ +60-77% mean balls vs raw.')
    args = p.parse_args()
    _policy_mode = args.policy
    _p7_env_mode = args.p7_env
    _p7_noise = {
        'aim_noise_deg': args.p7_aim_noise_deg,
        'force_noise_pct': args.p7_force_noise_pct,
        'spin_noise': args.p7_spin_noise,
    }
    _p7_search_mc = args.p7_search_mc
    _device = torch.device(args.device)
    if _policy_mode == 'network':
        load_net(args.ckpt, embed_dim=args.embed_dim, num_heads=args.num_heads,
                 num_layers=args.num_layers, ff_dim=args.ff_dim, device=args.device)
    elif _policy_mode == 'phase7':
        _p7_net = PoolGameNet(embed_dim=args.p7_embed_dim,
                              num_heads=args.p7_num_heads,
                              num_layers=args.p7_num_layers).to(_device)
        _p7_net.load_state_dict(torch.load(args.p7_ckpt, map_location=_device,
                                            weights_only=True))
        _p7_net.eval()
        print(f'Loaded Phase 7 net: {args.p7_ckpt}')
        if any(v > 0 for v in _p7_noise.values()):
            print(f'  noise: aim={_p7_noise["aim_noise_deg"]}° '
                  f'force={_p7_noise["force_noise_pct"]*100:.1f}% '
                  f'spin={_p7_noise["spin_noise"]} | MC={_p7_search_mc}')
        else:
            print(f'  deterministic env (MC search auto-collapses to 1)')
    print(f'policy mode: {_policy_mode}')
    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'demo serving at http://localhost:{args.port}/')
    srv.serve_forever()


if __name__ == '__main__':
    main()
