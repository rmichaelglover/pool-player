"""
Demo server for watching two 8-ball AIs play each other.

Serves:
  GET  /                  -> demo_eight_ball.html
  GET  /geometry          -> table geometry (pockets, cushions)
  POST /start             -> initialize new game, auto-break, return state
  POST /next_shot         -> run one AI shot, return trajectory + state
  POST /legal_shots       -> legal shots with network probabilities

Usage:
    python demo_eight_ball_server.py \
        --ckpt checkpoints/eight_ball_8ball_v1_best.pt \
        --port 8002
"""
from __future__ import annotations
import argparse
import copy
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'shared'))

from eight_ball_env import EightBallEnv, GAME_OVER, OPEN_TABLE, PLAYING
from eight_ball_net import (EightBallNet, EightBallObs, MAX_SHOTS,
                            TABLE_LENGTH, TABLE_WIDTH,
                            decode_force, decode_spin)
from shot_enumerator import POCKETS, POCKET_NAMES, R

_net = None
_device = None
_sessions: dict = {}
_histories: dict = {}
HISTORY_MAX = 50


def load_net(ckpt_path, embed_dim=128, num_heads=8, num_layers=4,
             device='cpu'):
    global _net, _device
    _device = torch.device(device)
    _net = EightBallNet(embed_dim=embed_dim, num_heads=num_heads,
                        num_layers=num_layers).to(_device)
    state = torch.load(ckpt_path, map_location=_device, weights_only=True)
    key = 'shot_encoder.0.weight'
    if key in state and state[key].shape[1] < _net.shot_encoder[0].in_features:
        pad_cols = _net.shot_encoder[0].in_features - state[key].shape[1]
        state[key] = torch.cat([
            state[key],
            torch.zeros(state[key].shape[0], pad_cols, device=state[key].device),
        ], dim=1)
    _net.load_state_dict(state, strict=False)
    _net.eval()
    print(f'Loaded 8-ball net: {ckpt_path} ({sum(p.numel() for p in _net.parameters()):,} params)')


def serialize_balls(env):
    return [{'id': bid, 'x': pos[0], 'y': pos[1]}
            for bid, pos in sorted(env.balls.items())]


def game_state_dict(env):
    phase_names = {0: 'break', 1: 'open_table', 2: 'playing', 3: 'game_over'}
    return {
        'current_player': env.current_player,
        'groups': {str(k): v for k, v in env.groups.items()},
        'remaining_0': env._my_remaining(0),
        'remaining_1': env._my_remaining(1),
        'on_8ball_0': env._on_8ball(0),
        'on_8ball_1': env._on_8ball(1),
        'phase': phase_names.get(env.phase, 'unknown'),
        'total_shots': env.total_shots,
        'winner': env.winner,
        'ball_in_hand': env.ball_in_hand,
    }


def obs_to_batch(obs, device):
    return {
        'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(device),
        'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(device),
        'ball_group': torch.from_numpy(obs.ball_group).unsqueeze(0).to(device),
        'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(device),
        'game_state': torch.from_numpy(obs.game_state).unsqueeze(0).to(device),
        'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(device),
        'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(device),
    }


def handle_start(payload):
    session_id = payload.get('session_id', 'default')
    env = EightBallEnv(max_shots_per_game=200)
    _sessions[session_id] = env
    _histories[session_id] = []
    return {
        'session_id': session_id,
        'cue': list(env.cue),
        'balls': serialize_balls(env),
        'table_length': TABLE_LENGTH,
        'table_width': TABLE_WIDTH,
        'ball_radius': R,
        **game_state_dict(env),
    }


def handle_next_shot(payload):
    session_id = payload.get('session_id', 'default')
    if session_id not in _sessions:
        return {'error': 'no such session'}
    env = _sessions[session_id]
    if env.phase == GAME_OVER:
        return {'error': 'game over', 'done': True, **game_state_dict(env)}

    stack = _histories.setdefault(session_id, [])
    stack.append(copy.deepcopy(env))
    if len(stack) > HISTORY_MAX:
        stack.pop(0)

    t0 = time.time()
    player = env.current_player
    obs = env.get_obs()

    # Handle learned ball-in-hand placement
    if env.awaiting_placement:
        batch = obs_to_batch(obs, _device)
        with torch.no_grad():
            _, xn, yn, _, value = _net.get_action(batch, deterministic=True)
        obs_next, reward, done, info = env.step_placement(xn.item(), yn.item())
        return {
            'trajectory': [], 'ball_ids_order': [],
            'pocketed_ids': [], 'scratch': False,
            'foul': None,
            'is_safety': False,
            'is_placement': True,
            'player': player,
            'called_id': None, 'called_pocket': -1,
            'aim_deg': 0, 'force': 0, 'spin_factor': 0,
            'done': done,
            'cue_after': list(env.cue),
            'balls_after': serialize_balls(env),
            'search_ms': (time.time() - t0) * 1000,
            'value': float(value.item()),
            'reason': f'BIH placement at ({env.cue[0]:.1f}, {env.cue[1]:.1f})',
            **game_state_dict(env),
        }

    if not obs.shot_meta:
        cue_before = list(env.cue)
        balls_before = serialize_balls(env)
        obs_next, reward, done, info = env.step(
            0, 0.0, 0.0, obs,
            record_trajectory=False,
        )
        # Synthesize a brief stub trajectory so the foul reads as a
        # deliberate "no shot taken" beat instead of a silent teleport.
        ball_ids = [0] + [b['id'] for b in balls_before]
        positions = [(cue_before[0], cue_before[1])] + [
            (b['x'], b['y']) for b in balls_before]
        stub_frame = [list(p) for p in positions]
        stub_traj = [stub_frame for _ in range(30)]
        return {
            'trajectory': stub_traj, 'ball_ids_order': ball_ids,
            'pocketed_ids': [], 'scratch': False,
            'foul': info.get('foul'),
            'is_safety': False,
            'player': player,
            'called_id': None, 'called_pocket': -1,
            'aim_deg': 0, 'force': 0, 'spin_factor': 0,
            'done': done,
            'cue_after': list(env.cue),
            'balls_after': serialize_balls(env),
            'search_ms': (time.time() - t0) * 1000,
            'reason': info.get('reason', 'no legal shots'),
            **game_state_dict(env),
        }

    batch = obs_to_batch(obs, _device)
    with torch.no_grad():
        scores, f_means, s_means, _, value, _ = _net.forward(**batch)

    n_legal = len(obs.shot_meta)
    action_idx_val = int(scores[0, :n_legal].argmax().item())
    force_raw_val = float(f_means[0, action_idx_val].item())
    spin_raw_val = float(s_means[0, action_idx_val].item())

    shot = obs.shot_meta[action_idx_val]

    obs_next, reward, done, info = env.step(
        action_idx_val, force_raw_val, spin_raw_val, obs,
        record_trajectory=True, traj_max_frames=600,
    )

    decision_ms = (time.time() - t0) * 1000
    aim_deg = math.degrees(info.get('aim_angle', 0))

    return {
        'trajectory': info.get('trajectory', []),
        'ball_ids_order': info.get('trajectory_ball_ids', []),
        'pocketed_ids': info.get('pocketed_ids', []),
        'scratch': info.get('scratch', False),
        'foul': info.get('foul'),
        'is_safety': info.get('is_safety', False),
        'player': player,
        'called_id': shot.ball_id,
        'called_pocket': shot.pocket_idx,
        'aim_deg': aim_deg,
        'force': info.get('force', 0),
        'spin_factor': info.get('spin', 0),
        'done': done,
        'cue_after': list(env.cue),
        'balls_after': serialize_balls(env),
        'search_ms': decision_ms,
        'value': float(value.item()),
        'reason': info.get('reason', ''),
        **game_state_dict(env),
    }


def handle_undo(payload):
    session_id = payload.get('session_id', 'default')
    if session_id not in _sessions:
        return {'error': 'no such session'}
    stack = _histories.get(session_id, [])
    if not stack:
        return {'error': 'nothing to undo'}
    env = stack.pop()
    _sessions[session_id] = env
    return {
        'session_id': session_id,
        'cue': list(env.cue),
        'balls': serialize_balls(env),
        'table_length': TABLE_LENGTH,
        'table_width': TABLE_WIDTH,
        'ball_radius': R,
        'history_len': len(stack),
        **game_state_dict(env),
    }


def handle_diagnose(payload):
    """Per-aim-point breakdown for the current session: for each own-group
    ball x pocket, report each aim candidate and whether it passed each
    check (corridor clearance is already gated by pocket_aim_candidates).
    """
    session_id = payload.get('session_id', 'default')
    if session_id not in _sessions:
        return {'error': 'no such session'}
    env = _sessions[session_id]

    from shot_enumerator import (ghost_ball, cut_angle_deg, _ghost_on_table,
                                  _segment_blocked, POCKETS, POCKET_NAMES)
    from table_geometry import pocket_aim_candidates

    cue_t = tuple(env.cue)
    ball_map = {bid: tuple(pos) for bid, pos in env.balls.items()}

    if env.phase == 1:  # OPEN_TABLE
        targets = [b for b in env.balls if b != 8]
    elif env._on_8ball():
        targets = [8] if 8 in env.balls else []
    else:
        targets = list(env._my_ball_ids())

    out = []
    for ball_id in targets:
        if ball_id not in ball_map:
            continue
        ball_pos = ball_map[ball_id]
        for p_idx, pocket_pos in enumerate(POCKETS):
            cands = pocket_aim_candidates(ball_pos, p_idx)
            if not cands:
                out.append({'ball_id': ball_id, 'pocket': POCKET_NAMES[p_idx],
                            'aim_idx': -1, 'aim_xy': None,
                            'status': 'REJECT:no_corridor_aim'})
                continue
            for ai, aim_point in enumerate(cands):
                ghost = ghost_ball(ball_pos, aim_point)
                reason = None
                if not _ghost_on_table(ghost):
                    reason = 'ghost_off_table'
                else:
                    bpx = ball_pos[0] - ghost[0]
                    bpy = ball_pos[1] - ghost[1]
                    cgx = ghost[0] - cue_t[0]
                    cgy = ghost[1] - cue_t[1]
                    if bpx * cgx + bpy * cgy <= 0:
                        reason = 'cue_wrong_side'
                    else:
                        cut = cut_angle_deg(cue_t, ghost, ball_pos, aim_point)
                        if cut > 80.0:
                            reason = f'cut_too_steep_{cut:.1f}'
                        elif _segment_blocked(cue_t, ghost, ball_map,
                                               exclude_ids=[ball_id]):
                            reason = 'cue_path_blocked'
                        elif _segment_blocked(ball_pos, POCKETS[p_idx], ball_map,
                                               exclude_ids=[ball_id]):
                            reason = 'ball_to_pocket_blocked'
                        elif _segment_blocked(ball_pos, aim_point, ball_map,
                                               exclude_ids=[ball_id]):
                            reason = 'ball_to_aim_blocked'
                out.append({
                    'ball_id': ball_id,
                    'pocket': POCKET_NAMES[p_idx],
                    'aim_idx': ai,
                    'aim_xy': [round(aim_point[0], 2), round(aim_point[1], 2)],
                    'status': 'KEPT' if reason is None else f'REJECT:{reason}',
                })
    return {
        'session_id': session_id,
        'cue': [round(env.cue[0], 2), round(env.cue[1], 2)],
        'balls': {bid: [round(p[0], 2), round(p[1], 2)]
                  for bid, p in sorted(env.balls.items())},
        'phase': env.phase,
        'groups': env.groups,
        'current_player': env.current_player,
        'targets': targets,
        'candidates': out,
    }


def handle_legal_shots(payload):
    session_id = payload.get('session_id', 'default')
    if session_id not in _sessions:
        return {'error': 'no such session'}
    env = _sessions[session_id]
    if env.phase == GAME_OVER:
        return {'shots': []}

    obs = env.get_obs()
    if not obs.shot_meta:
        return {'shots': []}

    batch = obs_to_batch(obs, _device)
    with torch.no_grad():
        scores, f_means, s_means, safety_logit, value, _ = _net.forward(**batch)

    try:
        temperature = max(1.0, float(payload.get('temperature', 4.0)))
    except (TypeError, ValueError):
        temperature = 4.0

    n_legal = len(obs.shot_meta)
    score_arr = scores[0, :n_legal].cpu().numpy()
    scaled = score_arr / temperature
    m = scaled.max()
    exp_scores = np.exp(scaled - m)
    probs = exp_scores / exp_scores.sum()

    return {
        'shots': [
            {
                'ball_id': s.ball_id,
                'pocket_idx': s.pocket_idx,
                'aim_point': [s.aim_point[0], s.aim_point[1]],
                'is_bank': bool(s.is_bank),
                'is_kick': bool(getattr(s, 'is_kick', False)),
                'reflection_point': (
                    [s.reflection_point[0], s.reflection_point[1]]
                    if s.reflection_point is not None else None),
                'cut_angle_deg': s.cut_angle_deg,
                'prob': float(probs[i]),
            }
            for i, s in enumerate(obs.shot_meta)
        ],
        'value': float(value.item()),
    }


HTML_PATH = HERE / 'demo_eight_ball.html'


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
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'demo_eight_ball.html not found')
        elif path == '/geometry':
            from table_geometry import to_dict
            self._send_json(200, to_dict())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else '{}'
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            self._send_json(400, {'error': 'bad json'})
            return
        try:
            if path == '/start':
                self._send_json(200, handle_start(payload))
            elif path == '/next_shot':
                self._send_json(200, handle_next_shot(payload))
            elif path == '/legal_shots':
                self._send_json(200, handle_legal_shots(payload))
            elif path == '/undo':
                self._send_json(200, handle_undo(payload))
            elif path == '/diagnose':
                self._send_json(200, handle_diagnose(payload))
            else:
                self._send_json(404, {'error': 'unknown endpoint'})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {'error': str(e)})


def main():
    p = argparse.ArgumentParser(description='8-ball AI vs AI demo server')
    p.add_argument('--ckpt',
                   default=str(HERE / 'checkpoints' / 'eight_ball_8ball_v4_best.pt'))
    p.add_argument('--port', type=int, default=8002)
    p.add_argument('--device', default='cpu')
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=4)
    args = p.parse_args()

    load_net(args.ckpt, embed_dim=args.embed_dim, num_heads=args.num_heads,
             num_layers=args.num_layers, device=args.device)

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'8-ball demo at http://localhost:{args.port}/')
    srv.serve_forever()


if __name__ == '__main__':
    main()
