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
_p7_no_search = False    # if True, skip shot_search and use net's argmax
_p7_search_prob_threshold = 0.0   # search only over shots above this prob
_p7_break_ball_suppression = False   # if True, suppress candidate break balls
_sessions: dict = {}
# Manual-mode preview cache: stores the (action, obs) chosen by /preview
# so that the subsequent /next_shot uses the same decision rather than
# re-running search (which has MC noise and could pick a different action).
# Key: session_id. Cleared when /next_shot consumes it.
_preview_cache: dict = {}


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


def handle_preview(payload):
    """Run the policy's decision logic WITHOUT stepping the env. Returns
    the called ball / pocket / aim that the next /next_shot will use.
    Caches the chosen action by session_id so /next_shot can reuse it
    (avoids MC-search noise giving a different decision when actually
    fired)."""
    session_id = payload.get('session_id', 'default')
    if session_id not in _sessions:
        return {'error': 'no such session'}
    env = _sessions[session_id]
    if env.done:
        return {'called_id': None, 'called_pocket': -1}

    if _policy_mode == 'phase7':
        obs = env.get_obs()
        if not obs.shot_meta:
            _preview_cache.pop(session_id, None)
            return {'called_id': None, 'called_pocket': -1, 'cue_pos': list(env.cue)}
        if _p7_no_search:
            # Bypass search — pick the network's deterministic argmax shot
            # with its mean force/spin. Used to diagnose whether search is
            # making the right call.
            batch = {
                'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(_device),
                'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(_device),
                'ball_is_cue': torch.from_numpy(obs.ball_is_cue).unsqueeze(0).to(_device),
                'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(_device),
                'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(_device),
                'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(_device),
            }
            with torch.no_grad():
                scores, f_means, s_means, _ = _p7_net.forward(**batch)
            chosen_idx = int(scores[0].argmax().item())
            f_raw = float(f_means[0, chosen_idx].item())
            s_raw = float(s_means[0, chosen_idx].item())
            action = (chosen_idx, f_raw, s_raw)
        else:
            action = shot_search_phase7(_p7_net, env, obs, K_shots=8, M_per_shot=8,
                                         device=_device,
                                         noise_samples=_p7_search_mc,
                                         prob_threshold=_p7_search_prob_threshold)
        if action is None:
            _preview_cache.pop(session_id, None)
            return {'called_id': None, 'called_pocket': -1, 'cue_pos': list(env.cue)}
        _preview_cache[session_id] = action
        chosen_idx, f_raw, s_raw = action
        chosen_shot = obs.shot_meta[chosen_idx]
        from pool_game_net import decode_force, decode_spin
        return {
            'called_id': chosen_shot.ball_id,
            'called_pocket': chosen_shot.pocket_idx,
            'aim_point': list(chosen_shot.aim_point),
            'cut_angle_deg': chosen_shot.cut_angle_deg,
            'cue_pos': list(env.cue),
            'force': float(decode_force(f_raw)),
            'spin': float(decode_spin(s_raw)),
        }
    if _policy_mode == 'heuristic':
        ha = heuristic_action(env)
        chosen = ha.chosen_shot
        if chosen is not None:
            return {
                'called_id': chosen.ball_id,
                'called_pocket': chosen.pocket_idx,
                'aim_point': list(chosen.aim_point),
                'cut_angle_deg': chosen.cut_angle_deg,
                'cue_pos': list(env.cue),
            }
        return {'called_id': None, 'called_pocket': -1, 'cue_pos': list(env.cue)}
    # network mode: not currently supported for preview
    return {'called_id': None, 'called_pocket': -1, 'cue_pos': list(env.cue)}


# ── Saved-runs storage ─────────────────────────────────────────────────────
# Runs live in HERE/runs/ as JSON files named "run_<balls>_<timestamp>.json".
# Each save stores only the LAST `TAIL_SHOTS` shots — the ending is the
# diagnostic data, so even for long runs we only keep the final stretch.
# The starting state for the replay is the pre-shot state captured by the
# client at shots[N-TAIL_SHOTS] (the first of the saved shots).
#
# Save policy (caps per category, on a per-file basis):
#   balls >= LONG_RUN_THRESHOLD  -> unlimited (permanent record).
#   balls <= SHORT_RUN_THRESHOLD -> up to SHORT_RUN_MAX (failure catalog).
#   in between -> up to MIDDLE_RUN_MAX.
RUNS_DIR = HERE / 'runs'
LONG_RUN_THRESHOLD = 150
SHORT_RUN_THRESHOLD = 30
SHORT_RUN_MAX = 20
MIDDLE_RUN_MAX = 30
TAIL_SHOTS = 15


def _runs_listing():
    if not RUNS_DIR.exists():
        return []
    entries = []
    for p in RUNS_DIR.glob('run_*.json'):
        try:
            with open(p) as f:
                data = json.load(f)
            # Distinguish tail-only from full-game saves. Full saves have
            # the "_full" suffix in the filename and store the entire run.
            kind = data.get('kind') or (
                'full' if p.stem.endswith('_full') else 'tail')
            entries.append({
                'filename': p.name,
                'balls': data.get('total_pocketed', 0),
                'reracks': data.get('rerack_count', 0),
                'timestamp': data.get('timestamp', ''),
                'model': data.get('model', ''),
                'n_shots': len(data.get('shots', [])),
                'kind': kind,
            })
        except Exception:
            continue
    # Sort: balls desc, then kind so tail and full of the same run sit
    # adjacent (tail first because endings are usually what people open).
    entries.sort(key=lambda e: (-e['balls'], 0 if e['kind'] == 'tail' else 1))
    return entries


def _count_in_category(category):
    """Count files in RUNS_DIR matching `long` / `short` / `middle`."""
    n = 0
    for p in RUNS_DIR.glob('run_*.json'):
        try:
            run_balls = int(p.name.split('_')[1])
        except (ValueError, IndexError):
            continue
        if category == 'long' and run_balls >= LONG_RUN_THRESHOLD:
            n += 1
        elif category == 'short' and run_balls <= SHORT_RUN_THRESHOLD:
            n += 1
        elif (category == 'middle' and SHORT_RUN_THRESHOLD < run_balls
              < LONG_RUN_THRESHOLD):
            n += 1
    return n


def handle_save_run(payload):
    """Save the LAST TAIL_SHOTS shots of a completed run. Category caps:
    long unlimited, short up to SHORT_RUN_MAX, middle up to MIDDLE_RUN_MAX.
    Payload: { initial_cue, initial_balls, shots: [{..., state_before:
    {cue, balls}}, ...], total_pocketed, rerack_count, timestamp }
    """
    balls = int(payload.get('total_pocketed', 0))
    if balls <= 0:
        return {'saved': False, 'reason': 'no balls pocketed'}
    RUNS_DIR.mkdir(exist_ok=True)
    # Categorize and apply cap.
    if balls >= LONG_RUN_THRESHOLD:
        category = 'long'   # unlimited
    elif balls <= SHORT_RUN_THRESHOLD:
        if _count_in_category('short') >= SHORT_RUN_MAX:
            return {'saved': False, 'reason':
                    f'short-run cap reached ({SHORT_RUN_MAX})'}
        category = 'short'
    else:
        if _count_in_category('middle') >= MIDDLE_RUN_MAX:
            return {'saved': False, 'reason':
                    f'middle-run cap reached ({MIDDLE_RUN_MAX})'}
        category = 'middle'
    full_shots = payload.get('shots', []) or []
    # Slice to last TAIL_SHOTS shots.
    tail = full_shots[-TAIL_SHOTS:]
    if not tail:
        return {'saved': False, 'reason': 'no shots to save'}
    # Determine the replay starting state: first saved shot's state_before
    # if present; otherwise fall back to the original initial state (only
    # accurate if the full run was saved).
    first = tail[0]
    state_before = first.get('state_before') or {}
    init_cue = state_before.get('cue') or payload.get('initial_cue')
    init_balls = state_before.get('balls') or payload.get('initial_balls')
    # Strip state_before from each shot in the saved record (server doesn't
    # need it for replay; client only sent it for the slice computation).
    saved_shots = [{k: v for k, v in s.items() if k != 'state_before'}
                   for s in tail]
    ts = payload.get('timestamp') or time.strftime('%Y%m%d-%H%M%S')
    ckpt = os.path.basename(_ckpt_path or 'unknown')
    record = {
        'initial_cue': init_cue,
        'initial_balls': init_balls,
        'shots': saved_shots,
        'total_pocketed': balls,    # final total at run end (informational)
        'rerack_count': int(payload.get('rerack_count', 0)),
        'tail_shots': len(saved_shots),
        'full_run_shots': len(full_shots),  # original run length
        'timestamp': ts,
        'model': ckpt,
        'category': category,
        'kind': 'tail',
    }
    fname = f'run_{balls:04d}_{ts}.json'
    fpath = RUNS_DIR / fname
    with open(fpath, 'w') as f:
        json.dump(record, f)
    saved_files = [fname]
    # For very long runs, also save the FULL transcript so we can study
    # what strategy enabled the run. Same naming with _full suffix.
    if category == 'long':
        # Compute the original starting state. If client sent it via
        # initial_cue/initial_balls AND there's a state_before in the
        # first shot, the latter overrides (more accurate for live runs).
        if full_shots and 'state_before' in full_shots[0]:
            orig_cue = full_shots[0]['state_before'].get('cue')
            orig_balls = full_shots[0]['state_before'].get('balls')
        else:
            orig_cue = payload.get('initial_cue')
            orig_balls = payload.get('initial_balls')
        full_record_shots = [
            {k: v for k, v in s.items() if k != 'state_before'}
            for s in full_shots
        ]
        full_record = {
            'initial_cue': orig_cue,
            'initial_balls': orig_balls,
            'shots': full_record_shots,
            'total_pocketed': balls,
            'rerack_count': int(payload.get('rerack_count', 0)),
            'tail_shots': None,
            'full_run_shots': len(full_shots),
            'timestamp': ts,
            'model': ckpt,
            'category': category,
            'kind': 'full',
        }
        full_fname = f'run_{balls:04d}_{ts}_full.json'
        with open(RUNS_DIR / full_fname, 'w') as f:
            json.dump(full_record, f)
        saved_files.append(full_fname)
    return {'saved': True, 'filename': fname, 'filenames': saved_files,
            'balls': balls, 'category': category,
            'top_runs': _runs_listing()}


def handle_list_runs(payload):
    """Return metadata for the saved runs (sorted by balls desc)."""
    return {'runs': _runs_listing()}


def handle_get_run(payload):
    """Return the full saved-run JSON for replay."""
    fname = payload.get('filename', '')
    if '/' in fname or '..' in fname or not fname.startswith('run_'):
        return {'error': 'invalid filename'}
    fpath = RUNS_DIR / fname
    if not fpath.exists():
        return {'error': 'not found'}
    with open(fpath) as f:
        return json.load(f)


def handle_replay_shot(payload):
    """Run one saved shot's physics during replay. Initializes the env to
    the saved initial state on the first call (session must have been
    /start'ed first). Payload: session_id, ball_id, pocket_idx, force_raw,
    spin_raw, OPTIONAL initial_cue/initial_balls (only used on first
    /replay_shot of a session, to set the starting state)."""
    session_id = payload.get('session_id', 'default')
    if session_id not in _sessions:
        return {'error': 'no such session — call /start first'}
    env = _sessions[session_id]
    # On the first shot of a replay, override the env's randomly-generated
    # initial state with the saved one.
    if payload.get('initial_cue') is not None:
        env.cue = list(payload['initial_cue'])
        env.balls = {int(k): list(v) for k, v in payload['initial_balls'].items()}
        env.done = False
        env.shot_idx = 0
        env.total_pocketed = 0
        env.rerack_count = 0
        env._post_rerack_break_pending = False
        env._break_ball_id_after_rerack = None
        env.pending_rerack = False
    target_ball = int(payload['ball_id'])
    target_pocket = int(payload['pocket_idx'])
    obs = env.get_obs()
    if not obs.shot_meta:
        return {'error': 'no legal shots in current state', 'done': True}
    chosen_idx = None
    for i, sh in enumerate(obs.shot_meta):
        if sh.ball_id == target_ball and sh.pocket_idx == target_pocket:
            chosen_idx = i
            break
    if chosen_idx is None:
        return {'error': f'saved shot {target_ball}->{target_pocket} not in '
                          f'current legal-shots set; enumerator may have changed',
                'done': False}
    f_raw = float(payload['force_raw'])
    s_raw = float(payload['spin_raw'])
    rerack_before = env.rerack_count
    _, reward, done, info = env.step(
        chosen_idx, f_raw, s_raw, obs,
        record_trajectory=True, traj_max_frames=600,
    )
    rerack_happened = env.rerack_count > rerack_before
    chosen_shot = obs.shot_meta[chosen_idx]
    return {
        'trajectory': info.get('trajectory', []),
        'ball_ids_order': info.get('trajectory_ball_ids', []),
        'pocketed_ids': info.get('pocketed_ids', []),
        'called_id': chosen_shot.ball_id,
        'called_pocket': chosen_shot.pocket_idx,
        'called_actual_pocket': chosen_shot.pocket_idx if info.get('called_ok') else -1,
        'called_shot_valid': info.get('called_ok', False),
        'rerack_happened': rerack_happened,
        'cue_after': list(env.cue),
        'balls_after': serialize_balls(env),
        'shot_idx': env.shot_idx,
        'rerack_count': env.rerack_count,
        'total_pocketed': env.total_pocketed,
        'done': env.done,
        'force': f_raw,
        'spin_factor': s_raw,
    }


def handle_dump_state(payload):
    """Return current env state (cue + ball positions) for debugging.
    If session_id matches: return that one session.
    Otherwise: return ALL sessions so the caller can identify which matches
    their browser view (multiple sessions can accumulate across refreshes)."""
    session_id = payload.get('session_id', None)
    if session_id and session_id in _sessions:
        env = _sessions[session_id]
        return {
            'session_id': session_id,
            'cue': list(env.cue),
            'balls': {bid: list(pos) for bid, pos in env.balls.items()},
            'shot_idx': env.shot_idx,
        }
    # No specific session — return all of them.
    if not _sessions:
        return {'error': 'no active sessions; click Start in the browser first'}
    return {
        'all_sessions': [
            {
                'session_id': sid,
                'cue': list(e.cue),
                'balls': {bid: list(pos) for bid, pos in e.balls.items()},
                'shot_idx': e.shot_idx,
            }
            for sid, e in _sessions.items()
        ]
    }


def handle_legal_shots(payload):
    """Return all legal shots from the current env state — used by the
    demo's manual mode to display dashed OB→pocket lines. In phase7 mode
    each shot also carries the network's softmax probability (so the demo
    can show per-shot prob alongside the line).

    Optional payload['temperature']: float ≥ 1. Divides the network's logits
    before softmax, flattening the displayed probabilities. T=1 is the raw
    network output; T=4 produces noticeably flatter rankings without
    changing which shot is highest."""
    session_id = payload.get('session_id', 'default')
    if session_id not in _sessions:
        return {'error': 'no such session'}
    env = _sessions[session_id]
    if env.done:
        return {'shots': []}
    try:
        temperature = max(1.0, float(payload.get('temperature', 1.0)))
    except (TypeError, ValueError):
        temperature = 1.0

    if _policy_mode == 'phase7':
        # Phase7Env's obs already carries the legal-shot list (shot_meta) in
        # the same order as the network's shot tokens, so logits at index i
        # correspond to obs.shot_meta[i].
        obs = env.get_obs()
        if not obs.shot_meta:
            return {'shots': []}
        batch = {
            'balls': torch.from_numpy(obs.balls).unsqueeze(0).to(_device),
            'ball_mask': torch.from_numpy(obs.ball_mask).unsqueeze(0).to(_device),
            'ball_is_cue': torch.from_numpy(obs.ball_is_cue).unsqueeze(0).to(_device),
            'pockets': torch.from_numpy(obs.pockets).unsqueeze(0).to(_device),
            'shots': torch.from_numpy(obs.shots).unsqueeze(0).to(_device),
            'shot_mask': torch.from_numpy(obs.shot_mask).unsqueeze(0).to(_device),
        }
        with torch.no_grad():
            scores, _, _, _ = _p7_net.forward(**batch)
        n_legal = len(obs.shot_meta)
        score_arr_raw = scores[0, :n_legal].cpu().numpy()
        # Apply temperature before softmax: dividing logits by T > 1 flattens
        # the resulting distribution. T=1 = raw network output.
        scaled_raw = score_arr_raw / temperature
        m_raw = scaled_raw.max()
        exp_raw = np.exp(scaled_raw - m_raw)
        probs_raw = exp_raw / exp_raw.sum()
        # Suppressed scores (only different from raw when suppression is on).
        if _p7_break_ball_suppression:
            from shot_search_phase7 import apply_break_ball_suppression
            score_arr = apply_break_ball_suppression(
                score_arr_raw, obs.shot_meta, env.balls)
        else:
            score_arr = score_arr_raw
        scaled = score_arr / temperature
        m = scaled.max()
        exp_scores = np.exp(scaled - m)
        probs_legal = exp_scores / exp_scores.sum()
        # When suppression is enabled, return candidate ball IDs so the demo
        # can highlight them.
        candidate_ids = []
        if _p7_break_ball_suppression:
            from shot_search_phase7 import is_candidate_break_ball
            candidate_ids = [bid for bid, pos in env.balls.items()
                              if is_candidate_break_ball(pos)]
        return {
            'shots': [
                {
                    'ball_id': s.ball_id,
                    'pocket_idx': s.pocket_idx,
                    'cut_angle_deg': s.cut_angle_deg,
                    'aim_point': list(s.aim_point),
                    'prob': float(probs_legal[i]),
                    'prob_raw': float(probs_raw[i]),
                }
                for i, s in enumerate(obs.shot_meta)
            ],
            'break_candidate_ball_ids': candidate_ids,
        }

    # Heuristic / network modes: no per-shot probability available.
    from shot_enumerator import generate_legal_shots
    shots = generate_legal_shots(env.cue, env.balls, max_cut_deg=80.0)
    return {
        'shots': [
            {
                'ball_id': s.ball_id,
                'pocket_idx': s.pocket_idx,
                'cut_angle_deg': s.cut_angle_deg,
                'aim_point': list(s.aim_point),
                'prob': None,
            }
            for s in shots
        ]
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
        return _handle_phase7_shot(env, t0, session_id=session_id)

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


def _handle_phase7_shot(env: Phase7Env, t0, session_id: str = None):
    """Run one shot using the Phase 7 network in deterministic mode.
    If a previewed action is cached for this session, use it instead of
    re-running search (so the demo's manual mode shows a stable plan)."""
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

    # If /preview ran search recently for this session, reuse its action so
    # what the user saw in the preview matches what's executed. Otherwise
    # run search fresh — or, if --p7_no_search is set, use the network's
    # deterministic argmax (no V-head bootstrap).
    cached = _preview_cache.pop(session_id, None) if session_id else None
    if cached is not None:
        action = cached
    elif _p7_no_search:
        with torch.no_grad():
            scores, f_means, s_means, _ = _p7_net.forward(**batch)
        chosen_idx = int(scores[0].argmax().item())
        f_raw = float(f_means[0, chosen_idx].item())
        s_raw = float(s_means[0, chosen_idx].item())
        action = (chosen_idx, f_raw, s_raw)
    else:
        # Depth-1 shot search at inference (K=8, M=8). With execution noise
        # active, MC averaging over `_p7_search_mc` rollouts per candidate
        # makes search prefer shots that succeed *robustly* under noise.
        action = shot_search_phase7(_p7_net, env, obs, K_shots=8, M_per_shot=8,
                                     device=_device,
                                     noise_samples=_p7_search_mc,
                                     prob_threshold=_p7_search_prob_threshold,
                                     break_ball_suppression=_p7_break_ball_suppression)
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
        'force_raw': float(f_raw),     # for save/replay
        'spin_raw': float(s_raw),      # for save/replay
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
            elif path == '/legal_shots':
                self._send_json(200, handle_legal_shots(payload))
            elif path == '/preview':
                self._send_json(200, handle_preview(payload))
            elif path == '/dump_state':
                self._send_json(200, handle_dump_state(payload))
            elif path == '/save_run':
                self._send_json(200, handle_save_run(payload))
            elif path == '/list_runs':
                self._send_json(200, handle_list_runs(payload))
            elif path == '/get_run':
                self._send_json(200, handle_get_run(payload))
            elif path == '/replay_shot':
                self._send_json(200, handle_replay_shot(payload))
            else:
                self._send_json(404, {'error': 'unknown endpoint'})
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {'error': str(e)})


def main():
    global _policy_mode, _p7_env_mode, _p7_net, _device, _p7_noise, _p7_search_mc
    global _p7_no_search, _p7_search_prob_threshold, _p7_break_ball_suppression
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
    p.add_argument('--p7_no_search', action='store_true',
                   help='Bypass shot_search at inference; use the network\'s '
                        'deterministic argmax shot. Diagnostic for whether '
                        'search is hurting in cases where the network is '
                        'confident.')
    p.add_argument('--p7_search_prob_threshold', type=float, default=0.0,
                   help='Search only over shots whose network probability '
                        'is at least this. 0 (default) = search top-K by '
                        'score regardless. 0.001 = ignore shots the network '
                        'has effectively rejected. Prevents search from '
                        'overruling a confident network pick.')
    p.add_argument('--p7_search_verbose', action='store_true',
                   help='Print per-candidate (ball, pocket, net_prob, '
                        'imm_r, next_V, Q) breakdown each search call. '
                        'Diagnostic for cases where search disagrees with '
                        'the network argmax.')
    p.add_argument('--p7_break_ball_suppression', action='store_true',
                   help='Tier 1 break-ball protection: suppress probability '
                        'of shots whose target ball is a candidate break ball '
                        '(near rack apex, not in rack). Suppression scales '
                        'with ball count — heavy when many balls left, none '
                        'with few left. Soft (mul=0.1 minimum), so the agent '
                        'can still pick a candidate if forced.')
    args = p.parse_args()
    _policy_mode = args.policy
    _p7_env_mode = args.p7_env
    _p7_noise = {
        'aim_noise_deg': args.p7_aim_noise_deg,
        'force_noise_pct': args.p7_force_noise_pct,
        'spin_noise': args.p7_spin_noise,
    }
    _p7_search_mc = args.p7_search_mc
    _p7_no_search = args.p7_no_search
    _p7_search_prob_threshold = args.p7_search_prob_threshold
    _p7_break_ball_suppression = args.p7_break_ball_suppression
    if args.p7_search_verbose:
        import shot_search_phase7 as _sp7
        _sp7.set_verbose(True)
        print('search verbose: ON (per-candidate breakdown will print)')
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
