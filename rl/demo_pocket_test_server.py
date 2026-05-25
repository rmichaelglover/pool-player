"""Manual shot tester — pick a (cut, Dc, Do) setup, dial in force/spin/noise,
fire N trials, see cumulative pocketed/scratched stats. Useful for sanity-
checking simulator behavior on a single shot.

GET  /              → serves demo_pocket_test.html
GET  /setup?...     → returns geometry for the current sliders
GET  /shoot?...     → runs N noisy trials, returns counts + last trajectory

Usage:  python3 demo_pocket_test_server.py [--port 8002]
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pool_sim import simulate_shot

R = 1.125
POCKET_AIM = (3.0, 3.0)  # TL corner aim point
_DIAG_LEN = math.hypot(100 - 6, 50 - 6)
OB_DIR = ((100 - 6) / _DIAG_LEN, (50 - 6) / _DIAG_LEN)


def setup_shot(cut_deg, d_cb, d_op):
    """Returns dict with cue, ob, gb, aim_dir. Picks the cue placement (one
    of two rotation directions) that puts the cue ball furthest from any rail."""
    theta = math.radians(cut_deg)
    ob = (POCKET_AIM[0] + d_op * OB_DIR[0],
          POCKET_AIM[1] + d_op * OB_DIR[1])
    gb = (ob[0] + 2 * R * OB_DIR[0], ob[1] + 2 * R * OB_DIR[1])
    op_dir = (-OB_DIR[0], -OB_DIR[1])
    best = None
    best_m = -1e9
    for sign in (+1, -1):
        cs, sn = math.cos(sign * theta), math.sin(sign * theta)
        cb_to_gb = (op_dir[0] * cs - op_dir[1] * sn,
                    op_dir[0] * sn + op_dir[1] * cs)
        cue = (gb[0] - d_cb * cb_to_gb[0], gb[1] - d_cb * cb_to_gb[1])
        m = min(cue[0] - R, 100 - R - cue[0], cue[1] - R, 50 - R - cue[1])
        if m > best_m:
            best_m, best = m, (cue, cb_to_gb)
    cue, cb_to_gb = best
    return {
        'cue':  list(cue),
        'ob':   list(ob),
        'gb':   list(gb),
        'aim':  list(cb_to_gb),
        'pocket_aim': list(POCKET_AIM),
        'on_table': best_m > 0,
    }


def run_trials(cue, ob, aim_dir, force, spin, aim_noise_deg,
                force_noise_pct, spin_noise, n, want_traj=True):
    aim_dx, aim_dy = aim_dir
    aim_ang0 = math.atan2(aim_dy, aim_dx)
    rng = np.random.default_rng()  # fresh seed each call
    n_made = n_pocketed = n_scratched = 0
    last_traj = None
    last_traj_ids = None
    for i in range(n):
        ang = aim_ang0 + rng.normal() * aim_noise_deg * (math.pi / 180.0)
        f = force * (1.0 + rng.normal() * force_noise_pct)
        f = max(20.0, min(280.0, f))
        s = spin + rng.normal() * spin_noise
        s = max(-2.5, min(2.5, s))
        ax, ay = math.cos(ang), math.sin(ang)
        # Record trajectory only for the last trial (to animate).
        record = want_traj and (i == n - 1)
        r = simulate_shot(cue, {1: ob}, ax * f, ay * f, s, ax, ay,
                          record_trajectory=record, traj_max_frames=600)
        ob_in = (1 in r.pocketed_ids)
        cue_scr = r.cue_scratched
        if ob_in:
            n_pocketed += 1
            if not cue_scr:
                n_made += 1
        if cue_scr:
            n_scratched += 1
        if record and r.trajectory is not None:
            last_traj = r.trajectory.tolist()
            last_traj_ids = [0, 1]   # cue, ob
    return {
        'n': n,
        'n_made': n_made,
        'n_pocketed': n_pocketed,
        'n_scratched': n_scratched,
        'trajectory': last_traj,
        'trajectory_ids': last_traj_ids,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quieter

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype='text/html; charset=utf-8'):
        try:
            data = open(path, 'rb').read()
        except FileNotFoundError:
            self.send_response(404); self.end_headers()
            self.wfile.write(b'not found'); return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == '/' or u.path == '/index.html':
                self._file(str(HERE.parent / 'demo_pocket_test.html'))
                return
            if u.path == '/setup':
                cut = float(q.get('cut', 0))
                dc = float(q.get('dc', 10))
                do = float(q.get('do', 10))
                self._json(setup_shot(cut, dc, do))
                return
            if u.path == '/shoot':
                cut = float(q.get('cut', 0))
                dc = float(q.get('dc', 10))
                do = float(q.get('do', 10))
                force = float(q.get('force', 100))
                spin = float(q.get('spin', 0))
                aim_n = float(q.get('aim_noise', 0.2))
                force_n = float(q.get('force_noise', 0.02))
                spin_n = float(q.get('spin_noise', 0.02))
                n = int(q.get('n', 10))
                geo = setup_shot(cut, dc, do)
                if not geo['on_table']:
                    self._json({'error': 'Cue would be off table'}, 400); return
                r = run_trials(geo['cue'], geo['ob'], geo['aim'],
                                force, spin, aim_n, force_n, spin_n, n)
                self._json(r); return
            self._json({'error': 'unknown path'}, 404)
        except Exception as e:
            self._json({'error': str(e)}, 500)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8002)
    args = p.parse_args()
    s = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'serving at http://127.0.0.1:{args.port}/', flush=True)
    s.serve_forever()


if __name__ == '__main__':
    main()
