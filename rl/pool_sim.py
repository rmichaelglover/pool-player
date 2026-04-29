"""
Python wrapper for the C pool physics simulation.
Provides simulate_shot() which runs a full shot with ball physics,
spin effects, and pocket detection.
"""
import ctypes, os, math
import numpy as np
from dataclasses import dataclass
from typing import List, Set

_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, 'libpool_sim.so'))

_lib.simulate_shot.restype = ctypes.c_int
_lib.simulate_shot.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # pos_in
    ctypes.c_int,                      # n_balls
    ctypes.c_double, ctypes.c_double,  # cue_vx, cue_vy
    ctypes.c_double,                   # spin_factor (continuous, = wy/(vx/R))
    ctypes.c_double, ctypes.c_double,  # aim_dx, aim_dy
    ctypes.POINTER(ctypes.c_double),  # pos_out
    ctypes.POINTER(ctypes.c_int),     # pocketed_out
    ctypes.POINTER(ctypes.c_int),     # hit_ball
    ctypes.POINTER(ctypes.c_int),     # hit_rail
    ctypes.POINTER(ctypes.c_double),  # traj_out (may be NULL)
    ctypes.POINTER(ctypes.c_int),     # traj_n_out (may be NULL)
    ctypes.c_int,                      # traj_max_frames
]

# Mapping from legacy discrete spin_type → continuous spin_factor.
# Kept so existing RL training callers (spin_type=0/1/2) still work.
# Chosen to roughly match prior behavior ratios, but note that 0 is
# now *true* center hit (no spin), not natural roll.
_SPIN_TYPE_TO_FACTOR = {0: 0.0, 1: 1.3, 2: -0.5}


@dataclass
class ShotResult:
    """Result of a physics-simulated shot."""
    final_positions: dict    # {ball_id: (x, y)} for all balls
    pocketed_ids: Set[int]   # ball IDs that were pocketed during the shot
    cue_scratched: bool      # True if cue ball (id=0) was pocketed
    hit_ball: bool           # True if cue ball contacted any object ball
    hit_rail: bool           # True if any ball hit a cushion after contact
    trajectory: object = None  # Optional: np.ndarray (frames, n_balls, 2)


def simulate_shot(cue_pos, ball_positions, cue_vx, cue_vy,
                  spin, aim_dx, aim_dy,
                  record_trajectory=False, traj_max_frames=600):
    """
    Run a physics-simulated shot.

    Args:
        cue_pos: (x, y) cue ball position
        ball_positions: dict {ball_id: (x, y)} of object balls on table
        cue_vx, cue_vy: cue ball initial velocity
        spin: float spin_factor (continuous), OR int legacy spin_type
              (0=center, 1=follow, 2=draw — mapped to factors 0, 1.3, -0.5).
              spin_factor = wy_initial / (vx/R); sensible range [-2, +2].
        aim_dx, aim_dy: normalized aim direction

    Returns:
        ShotResult with final positions and outcomes.
    """
    # Accept legacy int spin_type for backward compat with RL training code.
    if isinstance(spin, int) and not isinstance(spin, bool):
        spin_factor = _SPIN_TYPE_TO_FACTOR.get(spin, 0.0)
    else:
        spin_factor = float(spin)
    # Build ball array: cue first, then object balls in ID order
    ball_ids = [0] + sorted(ball_positions.keys())
    n = len(ball_ids)

    pos_in = (ctypes.c_double * (n * 2))()
    pos_in[0] = cue_pos[0]
    pos_in[1] = cue_pos[1]
    for i, bid in enumerate(ball_ids):
        if bid == 0:
            continue
        pos_in[i*2] = ball_positions[bid][0]
        pos_in[i*2+1] = ball_positions[bid][1]

    pos_out = (ctypes.c_double * (n * 2))()
    pocketed_out = (ctypes.c_int * n)()
    hit_ball = ctypes.c_int(0)
    hit_rail = ctypes.c_int(0)

    if record_trajectory:
        traj_buf = (ctypes.c_double * (traj_max_frames * n * 2))()
        traj_n = ctypes.c_int(0)
        traj_ptr = traj_buf
        traj_n_ptr = ctypes.byref(traj_n)
        traj_max = traj_max_frames
    else:
        traj_buf = None
        traj_n = None
        traj_ptr = ctypes.POINTER(ctypes.c_double)()   # NULL
        traj_n_ptr = ctypes.POINTER(ctypes.c_int)()    # NULL
        traj_max = 0

    _lib.simulate_shot(
        pos_in, n,
        ctypes.c_double(cue_vx), ctypes.c_double(cue_vy),
        ctypes.c_double(spin_factor),
        ctypes.c_double(aim_dx), ctypes.c_double(aim_dy),
        pos_out, pocketed_out,
        ctypes.byref(hit_ball), ctypes.byref(hit_rail),
        traj_ptr, traj_n_ptr, ctypes.c_int(traj_max),
    )

    final = {}
    pocketed = set()
    for i, bid in enumerate(ball_ids):
        final[bid] = (pos_out[i*2], pos_out[i*2+1])
        if pocketed_out[i]:
            pocketed.add(bid)

    trajectory = None
    if record_trajectory:
        frames = traj_n.value
        arr = np.frombuffer(traj_buf, dtype=np.float64,
                             count=frames * n * 2).copy()
        trajectory = arr.reshape(frames, n, 2)

    return ShotResult(
        final_positions=final,
        pocketed_ids=pocketed,
        cue_scratched=(0 in pocketed),
        hit_ball=bool(hit_ball.value),
        hit_rail=bool(hit_rail.value),
        trajectory=trajectory,
    )
