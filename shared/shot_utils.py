"""Shot utility functions shared by 14.1 and 8-ball games."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_geometry import BALL_R as R, POCKETS as _POCKETS, pocket_captures as _pcap
from rack_geometry import RACK_APEX, RACK_POSITIONS

HEAD_SPOT = (25.0, 25.0)
HEAD_SPOT_ALT = (25.0, 20.0)


def first_ball_struck(cue_pos, aim_angle, balls_dict):
    """Geometric 'called ball': the first object ball that cue_pos's aim line
    would strike (treating cue as a point moving along aim_angle, and each
    object ball as a disk of radius 2R — collision threshold for cue+ball).

    Returns (ball_id, distance_to_contact) or (None, inf) if aim misses all balls.
    """
    cx, cy = cue_pos
    vx, vy = math.cos(aim_angle), math.sin(aim_angle)
    best_id, best_t = None, float('inf')
    two_r_sq = (2 * R) ** 2
    for bid, (bx, by) in balls_dict.items():
        dx, dy = bx - cx, by - cy
        t = dx * vx + dy * vy
        if t <= 0:
            continue
        perp_sq = max(0.0, (dx * dx + dy * dy) - t * t)
        if perp_sq > two_r_sq:
            continue
        t_contact = t - math.sqrt(two_r_sq - perp_sq)
        if t_contact < best_t:
            best_t = t_contact
            best_id = bid
    return best_id, best_t


_PX = [p[0] for p in _POCKETS]
_PY = [p[1] for p in _POCKETS]


def called_pocket_index(cue_pos, aim_angle, ball_pos, max_perp=3.0):
    """Given cue position, aim angle, and called ball position, return which
    pocket (0-5) the ball would naturally head to — defined as the pocket
    closest to the object ball's post-contact straight-line trajectory
    (ghost-ball geometry).

    Returns the pocket index (0-5), or -1 if aim misses the ball OR if no
    pocket is geometrically reachable from the ball's intended trajectory.
    """
    cx, cy = cue_pos
    vx, vy = math.cos(aim_angle), math.sin(aim_angle)
    bx, by = ball_pos
    dx, dy = bx - cx, by - cy
    t = dx * vx + dy * vy
    if t <= 0:
        return -1
    perp_sq = max(0.0, (dx * dx + dy * dy) - t * t)
    two_r_sq = (2 * R) ** 2
    if perp_sq > two_r_sq:
        return -1
    t_contact = t - math.sqrt(two_r_sq - perp_sq)
    contact_x = cx + t_contact * vx
    contact_y = cy + t_contact * vy
    ex = bx - contact_x
    ey = by - contact_y
    mag = math.hypot(ex, ey)
    if mag < 1e-6:
        return -1
    ex /= mag; ey /= mag
    best_p, best_perp = -1, float('inf')
    for p in range(6):
        pdx = _PX[p] - bx
        pdy = _PY[p] - by
        proj = pdx * ex + pdy * ey
        if proj <= 0:
            continue
        perp_x = pdx - proj * ex
        perp_y = pdy - proj * ey
        perp = math.hypot(perp_x, perp_y)
        if perp < best_perp:
            best_perp = perp; best_p = p
    if best_perp > max_perp:
        return -1
    return best_p


def pocket_index_of(pos, tol=0.2):
    """Return which pocket (0-5) a ball's final position falls into.

    Uses throat-based capture from table_geometry: corners are captured when
    the ball center crosses the diagonal throat line into the corner; sides
    when the ball center crosses the cushion line within the mouth.
    Returns -1 if not in any pocket."""
    for p in range(6):
        if _pcap(pos[0], pos[1], p):
            return p
    return -1


def rerack_positions():
    """Return the 14 positions used on rerack (head-ball apex is empty)."""
    return RACK_POSITIONS[1:]


def relocate_break_ball(remaining_pos, cue_pos):
    """If the remaining (break) ball is in the rack area, relocate to head spot.
    If head spot is blocked by cue, use alternate position."""
    if math.hypot(remaining_pos[0] - RACK_APEX[0],
                  remaining_pos[1] - RACK_APEX[1]) < 8.0:
        for candidate in [HEAD_SPOT, HEAD_SPOT_ALT, (25.0, 30.0), (30.0, 25.0)]:
            if math.hypot(cue_pos[0] - candidate[0],
                          cue_pos[1] - candidate[1]) > 3 * R:
                return list(candidate)
        return list(HEAD_SPOT)
    return list(remaining_pos)
