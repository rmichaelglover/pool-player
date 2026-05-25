"""
Legal shot enumerator for pool. The shot-selection problem is geometric and
should not be learned: for each (target ball, pocket) pair, the ghost ball
position is exact, and "is this shot physically possible" is a line-of-sight
check. The network's job is only to *choose among* legal shots (ball/pocket
pair), and to pick force/spin for position play on the next shot.

Public API:
    generate_legal_shots(cue_pos, balls, max_cut_deg=80.0) -> list[LegalShot]

Each LegalShot is a dataclass with fields useful for scoring and execution.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable

from table_geometry import (TABLE_LENGTH, TABLE_WIDTH, BALL_R as R,
                              POCKETS, POCKET_NAMES,
                              CORNER_RAIL_OFFSET, SIDE_HALF, pocket_captures,
                              optimal_pocket_aim)

# Backward-compat: some older callers expect a per-pocket radius (was used
# only as a corner/side flag — corner if radius < 2.6). Geometry is no
# longer circular, but we preserve the data shape so existing imports keep
# working. Use POCKET_NAMES or pocket_captures(...) directly for new code.
POCKET_RADII = [2.5, 2.75, 2.5, 2.5, 2.75, 2.5]


@dataclass
class LegalShot:
    ball_id: int
    pocket_idx: int
    aim_point: tuple              # (x, y) — chosen aim point on cushion-back chord
                                  # (NOT the pocket's nominal aim center —
                                  # adjusted per-shot so the trajectory threads
                                  # the cushion corridor with maximum clearance)
    ghost_pos: tuple              # (x, y) — where cue center must be at contact
    aim_angle: float              # radians, atan2(ghost_y - cue_y, ghost_x - cue_x)
    cut_angle_deg: float          # 0 = straight-in, 90 = impossible grazing
    cue_to_ghost_dist: float
    ball_to_pocket_dist: float

    @property
    def difficulty(self) -> float:
        """Rough difficulty metric — lower = easier.

        Heuristic: product of cut-angle penalty and total distance.
        Straight shots with short travel are easiest."""
        # Cut angle factor: 1 at 0°, rises steeply past 45°
        cut_factor = 1.0 / max(0.1, math.cos(math.radians(self.cut_angle_deg)))
        total_dist = self.cue_to_ghost_dist + self.ball_to_pocket_dist
        return cut_factor * (total_dist / 20.0)


def ghost_ball(ball_pos, pocket_pos):
    """Point the cue center must occupy at the instant of contact to send
    the object ball cleanly toward the pocket."""
    bx, by = ball_pos
    px, py = pocket_pos
    dx, dy = px - bx, py - by
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (bx, by)
    return (bx - 2 * R * dx / d, by - 2 * R * dy / d)


def cut_angle_deg(cue_pos, ghost_pos, ball_pos, pocket_pos):
    """Cut angle = angle between (cue→ghost) and (ball→pocket) directions, in degrees.
    0° = straight shot, 90° = grazing / impossible."""
    cx, cy = cue_pos
    gx, gy = ghost_pos
    bx, by = ball_pos
    px, py = pocket_pos
    v1x, v1y = gx - cx, gy - cy
    v2x, v2y = px - bx, py - by
    n1 = math.hypot(v1x, v1y); n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _segment_blocked(p1, p2, obstacles, exclude_ids: Iterable[int] = (),
                     clearance=2 * R + 0.4):
    """True if the straight segment from p1 to p2 passes within `clearance`
    of the center of any obstacle ball (excluding balls in exclude_ids).

    The pure ball-ball collision threshold is 2R (sum of radii). We add a
    small safety margin (0.4") so enumerated shots have a clear corridor —
    grazing passes where the cue just barely brushes another ball lead to
    unpredictable physics and are effectively not clean shots in real play.

    Endpoint proximity is also checked: if an obstacle is within `clearance`
    of either endpoint (not strictly on the segment), it blocks. This catches
    the case where a ball is adjacent to the target — its center may sit just
    past the ghost position, so its projection onto the segment gives t > seg_len,
    but the cue arriving at ghost is still too close to it."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return False
    ux, uy = dx / seg_len, dy / seg_len
    excl = set(exclude_ids)
    c_sq = clearance * clearance
    for bid, (bx, by) in obstacles.items():
        if bid in excl:
            continue
        ex, ey = bx - x1, by - y1
        t = ex * ux + ey * uy
        # Forward-component threshold: if the obstacle's projection onto the
        # motion direction is small or negative, the obstacle is essentially
        # perpendicular to (or behind) the cue's motion at p1. The cue is
        # either already past it (t<0) or moving sideways relative to it
        # (t≈0). In either case, distance to obstacle is non-decreasing along
        # the forward path, so no collision. The 0.5·BALL_R cutoff covers
        # numerical drift in u (when ghost isn't perfectly axis-aligned with
        # p1) and the frozen-ball-perpendicular case where another ball
        # touches the cue at right angles to the shot line. In a valid pool
        # state, distance(p1, obstacle) >= 2R, so the cue is not overlapping
        # at the start.
        if t <= 0.5 * R:
            continue
        if t > seg_len:
            # Obstacle is PAST the segment end (p2). The cue stops at p2, so
            # any obstacle within clearance of p2 would be contacted at rest.
            dx2 = bx - x2; dy2 = by - y2
            if dx2 * dx2 + dy2 * dy2 < c_sq:
                return True
            continue
        perp_x = ex - t * ux
        perp_y = ey - t * uy
        if perp_x * perp_x + perp_y * perp_y < c_sq:
            return True
    return False


def _ghost_on_table(ghost_pos):
    """Ghost position must be reachable — cue ball must fit there.

    Cue must be at least R from each rail line (cushion), and not inside any
    pocket capture region. Bounding box is the simple rail check; the pocket
    test rules out ghost positions deep in a pocket (where a real cue ball
    can't be staged for the shot)."""
    gx, gy = ghost_pos
    if not (R < gx < TABLE_LENGTH - R and R < gy < TABLE_WIDTH - R):
        return False
    for i in range(6):
        if pocket_captures(gx, gy, i):
            return False
    return True


def generate_legal_shots(cue_pos, balls, max_cut_deg=80.0, min_pocket_dist=1.0):
    """
    Enumerate all legal direct shots for the given state.

    A shot (ball, pocket) is legal if:
      1. Ghost ball is on the table (not behind a rail cushion).
      2. The ball→pocket straight-line segment is not obstructed by another ball.
      3. The cue→ghost segment is not obstructed by another ball (excluding the target).
      4. Cut angle is below `max_cut_deg`.
      5. Ball is at least `min_pocket_dist` from the pocket (otherwise it's already
         in/at pocket, which shouldn't happen with live balls).

    Args:
        cue_pos: (x, y) cue ball position
        balls: dict {ball_id: (x, y)} of object balls on table
        max_cut_deg: reject shots with cut angle above this
        min_pocket_dist: reject if ball is closer than this to the pocket (degenerate)

    Returns:
        list[LegalShot], possibly empty
    """
    shots: list[LegalShot] = []
    cue_t = tuple(cue_pos)
    ball_map = {bid: tuple(pos) for bid, pos in balls.items()}

    for ball_id, ball_pos in ball_map.items():
        for p_idx, pocket_pos in enumerate(POCKETS):
            # Degenerate: ball already at pocket (ignore)
            if math.hypot(pocket_pos[0] - ball_pos[0],
                          pocket_pos[1] - ball_pos[1]) < min_pocket_dist:
                continue

            # Compute optimal per-shot aim point. Returns None if no direct
            # straight-line trajectory threads the pocket corridor with ball-
            # radius clearance from each cushion corner — subsumes the old
            # `can_pocket_directly` check. The aim point sits on the cushion-
            # back chord (deeper than the pocket-mouth center), which is what
            # real players use for steep approaches.
            aim_point = optimal_pocket_aim(ball_pos, p_idx)
            if aim_point is None:
                continue

            ghost = ghost_ball(ball_pos, aim_point)
            if not _ghost_on_table(ghost):
                continue

            # Cue must be "behind" the ball relative to the aim — i.e.,
            # (cue→ghost) should be in roughly the same direction as (ghost→ball).
            # This rules out "shooting through the ball" from the far side.
            bpx = ball_pos[0] - ghost[0]
            bpy = ball_pos[1] - ghost[1]
            cgx = ghost[0] - cue_t[0]
            cgy = ghost[1] - cue_t[1]
            if bpx * cgx + bpy * cgy <= 0:
                continue

            cut = cut_angle_deg(cue_t, ghost, ball_pos, aim_point)
            if cut > max_cut_deg:
                continue

            # Line-of-sight: cue center travels from cue_t to ghost. Exclude
            # the target ball itself (it's at ball_pos, and ghost is offset
            # such that cue+target contact at ghost is clean).
            if _segment_blocked(cue_t, ghost, ball_map, exclude_ids=[ball_id]):
                continue

            # Line-of-sight: target ball travels from ball_pos toward the
            # pocket entrance. We check TWO segments:
            #   (a) ball → POCKETS[idx] (pocket-entrance line) — this is the
            #       OB's actual physical trajectory while still on the table.
            #   (b) ball → aim_point (deep-throat aim) — catches obstacles
            #       inside the throat region.
            # The aim_point alone is insufficient because for balls near a
            # rail, the deep-throat aim is OFF the table (y < 0 for top
            # pockets), and the angled segment misses obstacles that sit
            # right on the rail in the OB's actual path.
            if _segment_blocked(ball_pos, POCKETS[p_idx], ball_map,
                                 exclude_ids=[ball_id]):
                continue
            if _segment_blocked(ball_pos, aim_point, ball_map, exclude_ids=[ball_id]):
                continue

            aim = math.atan2(ghost[1] - cue_t[1], ghost[0] - cue_t[0])
            shots.append(LegalShot(
                ball_id=ball_id,
                pocket_idx=p_idx,
                aim_point=aim_point,
                ghost_pos=ghost,
                aim_angle=aim,
                cut_angle_deg=cut,
                cue_to_ghost_dist=math.hypot(ghost[0] - cue_t[0], ghost[1] - cue_t[1]),
                ball_to_pocket_dist=math.hypot(pocket_pos[0] - ball_pos[0],
                                                pocket_pos[1] - ball_pos[1]),
            ))
    return shots


def easiest_shot(shots: list[LegalShot]) -> LegalShot | None:
    """Simplest heuristic: pick the shot with lowest difficulty score."""
    if not shots:
        return None
    return min(shots, key=lambda s: s.difficulty)


if __name__ == '__main__':
    # Smoke test
    cue = (15.0, 25.0)
    balls = {i + 1: (30.0 + i * 5, 20.0 + (i % 3) * 5) for i in range(5)}
    shots = generate_legal_shots(cue, balls)
    print(f'{len(shots)} legal shots from cue={cue} with {len(balls)} balls:')
    for s in sorted(shots, key=lambda s: s.difficulty)[:10]:
        print(f'  ball {s.ball_id} → {POCKET_NAMES[s.pocket_idx]:6s} '
              f'(cut={s.cut_angle_deg:5.1f}°  '
              f'cue→ghost={s.cue_to_ghost_dist:5.1f}  '
              f'ball→pocket={s.ball_to_pocket_dist:5.1f}  '
              f'diff={s.difficulty:5.2f})')
