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
                              POCKETS, POCKET_NAMES, CUSHIONS, FACINGS,
                              CORNER_RAIL_OFFSET, SIDE_HALF, pocket_captures,
                              optimal_pocket_aim, pocket_aim_candidates,
                              _POCKET_FACING_PAIRS, _SIDE_POCKETS)

# Backward-compat: some older callers expect a per-pocket radius (was used
# only as a corner/side flag — corner if radius < 2.6). Geometry is no
# longer circular, but we preserve the data shape so existing imports keep
# working. Use POCKET_NAMES or pocket_captures(...) directly for new code.
POCKET_RADII = [2.5, 2.75, 2.5, 2.5, 2.75, 2.5]


@dataclass
class LegalShot:
    ball_id: int
    pocket_idx: int
    aim_point: tuple              # (x, y) — for direct: optimal cushion-back aim
                                  # for bank: virtual (mirrored) pocket position
    ghost_pos: tuple              # (x, y) — where cue center must be at contact
    aim_angle: float              # radians, atan2(ghost_y - cue_y, ghost_x - cue_x)
    cut_angle_deg: float          # 0 = straight-in, 90 = impossible grazing
    cue_to_ghost_dist: float
    ball_to_pocket_dist: float    # for bank: total path through reflection
    is_bank: bool = False
    rail_idx: int = -1            # CUSHIONS index (0-5), -1 for direct
    reflection_point: tuple | None = None
    is_defensive: bool = False    # tier-2: legal contact only, no pocket attempt

    @property
    def difficulty(self) -> float:
        cut_factor = 1.0 / max(0.1, math.cos(math.radians(self.cut_angle_deg)))
        total_dist = self.cue_to_ghost_dist + self.ball_to_pocket_dist
        base = cut_factor * (total_dist / 20.0)
        return base * (1.8 if self.is_bank else 1.0)


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


def _bank_pocket_feasible(approach_pos, pocket_idx):
    """Check if a ball approaching from approach_pos can enter the pocket.

    Like optimal_pocket_aim but without the ghost-on-table constraint —
    the ball is already in motion after a rail bounce."""
    bx, by = approach_pos
    fa_idx, fb_idx = _POCKET_FACING_PAIRS[pocket_idx]
    fa = FACINGS[fa_idx]; fb = FACINGS[fb_idx]
    if pocket_idx in _SIDE_POCKETS:
        endpoints = ((fa[0], fa[1]), (fb[0], fb[1]), (fa[2], fa[3]), (fb[2], fb[3]))
    else:
        endpoints = ((fa[0], fa[1]), (fb[0], fb[1]))
    px, py = POCKETS[pocket_idx]
    dx, dy = px - bx, py - by
    d_len = math.hypot(dx, dy)
    if d_len < 1e-6:
        return True
    for ex, ey in endpoints:
        perp_dist = abs(dy * (ex - bx) - dx * (ey - by)) / d_len
        if perp_dist < R:
            return False
    return True


def _mirror_pocket(pocket_pos, cushion):
    """Mirror pocket position across a rail, offset by ball radius."""
    x1, y1, x2, y2, nx, ny = cushion
    px, py = pocket_pos
    if abs(ny) > 0.5:
        # Horizontal rail (top or bottom). Rail surface at y = y1.
        # OB center bounces at y1 + ny*R (inward by R from cushion).
        rail_y = y1 + ny * R
        return (px, 2 * rail_y - py)
    else:
        # Vertical rail (left or right). Rail surface at x = x1.
        rail_x = x1 + nx * R
        return (2 * rail_x - px, py)


def _bank_reflection_point(ball_pos, virtual_pocket, cushion, margin=2.0):
    """Intersect ball→virtual_pocket line with the rail axis.

    Returns (x, y) of the reflection point, or None if the intersection
    is outside the valid cushion segment (with margin from endpoints)."""
    x1, y1, x2, y2, nx, ny = cushion
    bx, by = ball_pos
    vx, vy = virtual_pocket
    dx, dy = vx - bx, vy - by
    if abs(dy) < 1e-9 and abs(dx) < 1e-9:
        return None

    if abs(ny) > 0.5:
        # Horizontal rail at y = y1. OB center bounces at y1 + ny*R.
        rail_y = y1 + ny * R
        if abs(dy) < 1e-9:
            return None
        t = (rail_y - by) / dy
        if t <= 0.0:
            return None
        rx = bx + t * dx
        seg_min = min(x1, x2) + margin
        seg_max = max(x1, x2) - margin
        if rx < seg_min or rx > seg_max:
            return None
        return (rx, rail_y)
    else:
        # Vertical rail at x = x1. OB center bounces at x1 + nx*R.
        rail_x = x1 + nx * R
        if abs(dx) < 1e-9:
            return None
        t = (rail_x - bx) / dx
        if t <= 0.0:
            return None
        ry = by + t * dy
        seg_min = min(y1, y2) + margin
        seg_max = max(y1, y2) - margin
        if ry < seg_min or ry > seg_max:
            return None
        return (rail_x, ry)


def generate_bank_shots(cue_pos, balls, max_cut_deg=70.0, min_pocket_dist=1.0):
    """Enumerate all legal one-rail bank shots."""
    shots = []
    cue_t = tuple(cue_pos)
    ball_map = {bid: tuple(pos) for bid, pos in balls.items()}

    for ball_id, ball_pos in ball_map.items():
        for p_idx, pocket_pos in enumerate(POCKETS):
            if math.hypot(pocket_pos[0] - ball_pos[0],
                          pocket_pos[1] - ball_pos[1]) < min_pocket_dist:
                continue

            for c_idx, cushion in enumerate(CUSHIONS):
                virtual_pocket = _mirror_pocket(pocket_pos, cushion)

                refl = _bank_reflection_point(ball_pos, virtual_pocket,
                                              cushion)
                if refl is None:
                    continue

                if not _bank_pocket_feasible(refl, p_idx):
                    continue

                ghost = ghost_ball(ball_pos, virtual_pocket)
                if not _ghost_on_table(ghost):
                    continue

                bpx = ball_pos[0] - ghost[0]
                bpy = ball_pos[1] - ghost[1]
                cgx = ghost[0] - cue_t[0]
                cgy = ghost[1] - cue_t[1]
                if bpx * cgx + bpy * cgy <= 0:
                    continue

                cut = cut_angle_deg(cue_t, ghost, ball_pos, virtual_pocket)
                if cut > max_cut_deg:
                    continue

                if _segment_blocked(cue_t, ghost, ball_map,
                                    exclude_ids=[ball_id]):
                    continue
                if _segment_blocked(ball_pos, refl, ball_map,
                                    exclude_ids=[ball_id]):
                    continue
                if _segment_blocked(refl, pocket_pos, ball_map,
                                    exclude_ids=[ball_id]):
                    continue

                aim = math.atan2(ghost[1] - cue_t[1], ghost[0] - cue_t[0])
                shots.append(LegalShot(
                    ball_id=ball_id,
                    pocket_idx=p_idx,
                    aim_point=virtual_pocket,
                    ghost_pos=ghost,
                    aim_angle=aim,
                    cut_angle_deg=cut,
                    cue_to_ghost_dist=math.hypot(ghost[0] - cue_t[0],
                                                  ghost[1] - cue_t[1]),
                    ball_to_pocket_dist=math.hypot(virtual_pocket[0] - ball_pos[0],
                                                    virtual_pocket[1] - ball_pos[1]),
                    is_bank=True,
                    rail_idx=c_idx,
                    reflection_point=refl,
                ))
    return shots


def generate_legal_shots(cue_pos, balls, max_cut_deg=80.0, min_pocket_dist=1.0,
                         include_banks=False, bank_max_cut_deg=70.0):
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

            # Side pockets emit up to 3 aim candidates spread across the
            # feasible corridor; corner pockets emit 1 (their wide mouth
            # rarely benefits). Each candidate becomes its own LegalShot if
            # it survives the downstream checks below.
            for aim_point in pocket_aim_candidates(ball_pos, p_idx):
                ghost = ghost_ball(ball_pos, aim_point)
                if not _ghost_on_table(ghost):
                    continue

                # Cue must be "behind" the ball relative to the aim — i.e.,
                # (cue→ghost) should be in roughly the same direction as
                # (ghost→ball). This rules out "shooting through the ball"
                # from the far side.
                bpx = ball_pos[0] - ghost[0]
                bpy = ball_pos[1] - ghost[1]
                cgx = ghost[0] - cue_t[0]
                cgy = ghost[1] - cue_t[1]
                if bpx * cgx + bpy * cgy <= 0:
                    continue

                cut = cut_angle_deg(cue_t, ghost, ball_pos, aim_point)
                if cut > max_cut_deg:
                    continue

                # Line-of-sight: cue center travels from cue_t to ghost.
                # Exclude the target ball itself (it's at ball_pos, and
                # ghost is offset such that cue+target contact at ghost is
                # clean).
                if _segment_blocked(cue_t, ghost, ball_map, exclude_ids=[ball_id]):
                    continue

                # Line-of-sight: target ball travels from ball_pos toward
                # the pocket entrance. We check TWO segments:
                #   (a) ball → POCKETS[idx] (pocket-entrance line) — the
                #       OB's actual physical trajectory while still on the
                #       table.
                #   (b) ball → aim_point (deep-throat aim) — catches
                #       obstacles inside the throat region.
                if _segment_blocked(ball_pos, POCKETS[p_idx], ball_map,
                                     exclude_ids=[ball_id]):
                    continue
                if _segment_blocked(ball_pos, aim_point, ball_map,
                                     exclude_ids=[ball_id]):
                    continue

                aim = math.atan2(ghost[1] - cue_t[1], ghost[0] - cue_t[0])
                shots.append(LegalShot(
                    ball_id=ball_id,
                    pocket_idx=p_idx,
                    aim_point=aim_point,
                    ghost_pos=ghost,
                    aim_angle=aim,
                    cut_angle_deg=cut,
                    cue_to_ghost_dist=math.hypot(ghost[0] - cue_t[0],
                                                  ghost[1] - cue_t[1]),
                    ball_to_pocket_dist=math.hypot(pocket_pos[0] - ball_pos[0],
                                                    pocket_pos[1] - ball_pos[1]),
                ))

    if include_banks:
        banks = generate_bank_shots(cue_pos, balls,
                                     max_cut_deg=bank_max_cut_deg,
                                     min_pocket_dist=min_pocket_dist)
        shots.extend(banks)

    return shots


def _project_to_nearest_rail(pos, direction):
    """Where a ball traveling from `pos` in unit `direction` first hits a rail
    (ball-center coordinates, so cushion line is at R / TABLE_DIM-R)."""
    x, y = pos
    nx, ny = direction
    ts = []
    if nx > 1e-9:
        ts.append(((TABLE_LENGTH - R) - x) / nx)
    elif nx < -1e-9:
        ts.append((R - x) / nx)
    if ny > 1e-9:
        ts.append(((TABLE_WIDTH - R) - y) / ny)
    elif ny < -1e-9:
        ts.append((R - y) / ny)
    ts = [t for t in ts if t > 0]
    if not ts:
        return pos
    t = min(ts)
    return (x + nx * t, y + ny * t)


def generate_defensive_shots(cue_pos, balls, target_ids, min_target_dist=4 * R):
    """Tier-2 shots: straight-on contact with each reachable target ball.

    Used when no makeable (pocketable) shot exists, so the AI can attempt
    legal contact instead of forcing a 'no legal shots' foul. Each emitted
    shot aims at the ball center (cut_angle=0); `aim_point` is set to the
    rail point the object ball would reach if struck head-on, so the env's
    step() drives the ball along that line and rail contact provides the
    legal-shot requirement.
    """
    shots: list[LegalShot] = []
    cue_t = tuple(cue_pos)
    ball_map = {bid: tuple(pos) for bid, pos in balls.items()}

    for ball_id in target_ids:
        if ball_id not in ball_map:
            continue
        ball_pos = ball_map[ball_id]
        dx, dy = ball_pos[0] - cue_t[0], ball_pos[1] - cue_t[1]
        dist = math.hypot(dx, dy)
        if dist < min_target_dist:
            continue
        nx, ny = dx / dist, dy / dist
        ghost = (ball_pos[0] - 2 * R * nx, ball_pos[1] - 2 * R * ny)
        if not _ghost_on_table(ghost):
            continue
        if _segment_blocked(cue_t, ghost, ball_map, exclude_ids=[ball_id]):
            continue

        rail_target = _project_to_nearest_rail(ball_pos, (nx, ny))
        nearest_pocket_idx = min(
            range(len(POCKETS)),
            key=lambda i: math.hypot(POCKETS[i][0] - ball_pos[0],
                                      POCKETS[i][1] - ball_pos[1]))
        shots.append(LegalShot(
            ball_id=ball_id,
            pocket_idx=nearest_pocket_idx,
            aim_point=rail_target,
            ghost_pos=ghost,
            aim_angle=math.atan2(ghost[1] - cue_t[1], ghost[0] - cue_t[0]),
            cut_angle_deg=0.0,
            cue_to_ghost_dist=math.hypot(ghost[0] - cue_t[0], ghost[1] - cue_t[1]),
            ball_to_pocket_dist=math.hypot(rail_target[0] - ball_pos[0],
                                            rail_target[1] - ball_pos[1]),
            is_defensive=True,
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
    print(f'{len(shots)} direct shots from cue={cue} with {len(balls)} balls:')
    for s in sorted(shots, key=lambda s: s.difficulty)[:5]:
        print(f'  ball {s.ball_id} → {POCKET_NAMES[s.pocket_idx]:6s} '
              f'(cut={s.cut_angle_deg:5.1f}°  diff={s.difficulty:5.2f})')

    # Bank shot test
    shots_with_banks = generate_legal_shots(cue, balls, include_banks=True)
    banks = [s for s in shots_with_banks if s.is_bank]
    print(f'\n{len(banks)} bank shots found:')
    for s in sorted(banks, key=lambda s: s.difficulty)[:10]:
        print(f'  ball {s.ball_id} → {POCKET_NAMES[s.pocket_idx]:6s} '
              f'rail={s.rail_idx} refl=({s.reflection_point[0]:.1f},{s.reflection_point[1]:.1f}) '
              f'(cut={s.cut_angle_deg:5.1f}°  diff={s.difficulty:5.2f})')
    print(f'\nTotal: {len(shots_with_banks)} shots ({len(shots)} direct + {len(banks)} bank)')
