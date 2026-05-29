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
                              _POCKET_FACING_PAIRS)

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
    is_kick: bool = False         # tier-3: cue ball banks off a rail to reach target
    is_combo: bool = False        # cue→ball_id→combo_second_id→pocket
    combo_second_id: int = -1     # the SECOND ball (the one pocketed) in a combo

    @property
    def difficulty(self) -> float:
        cut_factor = 1.0 / max(0.1, math.cos(math.radians(self.cut_angle_deg)))
        total_dist = self.cue_to_ghost_dist + self.ball_to_pocket_dist
        base = cut_factor * (total_dist / 20.0)
        mult = 1.8 if self.is_bank else (2.0 if self.is_combo else 1.0)
        return base * mult


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
    # Entry gate = front mouth jaws only, for both pocket types. The ball is
    # captured at the mouth and never reaches the rear facings, so they must not
    # constrain entry (matches optimal_pocket_aim / pocket_aim_candidates).
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


def generate_combination_shots(cue_pos, balls, target_ids, max_cut_deg=65.0,
                               min_pocket_dist=1.0, max_combos=12):
    """Enumerate 2-ball combination shots: the cue strikes ball A (first
    contact), driving ball B into a pocket — A→B→pocket.

    Both A and B are drawn from target_ids (the player's own group), so the
    first contact is always legal and the pocketed ball is always ours. Mirrors
    the direct-shot geometry, two-staged:
        ghostB = where A's CENTER must be at contact to send B → pocket
        ghostA = where the CUE's center must be at contact to send A → ghostB
    A shot is kept only if all three corridors are clear (cue→ghostA,
    A→ghostB, B→pocket), both cut angles are within max_cut_deg, both ghosts
    are on the table, and the cue is on the strikeable side of A.

    This only puts combos in the ACTION SPACE — it does not score or prefer
    them. Value-search and distillation decide when a combo beats the
    alternatives, so good (and possibly novel) combos can emerge on their own.
    """
    shots = []
    cue_t = tuple(cue_pos)
    ball_map = {bid: tuple(pos) for bid, pos in balls.items()}
    tgt = [b for b in target_ids if b in ball_map]

    for b_id in tgt:                      # B = the ball that drops
        posB = ball_map[b_id]
        for p_idx, pocket_pos in enumerate(POCKETS):
            if math.hypot(pocket_pos[0] - posB[0],
                          pocket_pos[1] - posB[1]) < min_pocket_dist:
                continue
            if _segment_blocked(posB, pocket_pos, ball_map, exclude_ids=[b_id]):
                continue
            ghostB = ghost_ball(posB, pocket_pos)   # A's center at B-contact
            if not _ghost_on_table(ghostB):
                continue
            for a_id in tgt:              # A = first ball the cue strikes
                if a_id == b_id:
                    continue
                posA = ball_map[a_id]
                cutB = cut_angle_deg(posA, ghostB, posB, pocket_pos)
                if cutB > max_cut_deg:
                    continue
                ghostA = ghost_ball(posA, ghostB)   # cue center at A-contact
                if not _ghost_on_table(ghostA):
                    continue
                # Cue must be on the side of A that drives it toward ghostB.
                bpx, bpy = posA[0] - ghostA[0], posA[1] - ghostA[1]
                cgx, cgy = ghostA[0] - cue_t[0], ghostA[1] - cue_t[1]
                if bpx * cgx + bpy * cgy <= 0:
                    continue
                cutA = cut_angle_deg(cue_t, ghostA, posA, ghostB)
                if cutA > max_cut_deg:
                    continue
                # Corridors: cue→ghostA (B may block), A→ghostB, B→pocket(done).
                if _segment_blocked(cue_t, ghostA, ball_map, exclude_ids=[a_id]):
                    continue
                if _segment_blocked(posA, ghostB, ball_map,
                                    exclude_ids=[a_id, b_id]):
                    continue
                aim = math.atan2(ghostA[1] - cue_t[1], ghostA[0] - cue_t[0])
                shots.append(LegalShot(
                    ball_id=a_id,                   # first contact → legality
                    pocket_idx=p_idx,               # where B drops
                    aim_point=ghostB,               # cue sends A toward ghostB
                    ghost_pos=ghostA,
                    aim_angle=aim,
                    cut_angle_deg=max(cutA, cutB),
                    cue_to_ghost_dist=math.hypot(cgx, cgy),
                    ball_to_pocket_dist=(
                        math.hypot(ghostB[0] - posA[0], ghostB[1] - posA[1])
                        + math.hypot(pocket_pos[0] - posB[0],
                                     pocket_pos[1] - posB[1])),
                    is_combo=True,
                    combo_second_id=b_id,
                ))
    shots.sort(key=lambda s: s.difficulty)
    return shots[:max_combos]


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


def generate_kick_shots(cue_pos, balls, target_ids, min_target_dist=4 * R,
                         skip_blocking=False, first_contact_only=False):
    """Tier-3 shots: cue ball banks off one rail to make legal contact with
    each reachable target ball. Used when even tier-2 (direct defensive)
    is blocked. Mirror the target across each rail, find the reflection
    point where cue→mirror crosses the rail, and emit one shot per
    (target, rail) pair.

    A kick is a FOUL-AVOIDANCE shot, not a pocket attempt: when snookered the
    goal is to contact your own ball first (then drive a ball to a rail), not
    to pocket. So there are three line-of-sight tiers, tried in order:

      default (full clearance): cue→reflection and reflection→target both clear
          of every other ball. The clean, pocketable kick.
      first_contact_only=True: the cue must reach the first rail WITHOUT
          striking an illegal-first-contact ball (any ball not in target_ids —
          opponents, and the 8 when not on it). Own-group balls in the path are
          fine (still legal contact), and the post-rail path is NOT required to
          be clear. This is the foul-avoidance kick: legal contact is the win.
      skip_blocking=True: last resort — no line-of-sight checks at all, so the
          engine always has an action even in genuinely snookered positions
          where every kick fouls on an opponent before the rail.
    """
    shots: list[LegalShot] = []
    cue_t = tuple(cue_pos)
    ball_map = {bid: tuple(pos) for bid, pos in balls.items()}

    for ball_id in target_ids:
        if ball_id not in ball_map:
            continue
        ball_pos = ball_map[ball_id]
        for c_idx, cushion in enumerate(CUSHIONS):
            mirror_target = _mirror_pocket(ball_pos, cushion)
            refl = _bank_reflection_point(cue_t, mirror_target, cushion)
            if refl is None:
                continue
            if skip_blocking:
                pass  # last resort: emit even fouling kicks (see docstring)
            elif first_contact_only:
                # Foul-avoidance tier: the cue must reach the first rail without
                # striking an illegal-first-contact ball (any ball not in
                # target_ids). Own-group balls in the path are fine, and the
                # post-rail path is not required to be clear.
                illegal = {b: p for b, p in ball_map.items() if b not in target_ids}
                if _segment_blocked(cue_t, refl, illegal):
                    continue
            else:
                if _segment_blocked(cue_t, refl, ball_map, exclude_ids=[ball_id]):
                    continue
                if _segment_blocked(refl, ball_pos, ball_map, exclude_ids=[ball_id]):
                    continue
            dx = ball_pos[0] - refl[0]
            dy = ball_pos[1] - refl[1]
            d = math.hypot(dx, dy)
            if d < min_target_dist:
                continue
            nx, ny = dx / d, dy / d
            ghost = (ball_pos[0] - 2 * R * nx, ball_pos[1] - 2 * R * ny)
            nearest_pocket_idx = min(
                range(len(POCKETS)),
                key=lambda i: math.hypot(POCKETS[i][0] - ball_pos[0],
                                          POCKETS[i][1] - ball_pos[1]))
            shots.append(LegalShot(
                ball_id=ball_id,
                pocket_idx=nearest_pocket_idx,
                aim_point=mirror_target,
                ghost_pos=ghost,
                aim_angle=math.atan2(refl[1] - cue_t[1], refl[0] - cue_t[0]),
                cut_angle_deg=0.0,
                cue_to_ghost_dist=math.hypot(refl[0] - cue_t[0],
                                              refl[1] - cue_t[1]) + d,
                ball_to_pocket_dist=math.hypot(
                    POCKETS[nearest_pocket_idx][0] - ball_pos[0],
                    POCKETS[nearest_pocket_idx][1] - ball_pos[1]),
                is_defensive=True,
                is_kick=True,
                rail_idx=c_idx,
                reflection_point=refl,
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
