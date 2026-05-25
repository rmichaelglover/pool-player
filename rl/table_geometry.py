"""
Slate-only table geometry (the layout we settled on in the demo).

Playing surface:    100 × 50 inches
Slate:              107 × 57 (3.5″ overhang per side as cushion mounting surface)
Cushion depth:      2 inches
Corner pocket mouth: 4.5″ between cushion endpoints (chord)
Side pocket mouth:   5.0″ between cushion endpoints

Cushion's pocket-side edges:
  - Corner: slanted at 45° (parallel to the corner bisector). The cushion
    fabric is removed in a 6-vertex polygon at each corner: cushion-line
    endpoint → cushion-back endpoint (along the slant) → cushion-back
    inner corner → cushion-back endpoint (other rail) → cushion-line
    endpoint (other rail) → playing-area corner.
  - Side:   slanted at 12° from the perpendicular to the rail, converging
    inward. Cushion-back mouth is therefore narrower than cushion-line
    mouth (5″ → 4.15″ over 2″ depth).

Drop pockets (slate cutouts):
  - Corner: circle centered at the slate corner (e.g. (−3.5, −3.5) for
    TL); radius set by the constraint that the circle passes through the
    50% midpoint of each cushion slant (so r ≈ 6.21″).
  - Side:   circle centered 12 5/8″ outside the cushion line on the
    side-pocket bisector (perpendicular to the rail at TL/2); radius
    12 3/8″.

Coordinate system: x ∈ [0, TABLE_LENGTH], y ∈ [0, TABLE_WIDTH] for the
playing surface. y increases "downward" in screen orientation. Cushion
fabric occupies y ∈ (−CUSH_DEPTH, 0) for the top rail, etc.
"""
from __future__ import annotations
import math

# ── Playing surface ───────────────────────────────────────────────────────
TABLE_LENGTH = 100.0
TABLE_WIDTH  = 50.0
BALL_R       = 1.125

# ── Slate (mounting surface for cushions) ─────────────────────────────────
SLATE_OVERHANG = 3.5
SLATE_LENGTH   = TABLE_LENGTH + 2 * SLATE_OVERHANG   # 107
SLATE_WIDTH    = TABLE_WIDTH  + 2 * SLATE_OVERHANG   # 57

# ── Cushion ───────────────────────────────────────────────────────────────
CUSH_DEPTH = 2.0   # cushion fabric thickness perpendicular to the rail

# ── Corner pocket ─────────────────────────────────────────────────────────
CORNER_MOUTH       = 4.5
CORNER_RAIL_OFFSET = CORNER_MOUTH / math.sqrt(2.0)   # 3.1820 — distance from
                                                      # playing-area corner
                                                      # to cushion endpoint
                                                      # along each rail.

# Corner cushion slant: 45° (parallel to bisector). Cushion-line endpoint
# at (CORNER_RAIL_OFFSET, 0), cushion-back endpoint at
# (CORNER_RAIL_OFFSET − CUSH_DEPTH, −CUSH_DEPTH) for TL top rail.
# Slant midpoint (TL top, in playing-area coords):
#   x = CORNER_RAIL_OFFSET − CUSH_DEPTH/2  ≈ 2.1820
#   y = −CUSH_DEPTH/2                       = −1.0
# Slate corner (TL, in playing-area coords): (−SLATE_OVERHANG, −SLATE_OVERHANG).
# Distance from slate corner to slant midpoint:
#   dx = (CORNER_RAIL_OFFSET − CUSH_DEPTH/2) − (−SLATE_OVERHANG)
#   dy = (−CUSH_DEPTH/2) − (−SLATE_OVERHANG) = SLATE_OVERHANG − CUSH_DEPTH/2
_corner_pocket_dx = (CORNER_RAIL_OFFSET - CUSH_DEPTH / 2.0) + SLATE_OVERHANG
_corner_pocket_dy = SLATE_OVERHANG - CUSH_DEPTH / 2.0

# Corner drop-pocket circle: center at the slate corner, radius = distance
# from slate corner to slant midpoint.
CORNER_POCKET_R = math.hypot(_corner_pocket_dx, _corner_pocket_dy)   # ≈ 6.2086

# ── Side pocket ───────────────────────────────────────────────────────────
SIDE_MOUTH      = 5.0
SIDE_HALF       = SIDE_MOUTH / 2.0                                # 2.5
SIDE_SLANT_DEG  = 12.0                                             # 12° from perpendicular
SIDE_SHIFT      = CUSH_DEPTH * math.tan(math.radians(SIDE_SLANT_DEG))   # 0.4253

# Side drop-pocket circle: center 12 5/8″ outside the cushion line along
# the bisector, radius 12 3/8″.
SIDE_POCKET_OFFSET = 12 + 5.0/8.0   # 12.625
SIDE_POCKET_R      = 12 + 3.0/8.0   # 12.375

# ── Pocket aim points (where the AI aims the OB) ──────────────────────────
# Corner: midpoint of the throat chord. Side: midpoint of the cushion-line
# mouth.
POCKETS = [
    (CORNER_RAIL_OFFSET / 2.0, CORNER_RAIL_OFFSET / 2.0),                       # 0: TL
    (TABLE_LENGTH / 2.0, 0.0),                                                   # 1: T-side
    (TABLE_LENGTH - CORNER_RAIL_OFFSET / 2.0, CORNER_RAIL_OFFSET / 2.0),         # 2: TR
    (CORNER_RAIL_OFFSET / 2.0, TABLE_WIDTH - CORNER_RAIL_OFFSET / 2.0),          # 3: BL
    (TABLE_LENGTH / 2.0, TABLE_WIDTH),                                           # 4: B-side
    (TABLE_LENGTH - CORNER_RAIL_OFFSET / 2.0,
     TABLE_WIDTH - CORNER_RAIL_OFFSET / 2.0),                                    # 5: BR
]
POCKET_NAMES = ['TL', 'T-side', 'TR', 'BL', 'B-side', 'BR']

# Backward-compat: PHASE 7+ training code expects per-pocket radii ("corner
# vs side" was the only use). Keep the data structure with sentinel values;
# new code should use POCKET_NAMES or CORNER/SIDE constants.
POCKET_RADII = [2.5, 2.75, 2.5, 2.5, 2.75, 2.5]

# ── Pocket capture circles (drop pockets — slate cutouts) ─────────────────
# Each: (cx, cy, r). A ball center is "in the pocket" when it's inside this
# circle AND past the appropriate mouth/cushion line (see pocket_captures).
POCKET_CIRCLES = [
    (-SLATE_OVERHANG,           -SLATE_OVERHANG,           CORNER_POCKET_R),  # TL
    (TABLE_LENGTH / 2.0,        -SIDE_POCKET_OFFSET,       SIDE_POCKET_R),    # T-side
    (TABLE_LENGTH + SLATE_OVERHANG, -SLATE_OVERHANG,       CORNER_POCKET_R),  # TR
    (-SLATE_OVERHANG,           TABLE_WIDTH + SLATE_OVERHANG, CORNER_POCKET_R), # BL
    (TABLE_LENGTH / 2.0,        TABLE_WIDTH + SIDE_POCKET_OFFSET, SIDE_POCKET_R), # B-side
    (TABLE_LENGTH + SLATE_OVERHANG, TABLE_WIDTH + SLATE_OVERHANG, CORNER_POCKET_R), # BR
]


# Per-pocket facing pairs — indices into FACINGS whose cushion-back endpoints
# define the "back throat" chord. A ball must pass through this chord (with
# BALL_R clearance from each end) to reach the drop pocket without its center
# clipping a cushion facing.
_POCKET_FACING_PAIRS = (
    (0, 1),    # 0=TL: TL-top, TL-left
    (8, 9),    # 1=T-side: T-side L, T-side R
    (2, 3),    # 2=TR: TR-top, TR-right
    (4, 5),    # 3=BL: BL-bot, BL-left
    (10, 11),  # 4=B-side: B-side L, B-side R
    (6, 7),    # 5=BR: BR-bot, BR-right
)


# Side-pocket indices. The corridor narrows from cushion-line to cushion-back
# (facings converge at 12°), so the cushion-back facing endpoints add a
# binding clearance constraint. For corner pockets the facings are parallel,
# the corridor is constant-width, the sim excludes facing endpoints from
# bounces (see pool_sim.c), and the drop circle reaches into the playing
# area enough that the ball is captured before any facing clip — so only
# the cushion-line endpoints (where the cushion *segment* really does
# bounce) need clearance.
_SIDE_POCKETS = (1, 4)


def optimal_pocket_aim(ball_pos, idx: int, n_samples: int = 64):
    """Geometrically optimal aim point for a ball at `ball_pos` targeting
    pocket `idx`. Returns (px, py) on the cushion-back chord such that the
    ball's straight-line trajectory clears the binding pocket-corner
    endpoints with at least BALL_R perpendicular clearance, or None if no
    feasible direct aim exists.

    Binding endpoints (matched to the simulator's actual collision rules):
      - Side pockets: all four corridor corners (cushion-line endpoints +
        cushion-back endpoints). The converging facings narrow the corridor.
      - Corner pockets: cushion-line endpoints only. Cushion-back endpoints
        are facing endpoints that the simulator skips for bounces, and the
        corner drop circle captures the ball before it reaches them anyway.

    Implementation: sweep s along the cushion-back chord (full [0, 1] for
    corners; inset by BALL_R/L for sides where the chord clearance also
    binds), pick the s that maximizes the minimum perpendicular distance.
    """
    bx, by = ball_pos
    fa_idx, fb_idx = _POCKET_FACING_PAIRS[idx]
    fa = FACINGS[fa_idx]; fb = FACINGS[fb_idx]
    Ax, Ay = fa[2], fa[3]      # cushion-back endpoint of facing A
    Cx, Cy = fb[2], fb[3]      # cushion-back endpoint of facing C
    AC_dx = Cx - Ax; AC_dy = Cy - Ay
    L_back = math.hypot(AC_dx, AC_dy)
    if L_back < 1e-9:
        return None

    if idx in _SIDE_POCKETS:
        endpoints = ((fa[0], fa[1]), (fb[0], fb[1]), (Ax, Ay), (Cx, Cy))
        s_min = BALL_R / L_back
        s_max = 1.0 - s_min
    else:
        endpoints = ((fa[0], fa[1]), (fb[0], fb[1]))
        s_min = 0.0
        s_max = 1.0
    if s_min >= s_max:
        return None

    best_clearance = -1.0
    best_q = None
    for k in range(n_samples + 1):
        s = s_min + (s_max - s_min) * (k / n_samples)
        qx = Ax + s * AC_dx
        qy = Ay + s * AC_dy
        dx = qx - bx; dy = qy - by
        d_len = math.hypot(dx, dy)
        if d_len < 1e-9:
            continue
        # On-table ghost constraint: the cue ball center at the moment of
        # contact must be ≥ BALL_R from every cushion (otherwise the cue
        # ball would be embedded in the rail). For balls hugging a rail,
        # the throat-clearance-optimal aim may put the ghost just off the
        # playing area — those aims are physically infeasible regardless
        # of how much throat clearance they have. Reject early.
        ghost_x = bx - 2.0 * BALL_R * dx / d_len
        ghost_y = by - 2.0 * BALL_R * dy / d_len
        if not (BALL_R <= ghost_x <= TABLE_LENGTH - BALL_R and
                BALL_R <= ghost_y <= TABLE_WIDTH - BALL_R):
            continue
        clearance = float('inf')
        for ex, ey in endpoints:
            dist = abs(dy * (ex - bx) - dx * (ey - by)) / d_len
            if dist < clearance:
                clearance = dist
                if clearance < BALL_R:
                    break    # not feasible — early-exit
        if clearance >= BALL_R and clearance > best_clearance:
            best_clearance = clearance
            best_q = (qx, qy)
    return best_q


def can_pocket_directly(ball_pos, idx: int) -> bool:
    """Convenience wrapper: True iff a feasible direct aim exists."""
    return optimal_pocket_aim(ball_pos, idx) is not None


def pocket_captures(x: float, y: float, idx: int) -> bool:
    """True if a ball center at (x, y) is inside pocket `idx`'s capture
    region. The region is the pocket circle clipped by the appropriate
    mouth-side constraint (so the corner circles' overhang into the
    playing area only counts once a ball is past the mouth chord)."""
    cx, cy, r = POCKET_CIRCLES[idx]
    if (x - cx) * (x - cx) + (y - cy) * (y - cy) >= r * r:
        return False
    name = POCKET_NAMES[idx]
    if name == 'TL':
        return x + y < CORNER_RAIL_OFFSET
    if name == 'TR':
        return (TABLE_LENGTH - x) + y < CORNER_RAIL_OFFSET
    if name == 'BL':
        return x + (TABLE_WIDTH - y) < CORNER_RAIL_OFFSET
    if name == 'BR':
        return (TABLE_LENGTH - x) + (TABLE_WIDTH - y) < CORNER_RAIL_OFFSET
    if name == 'T-side':
        return y < 0.0 and (TABLE_LENGTH/2.0 - SIDE_HALF) < x < (TABLE_LENGTH/2.0 + SIDE_HALF)
    if name == 'B-side':
        return y > TABLE_WIDTH and (TABLE_LENGTH/2.0 - SIDE_HALF) < x < (TABLE_LENGTH/2.0 + SIDE_HALF)
    return False


# ── Cushion segments (axis-aligned, broken by pocket gaps) ────────────────
# Each: (x1, y1, x2, y2, nx, ny). (nx, ny) is the inward bounce normal —
# the direction the ball is pushed when it bounces off the segment.
_TL = TABLE_LENGTH
_TW = TABLE_WIDTH
_off = CORNER_RAIL_OFFSET
_sh = SIDE_HALF

CUSHIONS = [
    # Top rail (y=0): split by side pocket. Inward normal +y (into cloth).
    (_off,         0.0, _TL/2.0 - _sh, 0.0,  0.0, +1.0),
    (_TL/2.0 + _sh, 0.0, _TL - _off,    0.0,  0.0, +1.0),
    # Bottom rail (y=TW): split by side pocket. Inward normal -y.
    (_off,         _TW, _TL/2.0 - _sh, _TW,  0.0, -1.0),
    (_TL/2.0 + _sh, _TW, _TL - _off,    _TW,  0.0, -1.0),
    # Left rail (x=0): single segment between corner pockets. Inward +x.
    (0.0, _off, 0.0, _TW - _off,  +1.0, 0.0),
    # Right rail (x=TL): single segment. Inward -x.
    (_TL, _off, _TL, _TW - _off,  -1.0, 0.0),
]


# ── Facing segments (slanted cushion ends at each pocket) ─────────────────
# 8 corner facings (2 per corner, all at 45° = parallel to corner bisector)
# +
# 4 side facings (2 per side pocket, at 12° from perpendicular).
def _build_facings():
    facings = []
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # Corner facings — for each corner, two facings at 45° from the rails,
    # converging at the cushion-back inner corner.
    # TL top:    (off, 0)  → (off−CUSH, −CUSH).  Bounce normal toward +y/+playing.
    facings.append((_off, 0.0,
                     _off - CUSH_DEPTH, -CUSH_DEPTH,
                     -inv_sqrt2, +inv_sqrt2))
    # TL left:   (0, off) → (−CUSH, off−CUSH).
    facings.append((0.0, _off,
                     -CUSH_DEPTH, _off - CUSH_DEPTH,
                     +inv_sqrt2, -inv_sqrt2))
    # TR top:    (TL−off, 0) → (TL−off+CUSH, −CUSH).
    facings.append((_TL - _off, 0.0,
                     _TL - _off + CUSH_DEPTH, -CUSH_DEPTH,
                     +inv_sqrt2, +inv_sqrt2))
    # TR right:  (TL, off) → (TL+CUSH, off−CUSH).
    facings.append((_TL, _off,
                     _TL + CUSH_DEPTH, _off - CUSH_DEPTH,
                     -inv_sqrt2, -inv_sqrt2))
    # BL bottom: (off, TW) → (off−CUSH, TW+CUSH).
    facings.append((_off, _TW,
                     _off - CUSH_DEPTH, _TW + CUSH_DEPTH,
                     -inv_sqrt2, -inv_sqrt2))
    # BL left:   (0, TW−off) → (−CUSH, TW−off+CUSH).
    facings.append((0.0, _TW - _off,
                     -CUSH_DEPTH, _TW - _off + CUSH_DEPTH,
                     +inv_sqrt2, +inv_sqrt2))
    # BR bottom: (TL−off, TW) → (TL−off+CUSH, TW+CUSH).
    facings.append((_TL - _off, _TW,
                     _TL - _off + CUSH_DEPTH, _TW + CUSH_DEPTH,
                     +inv_sqrt2, -inv_sqrt2))
    # BR right:  (TL, TW−off) → (TL+CUSH, TW−off+CUSH).
    facings.append((_TL, _TW - _off,
                     _TL + CUSH_DEPTH, _TW - _off + CUSH_DEPTH,
                     -inv_sqrt2, +inv_sqrt2))

    # Side facings — 12° slant. Direction (±SIDE_SHIFT, ±CUSH_DEPTH),
    # length = sqrt(SIDE_SHIFT² + CUSH_DEPTH²). The inward bounce normal
    # is perpendicular to the segment and has a component pointing toward
    # the playing surface (+y for top, -y for bottom).
    side_len = math.hypot(SIDE_SHIFT, CUSH_DEPTH)
    side_nx = CUSH_DEPTH / side_len     # 0.978
    side_ny = SIDE_SHIFT / side_len     # 0.208

    # Top side, left-of-mouth: (TL/2−SH, 0) → (TL/2−SH+SIDE_SHIFT, −CUSH).
    # Bounce normal points +x (toward pocket center) and +y (into cloth).
    facings.append((_TL/2.0 - _sh, 0.0,
                     _TL/2.0 - _sh + SIDE_SHIFT, -CUSH_DEPTH,
                     +side_nx, +side_ny))
    # Top side, right-of-mouth: (TL/2+SH, 0) → (TL/2+SH−SIDE_SHIFT, −CUSH).
    # Bounce normal points -x and +y.
    facings.append((_TL/2.0 + _sh, 0.0,
                     _TL/2.0 + _sh - SIDE_SHIFT, -CUSH_DEPTH,
                     -side_nx, +side_ny))
    # Bottom side, left:  bounce normal +x, -y.
    facings.append((_TL/2.0 - _sh, _TW,
                     _TL/2.0 - _sh + SIDE_SHIFT, _TW + CUSH_DEPTH,
                     +side_nx, -side_ny))
    # Bottom side, right: bounce normal -x, -y.
    facings.append((_TL/2.0 + _sh, _TW,
                     _TL/2.0 + _sh - SIDE_SHIFT, _TW + CUSH_DEPTH,
                     -side_nx, -side_ny))
    return facings

FACINGS = _build_facings()


# ── Visual helpers (cushion gap polygons for the renderer) ────────────────
# Each pocket has a "cushion gap polygon" — the region where cushion fabric
# is removed at that pocket. The renderer fills the cushion band, then
# punches these polygons out of it.

def _corner_gap_polygon(idx):
    """6-vertex polygon for a corner cushion gap. idx in 0=TL, 2=TR, 3=BL, 5=BR."""
    if idx == 0:    # TL
        return [(_off, 0.0),
                (_off - CUSH_DEPTH, -CUSH_DEPTH),
                (-CUSH_DEPTH, -CUSH_DEPTH),
                (-CUSH_DEPTH, _off - CUSH_DEPTH),
                (0.0, _off),
                (0.0, 0.0)]
    if idx == 2:    # TR
        return [(_TL - _off, 0.0),
                (_TL - _off + CUSH_DEPTH, -CUSH_DEPTH),
                (_TL + CUSH_DEPTH, -CUSH_DEPTH),
                (_TL + CUSH_DEPTH, _off - CUSH_DEPTH),
                (_TL, _off),
                (_TL, 0.0)]
    if idx == 3:    # BL
        return [(_off, _TW),
                (_off - CUSH_DEPTH, _TW + CUSH_DEPTH),
                (-CUSH_DEPTH, _TW + CUSH_DEPTH),
                (-CUSH_DEPTH, _TW - _off + CUSH_DEPTH),
                (0.0, _TW - _off),
                (0.0, _TW)]
    if idx == 5:    # BR
        return [(_TL - _off, _TW),
                (_TL - _off + CUSH_DEPTH, _TW + CUSH_DEPTH),
                (_TL + CUSH_DEPTH, _TW + CUSH_DEPTH),
                (_TL + CUSH_DEPTH, _TW - _off + CUSH_DEPTH),
                (_TL, _TW - _off),
                (_TL, _TW)]
    raise IndexError(idx)


def _side_gap_polygon(idx):
    """4-vertex trapezoid for a side cushion gap. idx in 1=T-side, 4=B-side."""
    if idx == 1:    # T-side
        return [(_TL/2.0 - _sh, 0.0),
                (_TL/2.0 + _sh, 0.0),
                (_TL/2.0 + _sh - SIDE_SHIFT, -CUSH_DEPTH),
                (_TL/2.0 - _sh + SIDE_SHIFT, -CUSH_DEPTH)]
    if idx == 4:    # B-side
        return [(_TL/2.0 - _sh, _TW),
                (_TL/2.0 + _sh, _TW),
                (_TL/2.0 + _sh - SIDE_SHIFT, _TW + CUSH_DEPTH),
                (_TL/2.0 - _sh + SIDE_SHIFT, _TW + CUSH_DEPTH)]
    raise IndexError(idx)


CUSHION_GAP_POLYGONS = [
    _corner_gap_polygon(0),    # TL
    _side_gap_polygon(1),      # T-side
    _corner_gap_polygon(2),    # TR
    _corner_gap_polygon(3),    # BL
    _side_gap_polygon(4),      # B-side
    _corner_gap_polygon(5),    # BR
]


def to_dict():
    """Serialize all geometry to a JSON-friendly dict for the renderer."""
    return {
        'table_length': TABLE_LENGTH,
        'table_width':  TABLE_WIDTH,
        'ball_r':       BALL_R,
        'slate_overhang': SLATE_OVERHANG,
        'slate_length':   SLATE_LENGTH,
        'slate_width':    SLATE_WIDTH,
        'cush_depth':     CUSH_DEPTH,
        'corner_mouth':       CORNER_MOUTH,
        'corner_rail_offset': CORNER_RAIL_OFFSET,
        'corner_pocket_r':    CORNER_POCKET_R,
        'side_mouth':       SIDE_MOUTH,
        'side_half':        SIDE_HALF,
        'side_slant_deg':   SIDE_SLANT_DEG,
        'side_shift':       SIDE_SHIFT,
        'side_pocket_offset': SIDE_POCKET_OFFSET,
        'side_pocket_r':      SIDE_POCKET_R,
        'pockets':       POCKETS,
        'pocket_names':  POCKET_NAMES,
        'pocket_circles': POCKET_CIRCLES,
        'cushions': CUSHIONS,
        'facings':  FACINGS,
        'cushion_gap_polygons': CUSHION_GAP_POLYGONS,
    }


if __name__ == '__main__':
    d = to_dict()
    print('Geometry summary:')
    print(f'  Playing surface: {d["table_length"]} × {d["table_width"]} in')
    print(f'  Slate:           {d["slate_length"]} × {d["slate_width"]} in')
    print(f'  Cushion depth:   {d["cush_depth"]}')
    print(f'  Corner mouth:    {d["corner_mouth"]}  (rail offset {d["corner_rail_offset"]:.4f})')
    print(f'  Corner pocket r: {d["corner_pocket_r"]:.4f}')
    print(f'  Side mouth:      {d["side_mouth"]}')
    print(f'  Side slant:      {d["side_slant_deg"]}° (shift {d["side_shift"]:.4f})')
    print(f'  Side pocket:     center at {d["side_pocket_offset"]} outside, r={d["side_pocket_r"]}')
    print(f'  {len(d["cushions"])} cushion segments')
    print(f'  {len(d["facings"])}  facing segments')
    print(f'  Pocket aim points:')
    for n, p in zip(d['pocket_names'], d['pockets']):
        print(f'    {n:8s} ({p[0]:6.3f}, {p[1]:6.3f})')
