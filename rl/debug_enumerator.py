"""Debug script for the enumerator's false-negative cases.

Approximated positions from user-reported screenshots where the enumerator
incorrectly marked a shot as legal when a blocker should have prevented it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shot_enumerator import generate_legal_shots, _segment_blocked, POCKETS, R
import math


def test_case(name, cue, balls, expect_ball_id, expect_pocket_idx, should_be_legal):
    """Run enumerator on a state and check if a specific (ball, pocket)
    shot is in the legal list."""
    print(f'\n=== {name} ===')
    print(f'  cue: {cue}')
    print(f'  balls: {balls}')
    shots = generate_legal_shots(cue, balls)
    target = [(s.ball_id, s.pocket_idx) for s in shots]
    is_legal = (expect_ball_id, expect_pocket_idx) in target
    status = ('✓ correct' if is_legal == should_be_legal
              else f'✗ BUG: expected legal={should_be_legal}, got legal={is_legal}')
    print(f'  ball {expect_ball_id} → pocket {expect_pocket_idx}: '
          f'enumerator says legal={is_legal}; should be {should_be_legal} {status}')
    # If unexpectedly legal, show what blocker check returned
    if is_legal and not should_be_legal:
        ball_pos = balls[expect_ball_id]
        aim_point = POCKETS[expect_pocket_idx]
        print(f'    OB→pocket segment: from {ball_pos} to {aim_point}')
        # Manually check perpendicular distance for each other ball
        x1, y1 = ball_pos
        x2, y2 = aim_point
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        ux, uy = dx / seg_len, dy / seg_len
        clearance = 2 * R + 0.3
        for bid, (bx, by) in balls.items():
            if bid == expect_ball_id:
                continue
            ex, ey = bx - x1, by - y1
            t = ex * ux + ey * uy
            perp_x = ex - t * ux
            perp_y = ey - t * uy
            perp = math.hypot(perp_x, perp_y)
            in_seg = 0 <= t <= seg_len
            print(f'    ball {bid} at ({bx}, {by}): t={t:.2f} (seg_len={seg_len:.1f}), '
                  f'perp={perp:.3f}, in_seg={in_seg}, '
                  f'would_block={in_seg and perp < clearance}')


# Case 1: Ball 3 → BR with ball 15 between them (screenshot 06-30-33)
# Cue at (~33, 32), ball 3 at (~47, 48), ball 15 at (~78, 47), BR pocket idx 5
test_case(
    'Case 1: ball 3 → BR with ball 15 blocking',
    cue=(33, 32),
    balls={3: (47, 48), 15: (78, 47), 6: (54, 38), 10: (60, 47), 4: (32, 5), 12: (67, 5)},
    expect_ball_id=3, expect_pocket_idx=5,  # BR corner
    should_be_legal=False,  # ball 15 should block
)

# Case 2: Ball 6 → BL with balls 11, 3 between them (screenshot 06-31-42)
# Cue at (~85, 25), ball 6 at (~60, 35), ball 11 at (~30, 40), ball 3 at (~40, 47)
test_case(
    'Case 2: ball 6 → BL with ball 11 blocking',
    cue=(85, 25),
    balls={6: (60, 35), 11: (30, 40), 3: (40, 47), 1: (25, 45), 15: (45, 33)},
    expect_ball_id=6, expect_pocket_idx=3,  # BL corner
    should_be_legal=False,
)

# Case 3: Ball 4 → TR with ball 12 between them (screenshot 06-45-14)
test_case(
    'Case 3: ball 4 → TR with ball 12 blocking',
    cue=(24, 38),
    balls={4: (32, 5), 12: (67, 5), 15: (13, 17), 6: (24, 38), 7: (62, 22)},
    expect_ball_id=4, expect_pocket_idx=2,  # TR corner
    should_be_legal=False,
)
