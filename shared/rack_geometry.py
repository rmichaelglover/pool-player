"""Rack geometry constants shared by 14.1 and 8-ball games."""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_geometry import TABLE_LENGTH, TABLE_WIDTH, BALL_R as R

RACK_APEX = (75.0, 25.0)

_ROW_DX = 2 * R * (math.sqrt(3) / 2) + 0.02
_BALL_DY = 2 * R + 0.02
_RACK_OFFSETS = []
for row in range(5):
    for col in range(row + 1):
        dx = row * _ROW_DX
        dy = (col - row / 2.0) * _BALL_DY
        _RACK_OFFSETS.append((dx, dy))
RACK_POSITIONS = [(RACK_APEX[0] + dx, RACK_APEX[1] + dy) for dx, dy in _RACK_OFFSETS]
assert len(RACK_POSITIONS) == 15


def sample_phase6_setup():
    """Place the 15-ball rack (fixed positions) and the cue ball (random in
    the head kitchen, x in [10, 25])."""
    balls = {i + 1: list(pos) for i, pos in enumerate(RACK_POSITIONS)}
    for _ in range(40):
        cx = 10.0 + random.random() * 15.0
        cy = 5.0 + random.random() * 40.0
        if 4 * R < cx < TABLE_LENGTH - 4 * R and 4 * R < cy < TABLE_WIDTH - 4 * R:
            return [cx, cy], balls
    return [15.0, 25.0], balls
