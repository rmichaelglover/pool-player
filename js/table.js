// table.js -- Table geometry, pockets, cushions
class Table {
  constructor() {
    this.width = C.TABLE_WIDTH;
    this.length = C.TABLE_LENGTH;
    this.railWidth = C.RAIL_WIDTH;

    this._initPockets();
    this._initCushions();
  }

  _initPockets() {
    const pr = C.POCKET_RADIUS;
    const prs = C.POCKET_RADIUS_SIDE;
    const L = this.length;
    const W = this.width;
    // Each pocket has two positions:
    //   (x, y) = center of the pocket hole (for physics/pocketing detection)
    //   (aimX, aimY) = center of the pocket MOUTH (for AI aiming)
    //
    // The mouth is the opening between the two cushion noses. For corner pockets,
    // this is offset diagonally inward from the corner. For side pockets, it's
    // at the rail line. The AI aims at the mouth center so the ball enters cleanly
    // between the cushion noses instead of clipping them.
    const mouthOffset = pr * 0.45; // how far the mouth center is from the corner
    const sideMouthOffset = prs * 0.15; // side pockets: slight inward offset
    this.pockets = [
      { x: 0,     y: 0,     radius: pr,  name: 'top-left',
        aimX: mouthOffset, aimY: mouthOffset },
      { x: L / 2, y: 0,     radius: prs, name: 'top-side',
        aimX: L / 2, aimY: sideMouthOffset },
      { x: L,     y: 0,     radius: pr,  name: 'top-right',
        aimX: L - mouthOffset, aimY: mouthOffset },
      { x: 0,     y: W,     radius: pr,  name: 'bottom-left',
        aimX: mouthOffset, aimY: W - mouthOffset },
      { x: L / 2, y: W,     radius: prs, name: 'bottom-side',
        aimX: L / 2, aimY: W - sideMouthOffset },
      { x: L,     y: W,     radius: pr,  name: 'bottom-right',
        aimX: L - mouthOffset, aimY: W - mouthOffset },
    ];
  }

  _initCushions() {
    // Cushion segments as line segments along inner rail edges.
    // Each segment: { x1, y1, x2, y2, nx, ny } where (nx, ny) is inward normal.
    // Rails are split around pocket openings.
    const L = this.length;
    const W = this.width;
    // Cushion endpoints: the cushion "nose" where the rubber starts.
    // BCA spec: corner pocket mouth ~5", so each cushion nose is ~2.5" from the corner.
    // Side pocket mouth ~5.5", so each cushion nose is ~2.75" from the side pocket center.
    const po = C.POCKET_RADIUS;       // corner: cushion starts at pocket radius from corner
    const spo = C.POCKET_RADIUS_SIDE; // side: cushion starts at pocket radius from center

    this.cushions = [
      // Top rail (y = 0, normal points down +y) -- two segments around side pocket
      { x1: po,         y1: 0, x2: L / 2 - spo, y2: 0, nx: 0, ny: 1 },
      { x1: L / 2 + spo, y1: 0, x2: L - po,     y2: 0, nx: 0, ny: 1 },
      // Bottom rail (y = W, normal points up -y)
      { x1: po,         y1: W, x2: L / 2 - spo, y2: W, nx: 0, ny: -1 },
      { x1: L / 2 + spo, y1: W, x2: L - po,     y2: W, nx: 0, ny: -1 },
      // Left rail (x = 0, normal points right +x)
      { x1: 0, y1: po,     x2: 0, y2: W - po, nx: 1, ny: 0 },
      // Right rail (x = L, normal points left -x)
      { x1: L, y1: po,     x2: L, y2: W - po, nx: -1, ny: 0 },
    ];
  }

  isPocketed(ball) {
    for (const p of this.pockets) {
      const dx = ball.x - p.x;
      const dy = ball.y - p.y;
      if (dx * dx + dy * dy < p.radius * p.radius) {
        return true;
      }
    }
    return false;
  }

  // Get rack positions for 15 balls in standard 8-ball triangle
  getRackPositions() {
    const positions = [];
    const R = C.BALL_RADIUS;
    const fx = C.FOOT_SPOT_X;
    const fy = C.FOOT_SPOT_Y;
    // Slight compression so adjacent balls pre-overlap by a tiny amount (~0.005").
    // This ensures the collision resolver detects contact between racked balls.
    // The deferred-separation approach (impulses without separation during iterations,
    // separation only at the end) means even tiny overlaps propagate correctly.
    const compression = 0.998;
    const rowSpacing = R * Math.sqrt(3) * compression;
    const D = 2 * R * compression;

    // Build 5-row triangle
    // Row 0: 1 ball, Row 1: 2, Row 2: 3, Row 3: 4, Row 4: 5
    const rackOrder = [];
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col <= row; col++) {
        const x = fx + row * rowSpacing;
        const y = fy + (col - row / 2) * D;
        rackOrder.push({ x, y });
      }
    }

    // Standard 8-ball rack:
    // Position indices in the triangle (0-14):
    //   Row 0: [0]
    //   Row 1: [1, 2]
    //   Row 2: [3, 4, 5]
    //   Row 3: [6, 7, 8, 9]
    //   Row 4: [10, 11, 12, 13, 14]
    //
    // Rules: 8-ball at index 4 (center of row 2)
    //        One solid and one stripe at back corners (index 10 and 14)
    //        Remaining balls mixed

    // Ball IDs to place (1-15)
    // 8-ball goes at position 4
    // Position 0 (apex): ball 1
    // Corners of last row (10, 14): one solid, one stripe
    const ballIds = [1, 9, 10, 11, 8, 3, 14, 7, 2, 15, 6, 12, 5, 13, 4];

    for (let i = 0; i < 15; i++) {
      positions.push({
        id: ballIds[i],
        x: rackOrder[i].x,
        y: rackOrder[i].y,
      });
    }

    return positions;
  }

  // 9-ball diamond rack: balls 1-9, diamond shape with 1 at apex, 9 in center
  get9BallRackPositions() {
    const R = C.BALL_RADIUS;
    const fx = C.FOOT_SPOT_X;
    const fy = C.FOOT_SPOT_Y;
    const compression = 0.998;
    const rowSpacing = R * Math.sqrt(3) * compression;
    const D = 2 * R * compression;

    // Diamond shape: 1 row of 1, 2, 3, 2, 1
    // Row 0: 1 ball (apex = ball 1)
    // Row 1: 2 balls
    // Row 2: 3 balls (center = ball 9)
    // Row 3: 2 balls
    // Row 4: 1 ball
    const rackOrder = [];
    const rowSizes = [1, 2, 3, 2, 1];
    for (let row = 0; row < 5; row++) {
      const n = rowSizes[row];
      for (let col = 0; col < n; col++) {
        const x = fx + row * rowSpacing;
        const y = fy + (col - (n - 1) / 2) * D;
        rackOrder.push({ x, y });
      }
    }

    // Ball placement: 1 at apex, 9 in center (position 4 = middle of row 2)
    // Other balls 2-8 placed randomly in remaining positions
    const remaining = [2, 3, 4, 5, 6, 7, 8];
    // Shuffle remaining
    for (let i = remaining.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
    }

    const ballIds = new Array(9);
    ballIds[0] = 1;  // apex
    ballIds[4] = 9;  // center of diamond
    let ri = 0;
    for (let i = 0; i < 9; i++) {
      if (i === 0 || i === 4) continue;
      ballIds[i] = remaining[ri++];
    }

    const positions = [];
    for (let i = 0; i < 9; i++) {
      positions.push({ id: ballIds[i], x: rackOrder[i].x, y: rackOrder[i].y });
    }
    return positions;
  }

  // 14.1 rack: same triangle as 8-ball but with apex spot empty (for re-rack)
  // When re-racking after 14 balls pocketed, the 15th ball stays in place.
  get141RackPositions(excludeBallId) {
    const R = C.BALL_RADIUS;
    const fx = C.FOOT_SPOT_X;
    const fy = C.FOOT_SPOT_Y;
    const compression = 0.998;
    const rowSpacing = R * Math.sqrt(3) * compression;
    const D = 2 * R * compression;

    const rackOrder = [];
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col <= row; col++) {
        const x = fx + row * rowSpacing;
        const y = fy + (col - row / 2) * D;
        rackOrder.push({ x, y });
      }
    }

    // For initial rack: all 15 balls, randomized (no special placement rules)
    // For re-rack: 14 balls (excludeBallId stays on table), apex spot left empty
    const allIds = [];
    for (let i = 1; i <= 15; i++) {
      if (i === excludeBallId) continue;
      allIds.push(i);
    }
    // Shuffle
    for (let i = allIds.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allIds[i], allIds[j]] = [allIds[j], allIds[i]];
    }

    const positions = [];
    // For re-rack: leave apex (index 0) empty
    const startIdx = excludeBallId ? 1 : 0;
    let bi = 0;
    for (let i = startIdx; i < rackOrder.length && bi < allIds.length; i++) {
      positions.push({ id: allIds[bi], x: rackOrder[i].x, y: rackOrder[i].y });
      bi++;
    }
    return positions;
  }
}
