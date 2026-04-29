// renderer.js -- Canvas rendering with overhead and shooter's perspective views
class Renderer {
  constructor(canvas, spinCanvas, table) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.spinCanvas = spinCanvas;
    this.spinCtx = spinCanvas.getContext('2d');
    this.table = table;
    this.S = C.SCALE;
    this.offsetX = C.RAIL_WIDTH * this.S;
    this.offsetY = C.RAIL_WIDTH * this.S;

    canvas.width = (C.TABLE_LENGTH + 2 * C.RAIL_WIDTH) * this.S;
    canvas.height = (C.TABLE_WIDTH + 2 * C.RAIL_WIDTH) * this.S;

    // View mode: 'overhead' or 'shooter'
    this.viewMode = 'overhead';

    // Camera state for shooter view
    this.cam = { x: 0, y: 0, z: 30, lookX: 1, lookY: 0 };
  }

  setViewMode(mode) {
    this.viewMode = mode;
  }

  // -- Overhead coordinate helpers --
  tx(x) { return x * this.S + this.offsetX; }
  ty(y) { return y * this.S + this.offsetY; }

  // -- Perspective projection for shooter view --
  // Camera is behind the cue ball at height z, looking along (lookX, lookY).
  // Project a 3D point (wx, wy, wz) in table space to 2D screen coords.
  _project(wx, wy, wz) {
    const cam = this.cam;
    // Vector from camera to point
    const dx = wx - cam.x;
    const dy = wy - cam.y;
    const dz = wz - cam.z;

    // Camera basis vectors
    // Forward: (lookX, lookY, -0.35) normalized -- looking slightly down at table
    const lookDown = -0.4;
    const fLen = Math.sqrt(cam.lookX * cam.lookX + cam.lookY * cam.lookY + lookDown * lookDown);
    const fx = cam.lookX / fLen;
    const fy = cam.lookY / fLen;
    const fz = lookDown / fLen;

    // Right: cross(forward, up) where up = (0, 0, 1)
    let rx = fy * 1 - fz * 0;  // fy
    let ry = fz * 0 - fx * 1;  // -fx
    let rz = fx * 0 - fy * 0;  // 0
    const rLen = Math.sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rLen; ry /= rLen; rz /= rLen;

    // Up: cross(right, forward)
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    // Project onto camera space
    const depth = dx * fx + dy * fy + dz * fz;
    const screenX = dx * rx + dy * ry + dz * rz;
    const screenY = dx * ux + dy * uy + dz * uz;

    if (depth < 1) return null; // behind camera

    // Perspective divide
    const focalLength = 300;
    const canvasW = this.canvas.width;
    const canvasH = this.canvas.height;
    const px = canvasW / 2 + (screenX / depth) * focalLength;
    const py = canvasH / 2 - (screenY / depth) * focalLength;

    // Apparent size scaling
    const scale = focalLength / depth;

    return { x: px, y: py, depth, scale };
  }

  // -- Main render dispatch --
  render(balls, game, shotVector) {
    this._currentBalls = balls; // store for _findBallById
    if (this.viewMode === 'shooter') {
      this._updateCamera(balls, shotVector, game);
      this._renderShooterView(balls, game, shotVector);
    } else {
      this._renderOverheadView(balls, game, shotVector);
    }
  }

  _updateCamera(balls, shotVector, game) {
    const cueBall = balls.find(b => b.id === 0 && !b.isPocketed);
    if (!cueBall) return;

    let aimDX = 1, aimDY = 0;
    if (shotVector && shotVector.aimDX !== undefined) {
      aimDX = shotVector.aimDX;
      aimDY = shotVector.aimDY;
    }

    const isPreview = game && game.state === 'SHOT_PREVIEW';

    if (isPreview) {
      // Shot preview: position camera behind the cue ball looking down the aim line
      const dist = 25;
      this.cam.x = cueBall.x - aimDX * dist;
      this.cam.y = cueBall.y - aimDY * dist;
      this.cam.z = 18;
      this.cam.lookX = aimDX;
      this.cam.lookY = aimDY;
      // Lock this position -- the shooter stays down on the shot
      this._shotCam = {
        x: this.cam.x, y: this.cam.y, z: this.cam.z,
        lookX: aimDX, lookY: aimDY,
      };
    } else if (this._shotCam) {
      // During simulation and after: stay exactly where the shooter was.
      // A real player stays down on the shot and watches from the same position.
      this.cam.x = this._shotCam.x;
      this.cam.y = this._shotCam.y;
      this.cam.z = this._shotCam.z;
      this.cam.lookX = this._shotCam.lookX;
      this.cam.lookY = this._shotCam.lookY;
    }
  }

  // __________________________________________________________________
  // OVERHEAD VIEW (original)
  // __________________________________________________________________

  _renderOverheadView(balls, game, shotVector) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    try {
      this.drawTable();
    } catch (err) {
      console.error('drawTable error:', err);
      ctx.fillStyle = '#300';
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.fillStyle = '#f88';
      ctx.font = '14px Arial';
      ctx.fillText('drawTable error: ' + err.message, 10, 20);
      return;
    }
    for (const ball of balls) {
      if (!ball.isPocketed) {
        try { this.drawBall(ball); }
        catch (err) { console.error('drawBall error on ball ' + ball.id + ':', err); }
      }
    }
    if (shotVector && game.state === 'SHOT_PREVIEW') {
      this.drawShotVector(shotVector);
      this.drawSpinIndicator(shotVector);
      // Highlight current target ball with red ring
      if (shotVector.targetId !== undefined) {
        const targetBall = balls.find(b => b.id === shotVector.targetId);
        if (targetBall && !targetBall.isPocketed) {
          ctx.beginPath();
          ctx.arc(this.tx(targetBall.x), this.ty(targetBall.y),
                  targetBall.radius * this.S + 4, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255, 100, 100, 0.7)';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      // Draw next-shot plan
      this._drawNextShotPlan(shotVector, balls);
    }
    this.drawPocketedBalls(balls);
  }

  // __________________________________________________________________
  // SHOOTER'S PERSPECTIVE VIEW
  // __________________________________________________________________

  _renderShooterView(balls, game, shotVector) {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    // Draw table surface
    this._drawTable3D();

    // Collect all balls with their projected positions, sort by depth (far first)
    const projectedBalls = [];
    for (const ball of balls) {
      if (ball.isPocketed) continue;
      const p = this._project(ball.x, ball.y, ball.radius); // center at ball radius height
      if (p) {
        projectedBalls.push({ ball, p });
      }
    }
    projectedBalls.sort((a, b) => b.p.depth - a.p.depth);

    // Draw balls
    for (const { ball, p } of projectedBalls) {
      this._drawBall3D(ball, p);
    }

    // Draw shot visualization
    if (shotVector && game.state === 'SHOT_PREVIEW') {
      this._drawShotVector3D(shotVector);
      this.drawSpinIndicator(shotVector);

      // Next ball highlight in 3D
      if (shotVector.nextBallId !== null && shotVector.nextBallId !== undefined) {
        const nb = balls.find(b => b.id === shotVector.nextBallId && !b.isPocketed);
        if (nb) {
          const np = this._project(nb.x, nb.y, C.BALL_RADIUS);
          if (np) {
            const nr = Math.max(3, C.BALL_RADIUS * np.scale + 4);
            ctx.beginPath();
            ctx.arc(np.x, np.y, nr, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(80, 160, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
    }
  }

  // Project a point, clamping to canvas edges instead of returning null.
  // This prevents the felt/rails from disappearing when corners are behind the camera.
  _projectClamped(wx, wy, wz) {
    const p = this._project(wx, wy, wz);
    if (p) return p;
    // Point is behind the camera -- extrapolate to a far edge of the canvas.
    // Use the raw camera-space coordinates to determine which edge.
    const cam = this.cam;
    const dxc = wx - cam.x;
    const dyc = wy - cam.y;
    const dzc = wz - cam.z;
    const lookDown = -0.4;
    const fLen = Math.sqrt(cam.lookX * cam.lookX + cam.lookY * cam.lookY + lookDown * lookDown);
    const fx = cam.lookX / fLen, fy = cam.lookY / fLen, fz = lookDown / fLen;
    let rx = fy, ry = -fx, rz = 0;
    const rLen = Math.sqrt(rx * rx + ry * ry);
    rx /= rLen; ry /= rLen;
    const screenX = dxc * rx + dyc * ry + dzc * rz;
    const W = this.canvas.width;
    const H = this.canvas.height;
    // Push to the appropriate edge of the canvas
    const px = screenX > 0 ? W + 100 : -100;
    const py = H + 100; // behind = push to bottom
    return { x: px, y: py, depth: 0.5, scale: 1 };
  }

  _drawTable3D() {
    const ctx = this.ctx;
    const L = C.TABLE_LENGTH;
    const W = C.TABLE_WIDTH;
    const rw = C.RAIL_WIDTH;
    const canvasW = this.canvas.width;
    const canvasH = this.canvas.height;

    // Always fill the lower half of the canvas with felt green as a base.
    // This ensures the table never appears black even when perspective
    // corners project behind the camera.
    ctx.fillStyle = '#0e7a2b';
    ctx.fillRect(0, canvasH * 0.3, canvasW, canvasH * 0.7);

    // Project felt and rail corners, using clamped version for robustness
    const felt = [
      this._projectClamped(0, 0, 0),
      this._projectClamped(L, 0, 0),
      this._projectClamped(L, W, 0),
      this._projectClamped(0, W, 0),
    ];

    const railH = 1.5;
    const corners = [
      this._projectClamped(-rw, -rw, 0),
      this._projectClamped(L + rw, -rw, 0),
      this._projectClamped(L + rw, W + rw, 0),
      this._projectClamped(-rw, W + rw, 0),
    ];
    const railTop = [
      this._projectClamped(-rw, -rw, railH),
      this._projectClamped(L + rw, -rw, railH),
      this._projectClamped(L + rw, W + rw, railH),
      this._projectClamped(-rw, W + rw, railH),
    ];

    // Rail wood base
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(corners[i].x, corners[i].y);
    ctx.closePath();
    ctx.fillStyle = '#5a2d00';
    ctx.fill();

    // Felt surface
    ctx.beginPath();
    ctx.moveTo(felt[0].x, felt[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(felt[i].x, felt[i].y);
    ctx.closePath();
    ctx.fillStyle = '#0e7a2b';
    ctx.fill();

    // Felt texture lines
    ctx.strokeStyle = 'rgba(0,0,0,0.03)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const t = i / 10;
      const p1 = this._project(L * t, 0, 0);
      const p2 = this._project(L * t, W, 0);
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    // Rail surfaces (draw each rail as a quad between railTop and felt edge)
    // Far rail
    ctx.beginPath();
    ctx.moveTo(railTop[0].x, railTop[0].y);
    ctx.lineTo(railTop[1].x, railTop[1].y);
    ctx.lineTo(felt[1].x, felt[1].y);
    ctx.lineTo(felt[0].x, felt[0].y);
    ctx.closePath();
    ctx.fillStyle = '#6b3300';
    ctx.fill();

    // Left rail
    ctx.beginPath();
    ctx.moveTo(railTop[0].x, railTop[0].y);
    ctx.lineTo(railTop[3].x, railTop[3].y);
    ctx.lineTo(felt[3].x, felt[3].y);
    ctx.lineTo(felt[0].x, felt[0].y);
    ctx.closePath();
    ctx.fillStyle = '#5a2800';
    ctx.fill();

    // Right rail
    ctx.beginPath();
    ctx.moveTo(railTop[1].x, railTop[1].y);
    ctx.lineTo(railTop[2].x, railTop[2].y);
    ctx.lineTo(felt[2].x, felt[2].y);
    ctx.lineTo(felt[1].x, felt[1].y);
    ctx.closePath();
    ctx.fillStyle = '#5a2800';
    ctx.fill();

    // Near rail
    ctx.beginPath();
    ctx.moveTo(railTop[3].x, railTop[3].y);
    ctx.lineTo(railTop[2].x, railTop[2].y);
    ctx.lineTo(felt[2].x, felt[2].y);
    ctx.lineTo(felt[3].x, felt[3].y);
    ctx.closePath();
    ctx.fillStyle = '#7a3800';
    ctx.fill();

    // Pockets
    for (const pocket of this.table.pockets) {
      const p = this._project(pocket.x, pocket.y, 0);
      if (p) {
        const r = Math.max(2, pocket.radius * p.scale);
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = '#111';
        ctx.fill();
      }
    }

    // Cushion edges
    ctx.strokeStyle = '#0a9933';
    ctx.lineWidth = 2;
    for (const c of this.table.cushions) {
      const p1 = this._project(c.x1, c.y1, railH * 0.7);
      const p2 = this._project(c.x2, c.y2, railH * 0.7);
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }
  }

  _drawBall3D(ball, p) {
    const ctx = this.ctx;
    const r = Math.max(2, C.BALL_RADIUS * p.scale);

    if (ball.isStripe) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.clip();
      ctx.fillStyle = ball.color;
      ctx.fillRect(p.x - r, p.y - r * 0.45, r * 2, r * 0.9);
      ctx.restore();
    } else {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = ball.color;
      ctx.fill();
    }

    // 3D shading
    const grad = ctx.createRadialGradient(p.x - r * 0.3, p.y - r * 0.35, r * 0.1, p.x, p.y, r);
    grad.addColorStop(0, 'rgba(255,255,255,0.5)');
    grad.addColorStop(0.4, 'rgba(255,255,255,0.15)');
    grad.addColorStop(1, 'rgba(0,0,0,0.35)');
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Ball shadow on felt
    const shadowP = this._project(ball.x, ball.y, 0);
    if (shadowP) {
      const sr = C.BALL_RADIUS * shadowP.scale * 0.9;
      ctx.beginPath();
      ctx.ellipse(shadowP.x + sr * 0.2, shadowP.y + sr * 0.1, sr, sr * 0.4, 0, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.15)';
      ctx.fill();
    }

    // Number
    if (!ball.isCueBall && r > 5) {
      const nr = r * 0.35;
      ctx.beginPath();
      ctx.arc(p.x, p.y, nr, 0, Math.PI * 2);
      ctx.fillStyle = '#FFF';
      ctx.fill();
      if (r > 8) {
        ctx.fillStyle = '#000';
        ctx.font = `bold ${Math.round(r * 0.5)}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ball.number.toString(), p.x, p.y + 0.5);
      }
    }
  }

  _drawShotVector3D(shot) {
    const ctx = this.ctx;
    const R = C.BALL_RADIUS;

    // Cue stick: line from behind the cue ball to the cue ball
    const force = shot.force;
    const vecLen = (force / C.MAX_CUE_SPEED) * 15; // in inches
    const startX = shot.cueBallX - shot.aimDX * vecLen;
    const startY = shot.cueBallY - shot.aimDY * vecLen;

    const p1 = this._project(startX, startY, R + 0.5);
    const p2 = this._project(shot.cueBallX, shot.cueBallY, R);
    if (p1 && p2) {
      // Cue shaft
      const shaftStart = this._project(startX - shot.aimDX * 20, startY - shot.aimDY * 20, R + 1.5);
      if (shaftStart) {
        ctx.beginPath();
        ctx.moveTo(shaftStart.x, shaftStart.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.strokeStyle = '#8b6914';
        ctx.lineWidth = Math.max(2, 5 * p1.scale / 4);
        ctx.stroke();
      }
      // Cue tip
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = '#d4a84b';
      ctx.lineWidth = Math.max(1.5, 3 * p2.scale / 4);
      ctx.stroke();
    }

    // Aim line on the felt
    const aimLen = 80;
    const aimEnd = this._project(
      shot.cueBallX + shot.aimDX * aimLen,
      shot.cueBallY + shot.aimDY * aimLen, 0);
    const aimStart = this._project(shot.cueBallX, shot.cueBallY, 0);
    if (aimStart && aimEnd) {
      ctx.beginPath();
      ctx.setLineDash([4, 6]);
      ctx.moveTo(aimStart.x, aimStart.y);
      ctx.lineTo(aimEnd.x, aimEnd.y);
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Ghost ball
    if (shot.ghostX !== undefined) {
      const gp = this._project(shot.ghostX, shot.ghostY, R);
      if (gp) {
        const gr = Math.max(2, R * gp.scale);
        ctx.beginPath();
        ctx.arc(gp.x, gp.y, gr, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Target-to-pocket line on felt
    if (shot.ghostX !== undefined && shot.targetPocketX !== undefined) {
      const tpDx = shot.targetPocketX - shot.ghostX;
      const tpDy = shot.targetPocketY - shot.ghostY;
      const tpLen = Math.sqrt(tpDx * tpDx + tpDy * tpDy);
      if (tpLen > 0.1) {
        const tpNx = tpDx / tpLen;
        const tpNy = tpDy / tpLen;
        const targetX = shot.ghostX + tpNx * 2 * R;
        const targetY = shot.ghostY + tpNy * 2 * R;
        const tp1 = this._project(targetX, targetY, 0);
        const tp2 = this._project(shot.targetPocketX, shot.targetPocketY, 0);
        if (tp1 && tp2) {
          ctx.beginPath();
          ctx.setLineDash([5, 4]);
          ctx.moveTo(tp1.x, tp1.y);
          ctx.lineTo(tp2.x, tp2.y);
          ctx.strokeStyle = 'rgba(255,180,0,0.5)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // Pocket indicator
    if (shot.targetPocketX !== undefined) {
      const pp = this._project(shot.targetPocketX, shot.targetPocketY, 0);
      if (pp) {
        const pr = Math.max(3, C.POCKET_RADIUS * pp.scale * 0.5);
        ctx.beginPath();
        ctx.arc(pp.x, pp.y, pr, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,200,0,0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Target ball highlight
    if (shot.targetId !== undefined) {
      // Find the target ball in the projected list -- draw a ring
      // (we don't have the projected list here, so re-project)
      const targetBall = this._findBallById(shot.targetId);
      if (targetBall) {
        const tp = this._project(targetBall.x, targetBall.y, R);
        if (tp) {
          const tr = Math.max(3, R * tp.scale + 3);
          ctx.beginPath();
          ctx.arc(tp.x, tp.y, tr, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255,100,100,0.6)';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    }
  }

  _findBallById(id) {
    // This is a bit hacky -- we need access to the balls array.
    // Store it during render.
    return this._currentBalls ? this._currentBalls.find(b => b.id === id && !b.isPocketed) : null;
  }

  // __________________________________________________________________
  // OVERHEAD VIEW METHODS (unchanged from before)
  // __________________________________________________________________

  drawTable() {
    const ctx = this.ctx;
    const S = this.S;
    const rw = C.RAIL_WIDTH * S;
    const totalW = this.canvas.width;
    const totalH = this.canvas.height;
    const playW = C.TABLE_LENGTH * S;
    const playH = C.TABLE_WIDTH * S;

    ctx.fillStyle = '#3d1f00';
    ctx.fillRect(0, 0, totalW, totalH);
    ctx.fillStyle = '#6b3300';
    ctx.fillRect(rw * 0.3, rw * 0.3, totalW - rw * 0.6, totalH - rw * 0.6);
    ctx.fillStyle = '#0e7a2b';
    ctx.fillRect(rw, rw, playW, playH);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.02)';
    for (let i = 0; i < playW; i += 4) ctx.fillRect(rw + i, rw, 1, playH);

    ctx.strokeStyle = '#0a9933';
    ctx.lineWidth = 3;
    for (const c of this.table.cushions) {
      ctx.beginPath();
      ctx.moveTo(this.tx(c.x1), this.ty(c.y1));
      ctx.lineTo(this.tx(c.x2), this.ty(c.y2));
      ctx.stroke();
    }

    // Draw pockets with proper geometry:
    // Corner pockets: quarter-circle at the corner (only the part inside the table)
    // Side pockets: semicircle opening INTO the rail (away from playing surface)
    for (const p of this.table.pockets) {
      const px = this.tx(p.x);
      const py = this.ty(p.y);
      const pr = p.radius * S;
      ctx.beginPath();
      ctx.fillStyle = '#111';

      if (p.name === 'top-left') {
        ctx.arc(px, py, pr, 0, Math.PI / 2);         // quarter circle: right+down
        ctx.fill();
      } else if (p.name === 'top-right') {
        ctx.arc(px, py, pr, Math.PI / 2, Math.PI);   // quarter circle: down+left
        ctx.fill();
      } else if (p.name === 'bottom-left') {
        ctx.arc(px, py, pr, -Math.PI / 2, 0);        // quarter circle: up+right
        ctx.fill();
      } else if (p.name === 'bottom-right') {
        ctx.arc(px, py, pr, Math.PI, 3 * Math.PI / 2); // quarter circle: left+up
        ctx.fill();
      } else if (p.name === 'top-side') {
        // Semicircle opening upward INTO the top rail (away from playing surface)
        ctx.arc(px, py, pr, Math.PI, 2 * Math.PI);
        ctx.fill();
      } else if (p.name === 'bottom-side') {
        // Semicircle opening downward INTO the bottom rail
        ctx.arc(px, py, pr, 0, Math.PI);
        ctx.fill();
      }
    }

    this._drawDiamonds();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.beginPath();
    ctx.arc(this.tx(C.FOOT_SPOT_X), this.ty(C.FOOT_SPOT_Y), 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(this.tx(C.HEAD_SPOT_X), this.ty(C.HEAD_SPOT_Y), 3, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(this.tx(C.HEAD_SPOT_X), this.ty(0));
    ctx.lineTo(this.tx(C.HEAD_SPOT_X), this.ty(C.TABLE_WIDTH));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _drawDiamonds() {
    const ctx = this.ctx;
    const L = C.TABLE_LENGTH;
    const W = C.TABLE_WIDTH;
    const rw = C.RAIL_WIDTH;
    ctx.fillStyle = '#c8a96e';
    const ds = 3;
    for (let i = 1; i <= 3; i++) {
      this._drawDiamond(this.tx((L / 4) * i), this.ty(0) - rw * this.S * 0.4, ds);
      this._drawDiamond(this.tx((L / 4) * i), this.ty(W) + rw * this.S * 0.4, ds);
    }
    this._drawDiamond(this.tx(0) - rw * this.S * 0.4, this.ty(W / 2), ds);
    this._drawDiamond(this.tx(L) + rw * this.S * 0.4, this.ty(W / 2), ds);
  }

  _drawDiamond(cx, cy, size) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(cx, cy - size);
    ctx.lineTo(cx + size, cy);
    ctx.lineTo(cx, cy + size);
    ctx.lineTo(cx - size, cy);
    ctx.closePath();
    ctx.fill();
  }

  drawBall(ball) {
    const ctx = this.ctx;
    const cx = this.tx(ball.x);
    const cy = this.ty(ball.y);
    const r = ball.radius * this.S;
    const orient = ball.orientAngle || 0;
    const roll = ball.rollPhase || 0;
    const numberVisible = Math.cos(roll);
    const numberScale = Math.max(0, numberVisible);

    if (ball.isStripe) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.clip();
      ctx.translate(cx, cy);
      ctx.rotate(orient);
      const bandShift = Math.sin(roll) * r * 0.5;
      ctx.fillStyle = ball.color;
      ctx.fillRect(-r, -r * 0.45 + bandShift, r * 2, r * 0.9);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.restore();
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = ball.color;
      ctx.fill();
    }

    const grad = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.1, cx, cy, r);
    grad.addColorStop(0, 'rgba(255,255,255,0.4)');
    grad.addColorStop(0.5, 'rgba(255,255,255,0.1)');
    grad.addColorStop(1, 'rgba(0,0,0,0.3)');
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    if (!ball.isCueBall && numberScale > 0.1) {
      const numOffsetX = Math.sin(orient) * r * 0.3 * (1 - numberScale);
      const numOffsetY = Math.sin(roll) * r * 0.3;
      const nr = r * 0.38 * numberScale;
      ctx.beginPath();
      ctx.arc(cx + numOffsetX, cy + numOffsetY, nr, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
      if (numberScale > 0.4) {
        ctx.fillStyle = '#000';
        ctx.font = `bold ${Math.round(r * 0.55 * numberScale)}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ball.number.toString(), cx + numOffsetX, cy + numOffsetY + 0.5);
      }
    }

    if (ball.isCueBall) {
      const dotDist = r * 0.4 * Math.max(0, Math.cos(roll));
      const dotX = cx + Math.cos(orient) * dotDist;
      const dotY = cy + Math.sin(orient) * dotDist;
      ctx.beginPath();
      ctx.arc(dotX, dotY, r * 0.08, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(200,200,200,0.6)';
      ctx.fill();
    }
  }

  drawShotVector(shot) {
    const ctx = this.ctx;
    const cx = this.tx(shot.cueBallX);
    const cy = this.ty(shot.cueBallY);
    const force = shot.force;
    const dx = shot.aimDX;
    const dy = shot.aimDY;
    const vecLen = (force / C.MAX_CUE_SPEED) * 80;
    const startX = cx - dx * vecLen;
    const startY = cy - dy * vecLen;

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(cx, cy);
    ctx.strokeStyle = '#d4a84b';
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(startX - dx * 60, startY - dy * 60);
    ctx.strokeStyle = '#8b6914';
    ctx.lineWidth = 5;
    ctx.stroke();

    // Aim line -- for kick shots, draw to bounce point then to ghost;
    // for direct shots, draw a simple forward line.
    if (shot.kickBounceX !== undefined) {
      // Kick shot: draw cue ball -> bounce point -> ghost ball
      const bx = this.tx(shot.kickBounceX);
      const by = this.ty(shot.kickBounceY);

      // Segment 1: cue ball to bounce point
      ctx.beginPath();
      ctx.setLineDash([4, 6]);
      ctx.moveTo(cx, cy);
      ctx.lineTo(bx, by);
      ctx.strokeStyle = 'rgba(255, 200, 100, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);

      // Bounce point marker
      ctx.beginPath();
      ctx.arc(bx, by, 4, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 200, 100, 0.7)';
      ctx.fill();

      // Segment 2: bounce point to ghost ball
      if (shot.ghostX !== undefined) {
        ctx.beginPath();
        ctx.setLineDash([4, 6]);
        ctx.moveTo(bx, by);
        ctx.lineTo(this.tx(shot.ghostX), this.ty(shot.ghostY));
        ctx.strokeStyle = 'rgba(255, 200, 100, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    } else {
      // Direct shot: simple aim line forward
      ctx.beginPath();
      ctx.setLineDash([4, 6]);
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + dx * 200, cy + dy * 200);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (shot.ghostX !== undefined) {
      ctx.beginPath();
      ctx.arc(this.tx(shot.ghostX), this.ty(shot.ghostY),
              C.BALL_RADIUS * this.S, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      const gx = this.tx(shot.ghostX);
      const gy = this.ty(shot.ghostY);
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(gx - 4, gy); ctx.lineTo(gx + 4, gy);
      ctx.moveTo(gx, gy - 4); ctx.lineTo(gx, gy + 4);
      ctx.stroke();
    }

    // Bank shot: show target -> bounce point -> pocket
    if (shot.bankBounceX !== undefined && shot.ghostX !== undefined && shot.targetPocketX !== undefined) {
      const tpDx0 = shot.targetPocketX - shot.ghostX;
      const tpDy0 = shot.targetPocketY - shot.ghostY;
      const tpLen0 = Math.sqrt(tpDx0 * tpDx0 + tpDy0 * tpDy0);
      if (tpLen0 > 0.1) {
        const tpNx0 = tpDx0 / tpLen0;
        const tpNy0 = tpDy0 / tpLen0;
        const targetX = shot.ghostX + tpNx0 * 2 * C.BALL_RADIUS;
        const targetY = shot.ghostY + tpNy0 * 2 * C.BALL_RADIUS;
        const bbx = this.tx(shot.bankBounceX);
        const bby = this.ty(shot.bankBounceY);

        // Segment 1: target -> bounce point (orange dashed)
        ctx.beginPath();
        ctx.setLineDash([5, 4]);
        ctx.moveTo(this.tx(targetX), this.ty(targetY));
        ctx.lineTo(bbx, bby);
        ctx.strokeStyle = 'rgba(255, 160, 50, 0.6)';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);

        // Bounce point dot
        ctx.beginPath();
        ctx.arc(bbx, bby, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 160, 50, 0.8)';
        ctx.fill();

        // Segment 2: bounce -> pocket (lighter orange dashed)
        ctx.beginPath();
        ctx.setLineDash([5, 4]);
        ctx.moveTo(bbx, bby);
        ctx.lineTo(this.tx(shot.targetPocketX), this.ty(shot.targetPocketY));
        ctx.strokeStyle = 'rgba(255, 160, 50, 0.4)';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    // Direct shot: target -> pocket corridor
    else if (shot.ghostX !== undefined && shot.targetPocketX !== undefined) {
      const tpDx = shot.targetPocketX - shot.ghostX;
      const tpDy = shot.targetPocketY - shot.ghostY;
      const tpLen = Math.sqrt(tpDx * tpDx + tpDy * tpDy);
      if (tpLen > 0.1) {
        const tpNx = tpDx / tpLen;
        const tpNy = tpDy / tpLen;
        const targetX = shot.ghostX + tpNx * 2 * C.BALL_RADIUS;
        const targetY = shot.ghostY + tpNy * 2 * C.BALL_RADIUS;
        const perpX = -tpNy;
        const perpY = tpNx;
        const ballR = C.BALL_RADIUS;

        ctx.beginPath();
        ctx.moveTo(this.tx(targetX + perpX * ballR), this.ty(targetY + perpY * ballR));
        ctx.lineTo(this.tx(shot.targetPocketX + perpX * ballR), this.ty(shot.targetPocketY + perpY * ballR));
        ctx.lineTo(this.tx(shot.targetPocketX - perpX * ballR), this.ty(shot.targetPocketY - perpY * ballR));
        ctx.lineTo(this.tx(targetX - perpX * ballR), this.ty(targetY - perpY * ballR));
        ctx.closePath();
        ctx.fillStyle = 'rgba(255,180,0,0.08)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,180,0,0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.moveTo(this.tx(targetX), this.ty(targetY));
        ctx.lineTo(this.tx(shot.targetPocketX), this.ty(shot.targetPocketY));
        ctx.strokeStyle = 'rgba(255,180,0,0.6)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (shot.ghostX !== undefined) {
      const cbDx = shot.ghostX - shot.cueBallX;
      const cbDy = shot.ghostY - shot.cueBallY;
      const cbLen = Math.sqrt(cbDx * cbDx + cbDy * cbDy);
      if (cbLen > 0.1) {
        const perpX = -cbDy / cbLen;
        const perpY = cbDx / cbLen;
        const ballR = C.BALL_RADIUS;
        ctx.beginPath();
        ctx.moveTo(this.tx(shot.cueBallX + perpX * ballR), this.ty(shot.cueBallY + perpY * ballR));
        ctx.lineTo(this.tx(shot.ghostX + perpX * ballR), this.ty(shot.ghostY + perpY * ballR));
        ctx.lineTo(this.tx(shot.ghostX - perpX * ballR), this.ty(shot.ghostY - perpY * ballR));
        ctx.lineTo(this.tx(shot.cueBallX - perpX * ballR), this.ty(shot.cueBallY - perpY * ballR));
        ctx.closePath();
        ctx.fillStyle = 'rgba(255,255,255,0.04)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    if (shot.targetPocketX !== undefined) {
      ctx.beginPath();
      ctx.arc(this.tx(shot.targetPocketX), this.ty(shot.targetPocketY),
              C.POCKET_RADIUS * this.S * 0.6, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,200,0,0.6)';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
  }

  // Draw the planned next shot: highlight next target ball, show estimated
  // cue ball position and planned path to the next ball.
  _drawNextShotPlan(shot, balls) {
    if (shot.nextBallId === null || shot.nextBallId === undefined) return;
    const ctx = this.ctx;
    const R = C.BALL_RADIUS;

    const nextBall = balls.find(b => b.id === shot.nextBallId && !b.isPocketed);
    if (!nextBall) return;

    // Highlight the next target ball with a blue ring
    ctx.beginPath();
    ctx.arc(this.tx(nextBall.x), this.ty(nextBall.y),
            nextBall.radius * this.S + 6, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(80, 160, 255, 0.6)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label "NEXT" near the ball
    ctx.fillStyle = 'rgba(80, 160, 255, 0.7)';
    ctx.font = 'bold 10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('NEXT', this.tx(nextBall.x), this.ty(nextBall.y) - nextBall.radius * this.S - 10);

    // Draw estimated cue ball position after current shot (small circle)
    if (shot.estCueBallX !== null && shot.estCueBallY !== null) {
      const ecx = this.tx(shot.estCueBallX);
      const ecy = this.ty(shot.estCueBallY);
      const er = R * this.S * 0.6;

      // Estimated cue ball position -- dashed circle
      ctx.beginPath();
      ctx.arc(ecx, ecy, er, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(200, 200, 255, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Small "CB" label
      ctx.fillStyle = 'rgba(200, 200, 255, 0.5)';
      ctx.font = '8px Arial';
      ctx.fillText('cb', ecx, ecy + 3);

      // Line from estimated cue ball to next ghost ball position
      if (shot.nextGhostX !== undefined) {
        ctx.beginPath();
        ctx.setLineDash([3, 5]);
        ctx.moveTo(ecx, ecy);
        ctx.lineTo(this.tx(shot.nextGhostX), this.ty(shot.nextGhostY));
        ctx.strokeStyle = 'rgba(80, 160, 255, 0.25)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Next shot: target ball to pocket -- draw as a corridor with ball width
      if (shot.nextPocketX !== undefined) {
        const npDx = shot.nextPocketX - nextBall.x;
        const npDy = shot.nextPocketY - nextBall.y;
        const npLen = Math.sqrt(npDx * npDx + npDy * npDy);
        if (npLen > 0.1) {
          const npNx = npDx / npLen;
          const npNy = npDy / npLen;
          const perpX = -npNy;
          const perpY = npNx;
          const ballR = R;

          // Corridor fill
          ctx.beginPath();
          ctx.moveTo(this.tx(nextBall.x + perpX * ballR), this.ty(nextBall.y + perpY * ballR));
          ctx.lineTo(this.tx(shot.nextPocketX + perpX * ballR), this.ty(shot.nextPocketY + perpY * ballR));
          ctx.lineTo(this.tx(shot.nextPocketX - perpX * ballR), this.ty(shot.nextPocketY - perpY * ballR));
          ctx.lineTo(this.tx(nextBall.x - perpX * ballR), this.ty(nextBall.y - perpY * ballR));
          ctx.closePath();
          ctx.fillStyle = 'rgba(80, 160, 255, 0.05)';
          ctx.fill();

          // Corridor edges
          ctx.strokeStyle = 'rgba(80, 160, 255, 0.2)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 4]);
          ctx.stroke();
          ctx.setLineDash([]);

          // Center line
          ctx.beginPath();
          ctx.setLineDash([4, 4]);
          ctx.moveTo(this.tx(nextBall.x), this.ty(nextBall.y));
          ctx.lineTo(this.tx(shot.nextPocketX), this.ty(shot.nextPocketY));
          ctx.strokeStyle = 'rgba(80, 160, 255, 0.35)';
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.setLineDash([]);

          // Next pocket indicator
          ctx.beginPath();
          ctx.arc(this.tx(shot.nextPocketX), this.ty(shot.nextPocketY),
                  C.POCKET_RADIUS * this.S * 0.5, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(80, 160, 255, 0.4)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }
  }

  drawSpinIndicator(shot) {
    const ctx = this.spinCtx;
    const canvas = this.spinCanvas;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.beginPath();
    ctx.roundRect(0, 0, w, h, 8);
    ctx.fill();

    ctx.fillStyle = '#ccc';
    ctx.font = '11px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Cue Contact', w / 2, 14);

    const ballR = 35;
    const bcx = w / 2;
    const bcy = h / 2 + 5;
    ctx.beginPath();
    ctx.arc(bcx, bcy, ballR, 0, Math.PI * 2);
    ctx.fillStyle = '#e8e8e8';
    ctx.fill();
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(bcx - ballR, bcy); ctx.lineTo(bcx + ballR, bcy);
    ctx.moveTo(bcx, bcy - ballR); ctx.lineTo(bcx, bcy + ballR);
    ctx.stroke();

    const dotX = bcx + (shot.contactX || 0) * ballR * 0.85;
    const dotY = bcy - (shot.contactY || 0) * ballR * 0.85;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ff3333';
    ctx.fill();
    ctx.strokeStyle = '#cc0000';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const scx = shot.contactX || 0;
    const scy = shot.contactY || 0;
    const elevDeg = (shot.elevation || 0) * 180 / Math.PI;
    let spinLabel = '';
    if (scy > 0.15) spinLabel = 'Follow';
    else if (scy < -0.15) spinLabel = 'Draw';
    if (scx > 0.15) spinLabel += (spinLabel ? ' + ' : '') + 'Right';
    else if (scx < -0.15) spinLabel += (spinLabel ? ' + ' : '') + 'Left';
    if (!spinLabel) spinLabel = 'Center';
    if (elevDeg > 1) spinLabel += ` ${elevDeg.toFixed(0)}deg`;

    ctx.fillStyle = '#aaa';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(spinLabel, w / 2, h - 6);

    // Cue elevation side-view diagram -- always shown
    // Shows the cue stick angle relative to the table surface
    const elevRad = shot.elevation || 0;
    const elevCenterX = w / 2;
    const elevY = bcy + ballR + 14;
    const cueLen = 40;
    const ballRSmall = 6;

    // Table surface line
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(10, elevY);
    ctx.lineTo(w - 10, elevY);
    ctx.stroke();

    // Small ball (side view)
    ctx.beginPath();
    ctx.arc(elevCenterX, elevY - ballRSmall, ballRSmall, 0, Math.PI * 2);
    ctx.fillStyle = '#ddd';
    ctx.fill();
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Cue stick at elevation angle
    const cueStartX = elevCenterX - 3;
    const cueStartY = elevY - ballRSmall;
    const cueEndX = cueStartX - Math.cos(elevRad) * cueLen;
    const cueEndY = cueStartY - Math.sin(elevRad) * cueLen;

    // Cue shaft (thicker, darker)
    ctx.strokeStyle = '#8b6914';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cueEndX, cueEndY);
    ctx.lineTo(cueEndX + (cueStartX - cueEndX) * 0.4, cueEndY + (cueStartY - cueEndY) * 0.4);
    ctx.stroke();

    // Cue tip (thinner, lighter)
    ctx.strokeStyle = '#d4a84b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cueEndX + (cueStartX - cueEndX) * 0.4, cueEndY + (cueStartY - cueEndY) * 0.4);
    ctx.lineTo(cueStartX, cueStartY);
    ctx.stroke();

    // Elevation label
    ctx.fillStyle = elevDeg > 1 ? '#e8c070' : '#777';
    ctx.font = '9px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(elevDeg > 0.5 ? `${elevDeg.toFixed(0)}deg` : 'Level', elevCenterX, elevY + 10);
  }

  drawPocketedBalls(balls) {
    const ctx = this.ctx;
    const S = this.S;
    const r = C.BALL_RADIUS * S * 0.6;
    const totalW = this.canvas.width;
    let solidX = this.offsetX + 10;
    let stripeX = totalW - this.offsetX - 10;
    const trayY = this.canvas.height - C.RAIL_WIDTH * S * 0.35;

    ctx.font = '10px Arial';
    ctx.fillStyle = '#c8a96e';
    ctx.textAlign = 'left';
    ctx.fillText('Solids', solidX, trayY - r - 4);
    ctx.textAlign = 'right';
    ctx.fillText('Stripes', stripeX, trayY - r - 4);

    for (const ball of balls) {
      if (!ball.isPocketed || ball.isCueBall) continue;
      if (ball.isSolid || ball.isEightBall) {
        ctx.beginPath();
        ctx.arc(solidX, trayY, r, 0, Math.PI * 2);
        ctx.fillStyle = ball.color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
        solidX += r * 2.5;
      } else if (ball.isStripe) {
        ctx.beginPath();
        ctx.arc(stripeX, trayY, r, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
        ctx.save();
        ctx.beginPath();
        ctx.arc(stripeX, trayY, r, 0, Math.PI * 2);
        ctx.clip();
        ctx.fillStyle = ball.color;
        ctx.fillRect(stripeX - r, trayY - r * 0.4, r * 2, r * 0.8);
        ctx.restore();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.arc(stripeX, trayY, r, 0, Math.PI * 2);
        ctx.stroke();
        stripeX -= r * 2.5;
      }
    }
  }
}
