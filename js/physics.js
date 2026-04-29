// physics.js -- Physics engine: friction, collisions, spin, cushions
const Physics = {

  // Event log for foul detection. Reset before each shot, read after simulation.
  // Events: { type: 'ball-hit', hitterId: N, hitId: N }
  //         { type: 'cushion', ballId: N }
  //         { type: 'pocketed', ballId: N }
  //         { type: 'off-table', ballId: N }
  events: [],

  resetEvents() {
    this.events = [];
    this._collisionPairsThisShot = new Set(); // track unique collision pairs
  },

  // Apply one simulation step to all balls
  simulateStep(balls, table, dt) {
    const activeBalls = balls.filter(b => !b.isPocketed);

    // 1. Apply friction and spin dynamics
    for (const ball of activeBalls) {
      if (ball.speed() < C.VELOCITY_THRESHOLD && ball.angularSpeed() < C.ANGULAR_VEL_THRESHOLD) {
        ball.vx = 0; ball.vy = 0;
        ball.wx = 0; ball.wy = 0; ball.wz = 0;
        continue;
      }
      this._applyFriction(ball, dt);
    }

    // 2. Move balls and accumulate visual rotation
    for (const ball of activeBalls) {
      ball.x += ball.vx * dt;
      ball.y += ball.vy * dt;
      ball.orientAngle += ball.wz * dt;
      const rollSpeed = Math.sqrt(ball.wx * ball.wx + ball.wy * ball.wy);
      ball.rollPhase += rollSpeed * dt;
    }

    // 3. Ball-ball collisions (with event logging)
    // Run multiple iterations so impulses propagate through tightly packed clusters
    // (e.g., the break rack) within a single timestep.
    //
    // Two-phase approach per iteration:
    //   Phase A: Apply impulses WITHOUT separating balls (preserve overlaps so
    //            the next iteration can still detect adjacent contacts)
    //   Phase B: After all impulse iterations, do a final separation pass.
    const COLLISION_ITERATIONS = 10;
    for (let iter = 0; iter < COLLISION_ITERATIONS; iter++) {
      let anyCollision = false;
      for (let i = 0; i < activeBalls.length; i++) {
        for (let j = i + 1; j < activeBalls.length; j++) {
          // On intermediate iterations, skip separation (impulse only).
          // On the final iteration, do full separation.
          const separateNow = (iter === COLLISION_ITERATIONS - 1);
          const collided = this._resolveBallCollision(activeBalls[i], activeBalls[j], separateNow);
          if (collided) {
            anyCollision = true;
            const pairKey = Math.min(activeBalls[i].id, activeBalls[j].id) + ':' +
                            Math.max(activeBalls[i].id, activeBalls[j].id);
            if (!this._collisionPairsThisShot.has(pairKey)) {
              this._collisionPairsThisShot.add(pairKey);
              this.events.push({ type: 'ball-hit', hitterId: activeBalls[i].id, hitId: activeBalls[j].id });
              this.events.push({ type: 'ball-hit', hitterId: activeBalls[j].id, hitId: activeBalls[i].id });
            }
          }
        }
      }
      if (!anyCollision) break;
    }
    // Final separation pass: push apart any balls that are still overlapping
    for (let i = 0; i < activeBalls.length; i++) {
      for (let j = i + 1; j < activeBalls.length; j++) {
        const b1 = activeBalls[i], b2 = activeBalls[j];
        const dx = b2.x - b1.x, dy = b2.y - b1.y;
        const distSq = dx * dx + dy * dy;
        const minDist = b1.radius + b2.radius;
        if (distSq < minDist * minDist && distSq > 0.0001) {
          const dist = Math.sqrt(distSq);
          const overlap = minDist - dist;
          const nx = dx / dist, ny = dy / dist;
          b1.x -= nx * overlap / 2;
          b1.y -= ny * overlap / 2;
          b2.x += nx * overlap / 2;
          b2.y += ny * overlap / 2;
        }
      }
    }

    // 4. Cushion collisions (with event logging)
    for (const ball of activeBalls) {
      const hitCushion = this._resolveCushionCollision(ball, table);
      if (hitCushion) {
        this.events.push({ type: 'cushion', ballId: ball.id });
      }
    }

    // 5. Check pocketing and off-table
    for (const ball of activeBalls) {
      // Check if ball went off the table (outside playing surface bounds with margin)
      const margin = ball.radius * 2;
      if (ball.x < -margin || ball.x > C.TABLE_LENGTH + margin ||
          ball.y < -margin || ball.y > C.TABLE_WIDTH + margin) {
        if (!ball.isPocketed) {
          this.events.push({ type: 'off-table', ballId: ball.id });
          ball.isPocketed = true; // temporarily remove; game.js will re-spot
          ball.vx = 0; ball.vy = 0;
          ball.wx = 0; ball.wy = 0; ball.wz = 0;
        }
      }
      // Check pocketing
      if (!ball.isPocketed && table.isPocketed(ball)) {
        ball.isPocketed = true;
        ball.vx = 0; ball.vy = 0;
        ball.wx = 0; ball.wy = 0; ball.wz = 0;
        this.events.push({ type: 'pocketed', ballId: ball.id });
      }
    }
  },

  // Sommerfeld billiard friction model (Lectures on Theoretical Physics, Vol 1, Appendix)
  //
  // A rigid sphere of mass m, radius R, moment of inertia I = (2/5)mR^2 moves on
  // a horizontal surface. At the contact point P the velocity is:
  //
  //   v_P = v_cm + w x (-R z_hat)
  //
  // where z_hat points up from the table. In component form (x-y plane of the table):
  //
  //   v_Px = vx + wy * R      (spin around y pushes contact point in +x)
  //   v_Py = vy - wx * R      (spin around x pushes contact point in -y)
  //
  // If |v_P| > 0 the ball SLIDES. Coulomb friction F = u_s * m * g opposes v_P.
  // This friction simultaneously:
  //   - decelerates v_cm:  dv/dt = -(u_s * g) * v_hat_P
  //   - accelerates w:     dw/dt = (R x F) / I = (5u_s*g)/(2R) * (component from v_hat_P)
  //
  // The slip speed decreases monotonically. When |v_P| = 0 the ball transitions to
  // ROLLING. The rolling constraint v_cm = -w x R*z_hat is thereafter maintained
  // exactly, and only rolling resistance u_r * m * g decelerates the ball.
  //
  // The transition happens once; the ball never returns to sliding unless an external
  // impulse (collision, cushion) disrupts the rolling constraint.
  //
  // Sidespin wz (english, about the vertical axis) is orthogonal to the rolling
  // constraint and decays independently via drill friction at the contact point.
  // While the ball slides, wz also produces a lateral curved path (Coriolis-like
  // deflection) because the total slip vector at P includes the wz contribution.

  _applyFriction(ball, dt) {
    const R = ball.radius;
    const speed = ball.speed();

    // --- Contact-point slip velocity (Sommerfeld eq.) ---
    const slipX = ball.vx + ball.wy * R;
    const slipY = ball.vy - ball.wx * R;
    const slipSpeed = Math.sqrt(slipX * slipX + slipY * slipY);

    // Threshold for the sliding -> rolling transition.
    // Use a tighter threshold than the global stop threshold so we don't
    // prematurely kill slow-but-sliding balls (e.g. draw shot decelerating).
    const SLIP_THRESHOLD = C.VELOCITY_THRESHOLD * 0.5;

    if (slipSpeed > SLIP_THRESHOLD) {
      // -- SLIDING PHASE --
      // Kinetic friction magnitude: F = u_s * m * g   (direction opposes v_P)
      const fricAccel = C.MU_SLIDE * C.G;   // acceleration = F/m
      const slipNX = slipX / slipSpeed;
      const slipNY = slipY / slipSpeed;

      // Check whether this timestep would overshoot the transition.
      // The slip speed decreases at rate (7/2) * u_s * g  (combined effect of
      // both the linear and angular equations -- Sommerfeld shows they converge
      // as dv_slip/dt = -(7/2)*u_s*g for a uniform sphere).
      const slipDecelRate = (7 / 2) * fricAccel;
      const slipReduction = slipDecelRate * dt;

      if (slipReduction >= slipSpeed) {
        // Transition happens within this timestep -- snap to rolling.
        // Use angular-momentum-conserving combined velocity (Sommerfeld):
        //   v_roll = (5*v_cm + 2*v_surface_from_spin) / 7
        // where v_surface_from_spin = (-wy*R,  wx*R)
        ball.vx = (5 * ball.vx - 2 * ball.wy * R) / 7;
        ball.vy = (5 * ball.vy + 2 * ball.wx * R) / 7;
        ball.wy = -ball.vx / R;
        ball.wx =  ball.vy / R;
      } else {
        // Normal sliding integration
        // Linear: dv/dt = -u*g * v_hat_P
        ball.vx -= fricAccel * slipNX * dt;
        ball.vy -= fricAccel * slipNY * dt;

        // Angular: dw/dt = (5/(2R)) * u*g * (torque direction from friction)
        //   Friction at P acts in direction -v_hat_P. Torque = (-R z_hat) x F.
        //   Working out components:
        //     dwx/dt = +(5/(2R)) * u*g * slipNY
        //     dwy/dt = -(5/(2R)) * u*g * slipNX
        const angAccel = (5 / (2 * R)) * fricAccel;
        ball.wx += angAccel * slipNY * dt;
        ball.wy -= angAccel * slipNX * dt;

        // -- Sidespin (wz) lateral curve (masse/swerve) --
        // With a LEVEL cue, english produces almost NO curve on the cue ball's
        // path. The ball goes straight to where it's aimed. English primarily
        // affects cushion rebound angles and slight object ball throw.
        //
        // Curve (swerve) only becomes significant with cue ELEVATION, where
        // the downward force component creates a vertical spin axis that
        // interacts with the cloth friction to produce lateral deflection.
        //
        // ball._cueElevation stores the elevation angle from the strike.
        // With elev = 0 (level): curve factor ~ 0 (no swerve)
        // With elev > 10deg: curve becomes noticeable (masse territory)
        if (Math.abs(ball.wz) > C.ANGULAR_VEL_THRESHOLD && speed > 2.0 && slipSpeed > 5.0) {
          const elevFactor = ball._cueElevation || 0; // radians
          // Only apply curve if cue was elevated (> ~5 degrees)
          if (elevFactor > 0.08) {
            const effectiveSlip = Math.max(slipSpeed, speed * 0.3);
            // Scale curve by sin(elevation) -- level cue = 0, 30deg = 0.5, 45deg = 0.7
            const curveAccel = Math.sin(elevFactor) * (2 / 3) * C.MU_SLIDE * C.G * (ball.wz * R) / effectiveSlip;
            const maxCurve = C.MU_SLIDE * C.G * 0.15;
            const clampedCurve = Math.max(-maxCurve, Math.min(maxCurve, curveAccel));
            const vnx = ball.vx / speed;
            const vny = ball.vy / speed;
            ball.vx -= clampedCurve * vny * dt;
            ball.vy += clampedCurve * vnx * dt;
          }
        }
      }

    } else {
      // -- ROLLING PHASE --
      // The no-slip constraint is already (approximately) satisfied.
      // Enforce it exactly, then apply rolling resistance.

      // Snap angular velocity to match linear (maintain constraint):
      ball.wy = -ball.vx / R;
      ball.wx =  ball.vy / R;

      // Rolling resistance: F_roll = u_r * m * g, opposing v_cm
      const rollSpeed = speed;  // recalc isn't needed, speed is current
      if (rollSpeed > C.VELOCITY_THRESHOLD) {
        const rollDecel = C.MU_ROLL * C.G;
        const dv = rollDecel * dt;
        if (dv >= rollSpeed) {
          ball.vx = 0; ball.vy = 0;
          ball.wx = 0; ball.wy = 0;
        } else {
          const factor = 1 - dv / rollSpeed;
          ball.vx *= factor;
          ball.vy *= factor;
          ball.wy = -ball.vx / R;
          ball.wx =  ball.vy / R;
        }
      } else {
        ball.vx = 0; ball.vy = 0;
        ball.wx = 0; ball.wy = 0;
      }
    }

    // -- Drill spin (wz) decay --
    // Sidespin about the vertical axis is orthogonal to the rolling constraint.
    // It decays via drill friction at the contact patch. Sommerfeld treats this
    // as a torque proportional to the normal force: tau_drill ~ u_drill * m*g * R.
    // We model it as a constant angular deceleration.
    if (Math.abs(ball.wz) > C.ANGULAR_VEL_THRESHOLD) {
      const spinDecel = C.MU_SPIN_DECEL * dt;
      if (Math.abs(ball.wz) <= spinDecel) {
        ball.wz = 0;
      } else {
        ball.wz -= Math.sign(ball.wz) * spinDecel;
      }
    } else {
      ball.wz = 0;
    }
  },

  _resolveBallCollision(b1, b2, separate) {
    // Skip if both balls are stationary -- avoids resolving pre-compressed
    // rack balls against each other before the break shot disturbs them.
    if (b1.speed() < 0.01 && b2.speed() < 0.01) return false;

    const dx = b2.x - b1.x;
    const dy = b2.y - b1.y;
    const distSq = dx * dx + dy * dy;
    const minDist = b1.radius + b2.radius;

    if (distSq >= minDist * minDist || distSq < 0.0001) return false;

    const dist = Math.sqrt(distSq);

    // Back-calculate the true collision point.
    // The balls overlapped during this timestep. Find the time t (0..dt) when
    // they first touched, and use positions at that time for the collision normal.
    // This eliminates the angular error from using overlapping positions.
    const dvx_full = b1.vx - b2.vx;
    const dvy_full = b1.vy - b2.vy;
    // Relative position before this step: the balls were further apart by dv*dt
    // We solve: |pos1(t) - pos2(t)| = minDist for the collision normal
    // pos_rel(t) = (dx - dvx*dt_back, dy - dvy*dt_back)  where dt_back is time to back up
    // For small overlaps, linear interpolation is accurate enough:
    const overlap = minDist - dist;
    const closingSpeed = dvx_full * dx / dist + dvy_full * dy / dist;
    // Time to back up to the contact moment:
    const dt_back = closingSpeed > 0.01 ? overlap / closingSpeed : 0;
    // Positions at contact moment:
    const cx1 = b1.x - b1.vx * dt_back;
    const cy1 = b1.y - b1.vy * dt_back;
    const cx2 = b2.x - b2.vx * dt_back;
    const cy2 = b2.y - b2.vy * dt_back;
    // True collision normal from positions at contact:
    const cdx = cx2 - cx1;
    const cdy = cy2 - cy1;
    const cdist = Math.sqrt(cdx * cdx + cdy * cdy);
    const nx = cdist > 0.001 ? cdx / cdist : dx / dist;
    const ny = cdist > 0.001 ? cdy / cdist : dy / dist;

    // Separate overlapping balls along the corrected normal (only if requested)
    if (separate !== false) {
      b1.x -= nx * overlap / 2;
      b1.y -= ny * overlap / 2;
      b2.x += nx * overlap / 2;
      b2.y += ny * overlap / 2;
    }

    // Relative velocity along corrected normal
    const dvx = b1.vx - b2.vx;
    const dvy = b1.vy - b2.vy;
    const dvn = dvx * nx + dvy * ny;

    // Only resolve if approaching
    if (dvn <= 0) return false;

    // Impulse (equal masses)
    const e = C.BALL_RESTITUTION;
    const j = (1 + e) * dvn / 2; // per unit mass

    b1.vx -= j * nx;
    b1.vy -= j * ny;
    b2.vx += j * nx;
    b2.vy += j * ny;

    // Spin transfer -- tangential friction during the brief collision.
    // The tangential impulse is limited by Coulomb friction: |jt| <= u * |jn|.
    // This prevents unrealistic lateral deflections on high-speed collisions
    // (like the break) where the spin contribution is large relative to
    // the tangential sliding velocity.
    const R = b1.radius;
    const tvx = dvx - dvn * nx;
    const tvy = dvy - dvn * ny;
    // Spin contributions at the contact point (scaled conservatively)
    const spinContribX = (b1.wy + b2.wy) * R * 0.3;
    const spinContribY = -(b1.wx + b2.wx) * R * 0.3;
    const totalTvx = tvx + spinContribX;
    const totalTvy = tvy + spinContribY;
    const tanSpeed = Math.sqrt(totalTvx * totalTvx + totalTvy * totalTvy);

    if (tanSpeed > 0.5) {
      // Coulomb limit: tangential impulse <= friction coefficient x normal impulse
      const mu_ball = 0.05; // ball-ball friction coefficient (very low, polished surfaces)
      const jtMax = mu_ball * Math.abs(j) * b1.mass;
      const jtDesired = C.SPIN_TRANSFER_COEFF * Math.abs(j) * b1.mass;
      const jt = Math.min(jtDesired, jtMax);
      const tnx = totalTvx / tanSpeed;
      const tny = totalTvy / tanSpeed;

      b1.vx -= jt * tnx / b1.mass;
      b1.vy -= jt * tny / b1.mass;
      b2.vx += jt * tnx / b2.mass;
      b2.vy += jt * tny / b2.mass;

      // Transfer sidespin (small amount)
      const wzTransfer = (b1.wz - b2.wz) * mu_ball;
      b1.wz -= wzTransfer * 0.3;
      b2.wz += wzTransfer * 0.3;
    }
    return true;
  },

  _resolveCushionCollision(ball, table) {
    const R = ball.radius;
    let hit = false;

    for (const cushion of table.cushions) {
      const dist = this._pointToSegmentDist(ball.x, ball.y,
        cushion.x1, cushion.y1, cushion.x2, cushion.y2);

      if (dist >= R) continue;

      const nx = cushion.nx;
      const ny = cushion.ny;
      const vn = ball.vx * nx + ball.vy * ny;
      if (vn >= 0) continue;

      const tx = -ny;
      const ty = nx;
      const vt = ball.vx * tx + ball.vy * ty;
      const vnNew = -vn * C.CUSHION_RESTITUTION;
      let vtNew = vt * (1 - C.CUSHION_FRICTION);
      vtNew += ball.wz * R * C.CUSHION_FRICTION * 0.8;
      ball.vx = vnNew * nx + vtNew * tx;
      ball.vy = vnNew * ny + vtNew * ty;

      if (nx !== 0) {
        if (nx > 0) ball.x = Math.max(ball.x, R);
        else ball.x = Math.min(ball.x, C.TABLE_LENGTH - R);
      }
      if (ny !== 0) {
        if (ny > 0) ball.y = Math.max(ball.y, R);
        else ball.y = Math.min(ball.y, C.TABLE_WIDTH - R);
      }

      ball.wz *= 0.7;
      if (Math.abs(nx) > 0.5) ball.wy += vn * 0.1;
      else ball.wx += vn * 0.1;

      hit = true;
    }
    return hit;
  },

  _pointToSegmentDist(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    if (lenSq < 0.0001) {
      const ex = px - x1, ey = py - y1;
      return Math.sqrt(ex * ex + ey * ey);
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const closestX = x1 + t * dx;
    const closestY = y1 + t * dy;
    const ex = px - closestX;
    const ey = py - closestY;
    return Math.sqrt(ex * ex + ey * ey);
  },

  // Strike the cue ball with given parameters.
  //
  // contactX, contactY: where the cue tip contacts the ball (-1 to +1 each)
  // elevation: cue stick angle above horizontal (radians), 0 = level.
  //   - Level cue (0deg): standard shot. Spin is proportional to contact offset.
  //   - Moderate elevation (5-15deg): amplifies backspin. The downward force
  //     component compresses the ball into the cloth, increasing initial friction
  //     and making the draw effect stronger.
  //   - High elevation (20-45deg): mass_ shot. Extreme backspin plus the downward
  //     push causes the ball to curve dramatically.
  //
  // Physics: the cue delivers an impulse along its axis. With elevation angle theta:
  //   - Horizontal speed component: v_h = speed * cos(theta)
  //   - Vertical component pushes ball into cloth: v_down = speed * sin(theta)
  //   - The contact point on the ball shifts: the effective vertical offset
  //     increases because the cue approaches from above.
  //   - Backspin magnification: the cue tip's approach angle increases the
  //     torque arm for spin, roughly by factor 1 / cos(theta) for small angles,
  //     plus the vertical push creates additional friction-driven spin.
  strikeCueBall(cueBall, aimDX, aimDY, force, contactX, contactY, elevation) {
    const R = cueBall.radius;
    const elev = elevation || 0; // default: level cue
    const cosE = Math.cos(elev);
    const sinE = Math.sin(elev);

    // Horizontal speed (what propels the ball across the table)
    const speed = force * cosE;

    // Linear velocity along aim direction
    cueBall.vx = speed * aimDX;
    cueBall.vy = speed * aimDY;

    // Contact point offsets (capped at 70% of radius for miscue safety)
    const a = contactX * R * 0.7; // horizontal offset -> english
    const b = contactY * R * 0.7; // vertical offset -> topspin/backspin

    // With elevation, the effective vertical offset increases slightly.
    const effectiveB = b - R * sinE * 0.3;

    // Store elevation for the friction model (curve only with elevated cue)
    cueBall._cueElevation = elev;

    // Sidespin (wz) from horizontal offset
    // Scale by 0.6 -- the cue tip doesn't grip perfectly, so less spin is
    // transferred than the theoretical maximum. This also keeps english
    // effects realistic: noticeable on cushion rebounds, subtle on ball throw.
    cueBall.wz = (5 * speed * a) / (2 * R * R) * 0.6;

    // Topspin/backspin from vertical offset.
    // The spin imparted by the cue is: w = (5 * v * b_eff) / (2 * R^2)
    // where v is the horizontal speed (not the full force).
    // Elevation adds a modest amplification (up to 1.4x at 30deg).
    // This produces realistic draw: a -0.35 contactY with moderate speed
    // gives w*R ~ 0.35 * v, so the rolling velocity = v * (5 - 2*0.35)/7
    // ~ 0.61*v -- the ball slows but doesn't dramatically reverse.
    // A strong draw (-0.5, 10deg elevation) gives w*R ~ 0.5*v,
    // rolling vel ~ v*(5-1.0)/7 = 0.57*v -- still forward, but with residual
    // backspin that causes the ball to slow and pull back after collision.
    const spinAmplification = 1 + sinE * 0.8; // modest: up to 1.4x at ~45deg
    const spinMag = (5 * speed * effectiveB * spinAmplification) / (2 * R * R);
    cueBall.wx =  spinMag * aimDY;
    cueBall.wy = -spinMag * aimDX;
  },

  // Check if all balls have stopped
  allStopped(balls) {
    for (const ball of balls) {
      if (!ball.isPocketed && ball.isMoving()) return false;
    }
    return true;
  },
};
