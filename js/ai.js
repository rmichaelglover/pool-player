// ai.js -- AI shot selection with physics-based force and 1-shot lookahead
class AI {
  constructor() {
    this.lastShotInfo = '';
  }

  // -- Physics helpers for force calculation --------------------------

  _distanceCoeff() {
    const muS = C.MU_SLIDE;
    const muR = C.MU_ROLL;
    const g = C.G;
    return 12 / (49 * muS * g) + 25 / (98 * muR * g);
  }

  _speedForDistance(dist) {
    return Math.sqrt(Math.max(0, dist) / this._distanceCoeff());
  }

  _distanceForSpeed(v0) {
    return v0 * v0 * this._distanceCoeff();
  }

  _speedAfterDistance(v0, dist) {
    const muS = C.MU_SLIDE;
    const muR = C.MU_ROLL;
    const g = C.G;
    const dSlide = 12 * v0 * v0 / (49 * muS * g);
    const vRoll = (5 / 7) * v0;
    if (dist <= dSlide) {
      const vSq = v0 * v0 - 2 * muS * g * dist;
      return vSq > 0 ? Math.sqrt(vSq) : 0;
    }
    const remainDist = dist - dSlide;
    const vSq = vRoll * vRoll - 2 * muR * g * remainDist;
    return vSq > 0 ? Math.sqrt(vSq) : 0;
  }

  _computeRequiredForce(cgDist, tpDist, cutAngleDeg, contactY, elevation) {
    const vObjNeeded = this._speedForDistance(tpDist) * 1.15;
    const e = C.BALL_RESTITUTION;
    const cutRad = cutAngleDeg * Math.PI / 180;
    const cosAngle = Math.max(0.15, Math.cos(cutRad));
    const transferFrac = ((1 + e) / 2) * cosAngle;
    const vCueAtImpact = vObjNeeded / transferFrac;

    // Account for spin-induced velocity loss during the cue ball's approach.
    //
    // When the cue ball has backspin (contactY < 0) and/or cue elevation,
    // the ball slides longer and transitions to rolling at a LOWER speed:
    //   v_roll = (5*v_linear - 2*w*R) / 7
    //
    // With backspin, w*R opposes the rolling direction, so the ball loses
    // significant speed at the sliding-to-rolling transition. We must compute
    // the required LAUNCH force such that after this loss, the ball still
    // arrives at the ghost position with vCueAtImpact.
    //
    // The spin ratio: w*R / v  at launch.
    // From strikeCueBall: w = (5*force*effectiveB*amplification) / (2*R^2)
    // And v = force * cos(elev), so w*R/v = (5*effectiveB*amplification) / (2*R*cos(elev))
    const R = C.BALL_RADIUS;
    const elev = elevation || 0;
    const cosE = Math.cos(elev);
    const sinE = Math.sin(elev);
    const b = contactY * R * 0.7;
    const effectiveB = b - R * sinE * 0.6;
    const spinAmp = 1 + sinE * 1.5;
    // Spin-to-speed ratio at launch: w*R / v_horizontal
    // w = (5 * force * effectiveB * spinAmp) / (2 * R^2)
    // w*R = (5 * force * effectiveB * spinAmp) / (2 * R)
    // v = force * cosE
    // ratio = (5 * effectiveB * spinAmp) / (2 * R * cosE)
    const spinRatio = (cosE > 0.01)
      ? (5 * effectiveB * spinAmp) / (2 * R * cosE)
      : 0;
    // spinRatio > 0 means topspin (helps), < 0 means backspin (hurts)

    // At the rolling transition: v_roll = (5*v + 2*spinRatio*v) / 7 = v*(5 + 2*spinRatio) / 7
    // (because w*R = spinRatio * v, and the formula is (5v - 2*wR)/7 but we use
    //  the sign convention where wy = -spinMag*aimDX, and slipX = vx + wy*R,
    //  so for backspin effectiveB < 0 -> spinMag < 0 -> wy positive -> wy*R positive
    //  -> slip is LARGER -> rolling v = (5v - 2*wy*R)/7 = (5v - 2*|spinRatio|*v)/7
    //  = v*(5 - 2*|spinRatio|)/7)
    // Actually let's be precise about signs:
    // spinRatio = 5*effectiveB*spinAmp / (2*R*cosE)
    // For backspin: effectiveB < 0 -> spinRatio < 0
    // wy = -spinMag*aimDX where spinMag = 5*force*effectiveB*spinAmp/(2R^2)
    //     = -(5*force*effectiveB*spinAmp/(2R^2))*aimDX
    // For +x motion: wy = -(negative number) = positive  (backspin)
    // slip_x = vx + wy*R = v + |spinRatio|*v = v*(1 + |spinRatio|)
    // v_roll = (5*v - 2*wy*R)/7 = (5*v - 2*|spinRatio|*v)/7 = v*(5 - 2*|spinRatio|)/7
    //
    // So: v_roll / v = (5 + 2*spinRatio) / 7  (spinRatio is negative for backspin)
    const rollFraction = Math.max(0.05, (5 + 2 * spinRatio) / 7);

    // Now compute required launch speed using spin-aware deceleration model.
    // Binary search: find force such that the ball arrives with vCueAtImpact.
    const v0 = this._speedForArrivalWithSpin(vCueAtImpact, cgDist, rollFraction);

    // Apply 1.5x multiplier and clamp
    const rawForce = v0 * 1.5;
    return Math.max(C.MIN_CUE_SPEED, Math.min(C.MAX_CUE_SPEED, rawForce));
  }

  // Speed-after-distance model accounting for spin.
  // rollFraction = v_roll / v_slide at the transition point (< 5/7 for backspin, > 5/7 for topspin)
  _speedAfterDistanceWithSpin(v0, dist, rollFraction) {
    const muS = C.MU_SLIDE;
    const muR = C.MU_ROLL;
    const g = C.G;

    // Sliding phase: ball decelerates at u_s*g
    // The slide distance depends on when the transition happens.
    // Slip decreases at rate (7/2)*u_s*g (for no-spin case).
    // With spin, the initial slip is larger but deceleration is still (7/2)*u*g.
    // Time to rolling: t_roll = initial_slip / ((7/2)*u*g)
    // initial_slip = v0 * (1 - rollFraction*7/5)... actually it's simpler to compute
    // v_roll = v0 * rollFraction, and the sliding phase distance from energy:
    // During sliding, linear deceleration = u_s*g, so:
    //   v_at_transition = v0 - u_s*g*t_roll
    // But v_at_transition = v_roll / (5/7) ... no, v_roll already accounts for the angular momentum.
    //
    // Simpler approach: the sliding phase ends when the ball reaches v_roll.
    // The ball decelerates linearly at u_s*g during sliding.
    // But the actual trajectory speed is the linear velocity (not the slip).
    // Linear decel = u_s * g (friction acts on the linear velocity)
    // The sliding ends when slip = 0, at which point v_linear = v0 * rollFraction * 7/5
    // No wait, let me use the actual formulas:
    //
    // v_linear(t) = v0 - u_s*g*t
    // w(t)*R = (spinRatio*v0) + (5/(2))*u_s*g*t   [spin accelerates toward rolling]
    // Wait, this depends on the sign. Let me just use the result:
    //   v_roll = v0 * rollFraction
    //   sliding ends when v_linear drops to a value where slip = 0
    //   v_linear_at_transition = v0 - u_s*g*t_roll
    //
    // From the Sommerfeld solution, the linear velocity at transition is NOT simply v_roll.
    // v_roll = (5v0 + 2wR)/7 accounts for angular momentum conservation.
    // During sliding, the linear velocity decreases while angular velocity increases.
    // At transition: v_roll = v0*rollFraction (by definition of rollFraction here).
    //
    // But the LINEAR velocity at the transition differs from v_roll only because
    // we defined v_roll using angular momentum conservation. Actually for the
    // Sommerfeld model: at the instant of transition, the LINEAR speed equals v_roll.
    // This is because v_roll IS the conserved combined speed.
    //
    // So: v0 decelerates to v_roll during sliding.
    // v_roll = v0 * rollFraction
    // Using v^2 = v0^2 - 2*a*d: d_slide = (v0^2 - v_roll^2) / (2*u_s*g)
    // But this isn't right either because the deceleration isn't just u_s*g on
    // the linear velocity -- the slip deceleration is (7/2)*u_s*g but the
    // linear deceleration is just u_s*g.
    //
    // OK let me just use: d_slide = (v0^2 - v_roll^2) / (2*u_s*g)
    // where v_roll = v0*rollFraction
    // This gives d_slide = v0^2*(1 - rollFraction^2) / (2*u_s*g)

    const vRoll = v0 * rollFraction;
    if (vRoll <= 0) {
      // Ball stops during sliding (extreme backspin)
      // Distance = v0^2 / (2*u_s*g)  [ball decelerates to 0]
      const dSlide = v0 * v0 / (2 * muS * g);
      return dist <= dSlide ? Math.sqrt(Math.max(0, v0 * v0 - 2 * muS * g * dist)) : 0;
    }

    const dSlide = (v0 * v0 - vRoll * vRoll) / (2 * muS * g);

    if (dist <= dSlide) {
      // Still in sliding phase
      const vSq = v0 * v0 - 2 * muS * g * dist;
      return vSq > 0 ? Math.sqrt(vSq) : 0;
    }

    // Past sliding, in rolling phase
    const remainDist = dist - dSlide;
    const vSq = vRoll * vRoll - 2 * muR * g * remainDist;
    return vSq > 0 ? Math.sqrt(vSq) : 0;
  }

  _speedForArrivalWithSpin(vTarget, dist, rollFraction) {
    let lo = vTarget;
    let hi = Math.max(vTarget + 20, vTarget * 4);
    while (this._speedAfterDistanceWithSpin(hi, dist, rollFraction) < vTarget) {
      hi *= 2;
      if (hi > C.MAX_CUE_SPEED * 5) break;
    }
    for (let i = 0; i < 30; i++) {
      const mid = (lo + hi) / 2;
      if (this._speedAfterDistanceWithSpin(mid, dist, rollFraction) < vTarget) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  _speedForArrival(vTarget, dist) {
    let lo = vTarget;
    let hi = Math.max(vTarget + 10, vTarget * 3);
    while (this._speedAfterDistance(hi, dist) < vTarget) {
      hi *= 2;
      if (hi > C.MAX_CUE_SPEED * 3) break;
    }
    for (let i = 0; i < 30; i++) {
      const mid = (lo + hi) / 2;
      if (this._speedAfterDistance(mid, dist) < vTarget) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  // -- Shot finding ---------------------------------------------------

  findBestShot(balls, table, assignedGroup, remainingTarget, cueBall) {
    if (!cueBall || cueBall.isPocketed) return null;

    let targets = balls.filter(b => {
      if (b.isPocketed || b.isCueBall) return false;
      if (assignedGroup === 'solids') return b.isSolid;
      if (assignedGroup === 'stripes') return b.isStripe;
      if (assignedGroup === 'eightball') return b.isEightBall;
      // 9-ball: target only the specific lowest ball (format: 'nine-ball-N')
      if (assignedGroup && assignedGroup.startsWith('nine-ball-')) {
        const targetId = parseInt(assignedGroup.split('-')[2]);
        return b.id === targetId;
      }
      return !b.isCueBall;
    });

    if (targets.length === 0 && assignedGroup !== 'eightball') {
      const eightBall = balls.find(b => b.id === 8 && !b.isPocketed);
      if (eightBall) targets = [eightBall];
    }

    const candidates = [];
    this._logRejections = false;
    for (const target of targets) {
      for (const pocket of table.pockets) {
        try {
          const shot = this._evaluateShot(cueBall, target, pocket, balls, table, assignedGroup);
          if (shot) candidates.push(shot);
        } catch (err) {
          console.warn(`Error evaluating ball ${target.id}->${pocket.name}:`, err.message);
        }
      }
    }

    if (candidates.length > 0) {
      candidates.sort((a, b) => b.score - a.score);
    }

    if (candidates.length === 0) {
      // No direct shot -- try combos, banks, and kicks

      let comboCandidates = [], bankCandidates = [], kickCandidates = [];
      try { comboCandidates = this._evaluateComboShots(cueBall, targets, balls, table, assignedGroup); }
      catch (err) { console.warn('Combo shot eval error:', err.message); }
      try { bankCandidates = this._evaluateBankShots(cueBall, targets, balls, table, assignedGroup); }
      catch (err) { console.warn('Bank shot eval error:', err.message); }
      try { kickCandidates = this._evaluateKickShots(cueBall, targets, balls, table, assignedGroup); }
      catch (err) { console.warn('Kick shot eval error:', err.message); }

      const allIndirect = [...comboCandidates, ...bankCandidates, ...kickCandidates];
      if (allIndirect.length > 0) {
        allIndirect.sort((a, b) => b.score - a.score);
        const best = allIndirect[0];
        const shotType = best.comboViaId !== undefined ? 'Combo' :
                         best.bankBounceX !== undefined ? 'Bank' : 'Kick';
        this.lastShotInfo = `${shotType}: ball ${best.targetId} -> ${best.pocketName} ` +
          `(force ${best.force.toFixed(0)} in/s)`;
        return best;
      }

      this.lastShotInfo = 'No clear shot -- playing safety';
      return this._playSafety(cueBall, targets, balls, table, assignedGroup);
    }

    const best = candidates[0];
    const nextInfo = best.nextBallId !== null
      ? ` | Next: ball ${best.nextBallId} -> ${best.nextPocketName}`
      : '';
    this.lastShotInfo = `Ball ${best.targetId} -> ${best.pocketName} ` +
      `(cut ${best.cutAngle.toFixed(0)}deg, force ${best.force.toFixed(0)} in/s)${nextInfo}`;
    return best;
  }

  planBreak(cueBall, balls, table) {
    const apex = balls.find(b => !b.isPocketed && !b.isCueBall &&
      b.x === Math.min(...balls.filter(bb => !bb.isPocketed && !bb.isCueBall).map(bb => bb.x)));
    if (!apex) return null;

    // Vary the aim point on the rack for different break outcomes.
    // A good break hits the head ball (apex) but the exact contact point varies:
    //   - Dead center: symmetric spread
    //   - Slight offset: asymmetric, balls favor one side -> more pocketing chances
    // Offset the aim point by up to ~0.7 ball radii to the left or right of the apex.
    const R = C.BALL_RADIUS;
    const aimOffset = (Math.random() - 0.5) * R * 1.4; // random offset in y

    const targetX = apex.x;
    const targetY = apex.y + aimOffset;

    const dx = targetX - cueBall.x;
    const dy = targetY - cueBall.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    let aimDX = dx / dist;
    let aimDY = dy / dist;

    // Vary the force slightly (85-95% of max) for a solid break
    const forceFrac = 0.85 + Math.random() * 0.10;

    // Break shot: center to slight topspin, NO english (sidespin causes swerving)
    const contactY = 0.05 + Math.random() * 0.1; // 0.05 to 0.15 topspin
    const contactX = 0; // no english on the break

    const offsetDesc = aimOffset > 0.1 ? ' (offset right)' :
                       aimOffset < -0.1 ? ' (offset left)' : ' (center)';
    this.lastShotInfo = `Break shot${offsetDesc} -- ${(forceFrac * 100).toFixed(0)}% power`;

    return {
      cueBallX: cueBall.x, cueBallY: cueBall.y,
      aimDX, aimDY,
      force: C.MAX_CUE_SPEED * forceFrac,
      contactX, contactY,
      targetId: apex.id, pocketName: 'rack', score: 100,
    };
  }

  // -- Ball-in-hand placement -----------------------------------------
  // Find the best position to place the cue ball anywhere on the table.
  // Try a grid of candidate positions, evaluate the best shot from each,
  // and pick the placement that yields the highest-scoring shot.

  findBestPlacement(balls, table, assignedGroup) {
    const R = C.BALL_RADIUS;
    const margin = R + 0.5;
    let bestPos = { x: C.HEAD_SPOT_X, y: C.HEAD_SPOT_Y };
    let bestScore = -Infinity;
    let bestShot = null;

    // Create a temporary cue ball for evaluation
    const cueBall = balls.find(b => b.id === 0);
    const origX = cueBall.x;
    const origY = cueBall.y;
    const wasPocketed = cueBall.isPocketed;
    cueBall.isPocketed = false;

    // Sample a grid of positions across the table (finer grid = better placement)
    const stepX = 4;
    const stepY = 4;
    for (let x = margin; x <= C.TABLE_LENGTH - margin; x += stepX) {
      for (let y = margin; y <= C.TABLE_WIDTH - margin; y += stepY) {
        // Check position isn't occupied by another ball
        let occupied = false;
        for (const b of balls) {
          if (b.id === 0 || b.isPocketed) continue;
          const dx = b.x - x;
          const dy = b.y - y;
          if (dx * dx + dy * dy < (2 * R + 0.5) * (2 * R + 0.5)) {
            occupied = true;
            break;
          }
        }
        if (occupied) continue;

        // Temporarily place cue ball here and evaluate
        cueBall.x = x;
        cueBall.y = y;

        // Determine target group
        let group = assignedGroup;
        if (group && group !== 'eightball' && !group.startsWith('nine-ball-')) {
          const remaining = group === 'solids'
            ? balls.filter(b => b.isSolid && !b.isPocketed).length
            : balls.filter(b => b.isStripe && !b.isPocketed).length;
          if (remaining === 0) group = 'eightball';
        }

        // Try all target/pocket combos from this position
        const targets = balls.filter(b => {
          if (b.isPocketed || b.isCueBall) return false;
          if (group === 'solids') return b.isSolid;
          if (group === 'stripes') return b.isStripe;
          if (group === 'eightball') return b.isEightBall;
          if (group && group.startsWith('nine-ball-')) {
            const nid = parseInt(group.split('-')[2]);
            return b.id === nid;
          }
          return !b.isCueBall; // open table / 14.1
        });

        for (const target of targets) {
          for (const pocket of table.pockets) {
            try {
              const shot = this._evaluateShot(cueBall, target, pocket, balls, table, group);
              if (shot) {
                let layoutBonus = this._clusterBreakBonus(
                  cueBall.x, cueBall.y, shot.ghostX, shot.ghostY,
                  shot.targetPocketX, shot.targetPocketY, balls, group);
                const totalScore = shot.score + layoutBonus;
                if (totalScore > bestScore) {
                  bestScore = totalScore;
                  bestPos = { x, y };
                  bestShot = shot;
                }
              }
            } catch (err) { /* skip this position/pocket combo */ }
          }
        }
      }
    }

    // Restore cue ball position (will be set properly by caller)
    cueBall.x = origX;
    cueBall.y = origY;
    cueBall.isPocketed = wasPocketed;

    if (bestShot) {
      console.log(`Ball-in-hand: placed at (${bestPos.x.toFixed(1)}, ${bestPos.y.toFixed(1)}) ` +
        `score=${bestScore.toFixed(1)} for ball ${bestShot.targetId}->${bestShot.pocketName}`);
    } else {
      console.log(`Ball-in-hand: NO valid position found, defaulting to head spot`);
    }

    return bestPos;
  }

  // -- Core shot evaluation -------------------------------------------

  _evaluateShot(cueBall, target, pocket, balls, table, assignedGroup) {
    const R = C.BALL_RADIUS;

    // Use the pocket's AIM point (mouth center) for ghost ball calculation,
    // not the pocket center (back corner). This ensures the ball enters
    // between the cushion noses instead of clipping them.
    const aimPX = pocket.aimX !== undefined ? pocket.aimX : pocket.x;
    const aimPY = pocket.aimY !== undefined ? pocket.aimY : pocket.y;

    // Vector from target to pocket aim point
    const tpx = aimPX - target.x;
    const tpy = aimPY - target.y;
    const tpDist = Math.sqrt(tpx * tpx + tpy * tpy);
    if (tpDist < 0.1) return null;
    const tpNX = tpx / tpDist;
    const tpNY = tpy / tpDist;

    // Ghost ball position: where cue ball center must be at contact
    // to send target ball toward the pocket mouth
    const ghostX = target.x - tpNX * 2 * R;
    const ghostY = target.y - tpNY * 2 * R;

    // Aim direction: from cue ball to ghost ball
    const cgx = ghostX - cueBall.x;
    const cgy = ghostY - cueBall.y;
    const cgDist = Math.sqrt(cgx * cgx + cgy * cgy);
    if (cgDist < 0.1) return null;
    const aimDX = cgx / cgDist;
    const aimDY = cgy / cgDist;

    // Cut angle between aim line and target-to-pocket line
    const dot = aimDX * tpNX + aimDY * tpNY;
    const cutAngle = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
    if (cutAngle > 75) {
      return null;
    }

    const clearance = 2 * R;

    // Pocket approach angle
    const pocketAngle = this._pocketApproachAngle(target, pocket);
    if (pocketAngle >= 999) {
      return null;
    }

    // Path 1: cue ball to ghost ball (check for ball obstructions)
    if (this._isPathObstructed(cueBall.x, cueBall.y, ghostX, ghostY, balls,
        [cueBall.id, target.id], clearance)) {
      return null;
    }

    // Path 1b: cue ball path must not cross any pocket opening (scratch risk)
    if (this._pathCrossesPocket(cueBall.x, cueBall.y, ghostX, ghostY, table.pockets, R)) {
      return null;
    }

    // Path 2: target ball to pocket
    if (this._isPathObstructed(target.x, target.y, pocket.x, pocket.y, balls,
        [target.id, cueBall.id], clearance)) {
      return null;
    }

    // Compute spin (determines english, draw/follow, and identifies next shot target)
    const spin = this._computeSpin(cueBall, target, pocket, ghostX, ghostY,
                                    aimDX, aimDY, cgDist, balls, table, assignedGroup);

    // Minimum force needed to pocket the object ball
    const minForce = this._computeRequiredForce(cgDist, tpDist, cutAngle, spin.cy, spin.elevation);

    // Adjust force for position play: if the desired next-shot position is far
    // from where a minimum-force shot would leave the cue ball, add extra force.
    // The cue ball's post-collision travel distance is proportional to the
    // excess speed it retains. More force -> more retained speed -> farther travel.
    let force = minForce;
    if (spin.nextGhostX !== null && spin.estCueBallX !== null) {
      // How far is the estimated position from the desired next position?
      const estToNext = Math.sqrt(
        (spin.estCueBallX - spin.nextGhostX) ** 2 +
        (spin.estCueBallY - spin.nextGhostY) ** 2);
      // If the estimate is far from the target, bump force to push the cue ball further
      // (this is an approximation -- more force = cue ball travels farther from the ghost)
      if (estToNext > 15) {
        const forceBoost = Math.min(estToNext * 0.4, 30); // up to 30 in/s extra
        force = Math.min(C.MAX_CUE_SPEED, force + forceBoost);
      }
      // If the estimate is close, we might want LESS force for a softer, more precise stop
      if (estToNext < 8 && force > minForce * 1.3) {
        force = minForce * 1.1; // just enough to pocket with minimal excess
      }
    }

    // -- Scoring --
    let score = 100;
    score -= (cutAngle / 75) ** 2 * 50;
    score -= ((cgDist + tpDist) / C.TABLE_LENGTH) * 15;
    // pocketAngle already computed and checked above (line 474)
    score -= (pocketAngle / 45) * 20;
    if (tpDist < 20) score += 10;
    if (cutAngle < 15) score += 15;
    if (force > C.MAX_CUE_SPEED * 0.8) score -= (force / C.MAX_CUE_SPEED) * 20;

    // Bonus for shots that break up clusters of own-group balls
    score += this._clusterBreakBonus(cueBall.x, cueBall.y, ghostX, ghostY,
      pocket.x, pocket.y, balls, assignedGroup);

    // 1-shot lookahead: where will the cue ball end up with the adjusted force?
    const lookahead = this._estimateCueBallPosition(
      ghostX, ghostY, aimDX, aimDY, force, cutAngle, spin.cy, cgDist, target, pocket, table.pockets);

    if (lookahead.scratchRisk) score -= 80;

    // Score based on proximity to the DESIRED next-shot position (not just center table)
    if (spin.nextGhostX !== null) {
      const distToNext = Math.sqrt(
        (lookahead.x - spin.nextGhostX) ** 2 + (lookahead.y - spin.nextGhostY) ** 2);
      score += Math.max(0, 25 - distToNext * 0.5);
    } else {
      const distToCenter = Math.sqrt(
        (lookahead.x - C.TABLE_LENGTH / 2) ** 2 + (lookahead.y - C.TABLE_WIDTH / 2) ** 2);
      score += Math.max(0, 15 - distToCenter * 0.3);
    }
    const followUpScore = this._evaluateFollowUpPosition(
      lookahead.x, lookahead.y, balls, table, assignedGroup, target.id, target);
    score += followUpScore * 0.5;
    for (const p of table.pockets) {
      const dp = Math.sqrt((lookahead.x - p.x) ** 2 + (lookahead.y - p.y) ** 2);
      if (dp < 10) score -= (10 - dp) * 8;
    }

    if (score < 0) return null; // only reject shots that are truly net negative

    return {
      cueBallX: cueBall.x, cueBallY: cueBall.y,
      aimDX, aimDY, force,
      contactX: spin.cx, contactY: spin.cy, elevation: spin.elevation,
      ghostX, ghostY,
      targetPocketX: pocket.x, targetPocketY: pocket.y,
      targetId: target.id, pocketName: pocket.name,
      score, cutAngle,
      // Next shot plan (for display)
      nextBallId: spin.nextBallId,
      nextPocketName: spin.nextPocketName,
      nextPocketX: spin.nextPocketX,
      nextPocketY: spin.nextPocketY,
      nextGhostX: spin.nextGhostX,
      nextGhostY: spin.nextGhostY,
      estCueBallX: spin.estCueBallX,
      estCueBallY: spin.estCueBallY,
    };
  }

  // -- Kick shot evaluation ------------------------------------------
  // A kick shot banks the cue ball off one rail to reach a target ball
  // that can't be reached directly. For each target, mirror the cue ball
  // position across each rail and check if the mirrored straight-line path
  // to the target is clear (excluding the rail bounce point).

  _evaluateKickShots(cueBall, targets, balls, table, assignedGroup) {
    const R = C.BALL_RADIUS;
    const clearance = 2 * R;
    const L = C.TABLE_LENGTH;
    const W = C.TABLE_WIDTH;
    const candidates = [];

    // Four rails to kick off: top (y=0), bottom (y=W), left (x=0), right (x=L)
    const rails = [
      { name: 'top',    mirrorX: (x) => x,     mirrorY: (y) => -y },
      { name: 'bottom', mirrorX: (x) => x,     mirrorY: (y) => 2 * W - y },
      { name: 'left',   mirrorX: (x) => -x,    mirrorY: (y) => y },
      { name: 'right',  mirrorX: (x) => 2 * L - x, mirrorY: (y) => y },
    ];

    for (const target of targets) {
      for (const pocket of table.pockets) {
        // Check target-to-pocket is clear (if it's not, the kick is pointless)
        const tpx = pocket.x - target.x;
        const tpy = pocket.y - target.y;
        const tpDist = Math.sqrt(tpx * tpx + tpy * tpy);
        if (tpDist < 0.1) continue;
        const tpNX = tpx / tpDist;
        const tpNY = tpy / tpDist;

        if (this._isPathObstructed(target.x, target.y, pocket.x, pocket.y, balls,
            [target.id, cueBall.id], clearance)) continue;

        // Ghost ball position
        const ghostX = target.x - tpNX * 2 * R;
        const ghostY = target.y - tpNY * 2 * R;

        for (const rail of rails) {
          // Mirror the cue ball across this rail
          const mirCueX = rail.mirrorX(cueBall.x);
          const mirCueY = rail.mirrorY(cueBall.y);

          // Straight line from mirrored cue ball to ghost ball
          const mdx = ghostX - mirCueX;
          const mdy = ghostY - mirCueY;
          const mDist = Math.sqrt(mdx * mdx + mdy * mdy);
          if (mDist < 1) continue;
          const mnx = mdx / mDist;
          const mny = mdy / mDist;

          // Find where this line crosses the rail
          let bounceX, bounceY;
          if (rail.name === 'top') {
            if (mny === 0) continue;
            const t = (0 - mirCueY) / mny;
            if (t < 0) continue;
            bounceX = mirCueX + mnx * t;
            bounceY = 0;
          } else if (rail.name === 'bottom') {
            if (mny === 0) continue;
            const t = (W - mirCueY) / mny;
            if (t < 0) continue;
            bounceX = mirCueX + mnx * t;
            bounceY = W;
          } else if (rail.name === 'left') {
            if (mnx === 0) continue;
            const t = (0 - mirCueX) / mnx;
            if (t < 0) continue;
            bounceX = 0;
            bounceY = mirCueY + mny * t;
          } else { // right
            if (mnx === 0) continue;
            const t = (L - mirCueX) / mnx;
            if (t < 0) continue;
            bounceX = L;
            bounceY = mirCueY + mny * t;
          }

          // Bounce point must be within table bounds
          if (bounceX < R || bounceX > L - R || bounceY < R || bounceY > W - R) continue;

          // Check two path segments are clear (including pocket crossings):
          if (this._isPathObstructed(cueBall.x, cueBall.y, bounceX, bounceY, balls,
              [cueBall.id], clearance)) continue;
          if (this._pathCrossesPocket(cueBall.x, cueBall.y, bounceX, bounceY, table.pockets, R)) continue;
          if (this._isPathObstructed(bounceX, bounceY, ghostX, ghostY, balls,
              [cueBall.id, target.id], clearance)) continue;
          if (this._pathCrossesPocket(bounceX, bounceY, ghostX, ghostY, table.pockets, R)) continue;

          // Cut angle at the ghost ball
          const seg2dx = ghostX - bounceX;
          const seg2dy = ghostY - bounceY;
          const seg2Len = Math.sqrt(seg2dx * seg2dx + seg2dy * seg2dy);
          if (seg2Len < 0.1) continue;
          const aimDX2 = seg2dx / seg2Len;
          const aimDY2 = seg2dy / seg2Len;
          const dot = aimDX2 * tpNX + aimDY2 * tpNY;
          const cutAngle = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
          if (cutAngle > 65) continue; // kick shots with steep cuts are very hard

          // Total distance
          const dist1 = Math.sqrt((bounceX - cueBall.x) ** 2 + (bounceY - cueBall.y) ** 2);
          const totalDist = dist1 + seg2Len;

          // Force: enough for the cue ball to travel totalDist and transfer to pocket the target
          const force = this._computeRequiredForce(totalDist, tpDist, cutAngle, 0);

          // Score (lower than direct shots -- kick shots are harder)
          let score = 50;
          score -= (cutAngle / 65) ** 2 * 25;
          score -= (totalDist / (L * 1.5)) * 15;
          if (tpDist < 25) score += 5;

          if (score < 15) continue;

          // Aim direction: from cue ball toward the bounce point
          const aimDX = (bounceX - cueBall.x) / dist1;
          const aimDY = (bounceY - cueBall.y) / dist1;

          candidates.push({
            cueBallX: cueBall.x, cueBallY: cueBall.y,
            aimDX, aimDY, force,
            contactX: 0, contactY: 0, elevation: 0,
            ghostX, ghostY,
            targetPocketX: pocket.x, targetPocketY: pocket.y,
            targetId: target.id,
            pocketName: `${rail.name} rail kick -> ${pocket.name}`,
            score, cutAngle,
            // Kick-specific: store bounce point for rendering
            kickBounceX: bounceX, kickBounceY: bounceY,
          });
        }
      }
    }

    return candidates;
  }

  // -- Combo shot evaluation ------------------------------------------
  // A combo shot: cue ball hits ball A, ball A hits ball B, ball B goes into a pocket.
  // Geometry: find the ghost position on B for the pocket, then find the ghost
  // position on A to send A toward B's ghost. The cue ball aims at A's ghost.

  _evaluateComboShots(cueBall, targets, balls, table, assignedGroup) {
    const R = C.BALL_RADIUS;
    const clearance = 2 * R;
    const candidates = [];

    // All non-pocketed object balls (ball B = the one that goes into the pocket)
    const allBalls = balls.filter(b => !b.isPocketed && !b.isCueBall);

    for (const ballB of allBalls) {
      for (const pocket of table.pockets) {
        // Check pocket approach angle for ball B
        const pa = this._pocketApproachAngle(ballB, pocket);
        if (pa >= 999) continue;

        // Use aim point for ghost ball direction
        const aPX = pocket.aimX !== undefined ? pocket.aimX : pocket.x;
        const aPY = pocket.aimY !== undefined ? pocket.aimY : pocket.y;

        // Ghost ball position for B (where A must hit B to send B into pocket)
        const bpx = aPX - ballB.x;
        const bpy = aPY - ballB.y;
        const bpDist = Math.sqrt(bpx * bpx + bpy * bpy);
        if (bpDist < 0.1) continue;
        const bpNX = bpx / bpDist;
        const bpNY = bpy / bpDist;
        const ghostBx = ballB.x - bpNX * 2 * R;
        const ghostBy = ballB.y - bpNY * 2 * R;

        // Path B to pocket must be clear
        if (this._isPathObstructed(ballB.x, ballB.y, pocket.x, pocket.y, balls,
            [ballB.id], clearance)) continue;

        // For each ball A (the one the cue ball hits first)
        for (const ballA of targets) {
          if (ballA.id === ballB.id) continue;

          // Can A reach B's ghost? Direction from A to ghost-on-B
          const agx = ghostBx - ballA.x;
          const agy = ghostBy - ballA.y;
          const agDist = Math.sqrt(agx * agx + agy * agy);
          if (agDist < 0.1 || agDist > 60) continue; // too far for reliable combo

          const agNX = agx / agDist;
          const agNY = agy / agDist;

          // Cut angle for A hitting B
          const abDot = agNX * bpNX + agNY * bpNY;
          const abCutAngle = Math.acos(Math.max(-1, Math.min(1, abDot))) * 180 / Math.PI;
          if (abCutAngle > 50) continue; // steep combo cuts are unreliable

          // Path A to ghost-on-B must be clear
          if (this._isPathObstructed(ballA.x, ballA.y, ghostBx, ghostBy, balls,
              [ballA.id, ballB.id], clearance)) continue;

          // Ghost ball position on A (where cue ball must hit A to send A toward ghost-on-B)
          const ghostAx = ballA.x - agNX * 2 * R;
          const ghostAy = ballA.y - agNY * 2 * R;

          // Cue ball to ghost-on-A
          const cgx = ghostAx - cueBall.x;
          const cgy = ghostAy - cueBall.y;
          const cgDist = Math.sqrt(cgx * cgx + cgy * cgy);
          if (cgDist < 0.1) continue;
          const aimDX = cgx / cgDist;
          const aimDY = cgy / cgDist;

          // Cut angle for cue hitting A
          const caDot = aimDX * agNX + aimDY * agNY;
          const caCutAngle = Math.acos(Math.max(-1, Math.min(1, caDot))) * 180 / Math.PI;
          if (caCutAngle > 60) continue;

          // Cue ball path to ghost-on-A must be clear and not cross pockets
          if (this._isPathObstructed(cueBall.x, cueBall.y, ghostAx, ghostAy, balls,
              [cueBall.id, ballA.id], clearance)) continue;
          if (this._pathCrossesPocket(cueBall.x, cueBall.y, ghostAx, ghostAy, table.pockets, R)) continue;

          // Total distance
          const totalDist = cgDist + agDist + bpDist;
          const force = this._computeRequiredForce(cgDist, agDist + bpDist, caCutAngle, 0);

          // Score -- combos are harder than direct shots but easier than banks
          let score = 40;
          score -= (caCutAngle / 60) ** 2 * 15;
          score -= (abCutAngle / 50) ** 2 * 15;
          score -= (totalDist / (C.TABLE_LENGTH * 1.5)) * 10;
          if (bpDist < 20) score += 8; // B is close to pocket
          if (agDist < 15) score += 5; // A is close to B

          if (score < 10) continue;

          candidates.push({
            cueBallX: cueBall.x, cueBallY: cueBall.y,
            aimDX, aimDY, force,
            contactX: 0, contactY: 0, elevation: 0,
            ghostX: ghostAx, ghostY: ghostAy,
            targetPocketX: pocket.x, targetPocketY: pocket.y,
            targetId: ballB.id,
            pocketName: `combo ${ballA.id}->${ballB.id}->${pocket.name}`,
            score, cutAngle: caCutAngle,
            comboViaId: ballA.id,
          });
        }
      }
    }
    return candidates;
  }

  // -- Bank shot evaluation ------------------------------------------
  // A bank shot hits the object ball so it bounces off one rail into a pocket.
  // For each target + pocket + rail, mirror the pocket across the rail.
  // The ghost ball is positioned to send the object ball toward the mirrored pocket.
  // The object ball travels to the rail, bounces, and enters the real pocket.

  _evaluateBankShots(cueBall, targets, balls, table, assignedGroup) {
    const R = C.BALL_RADIUS;
    const clearance = 2 * R;
    const L = C.TABLE_LENGTH;
    const W = C.TABLE_WIDTH;
    const candidates = [];

    const rails = [
      { name: 'top',    mirrorPocket: (p) => ({ x: p.x, y: -p.y }) },
      { name: 'bottom', mirrorPocket: (p) => ({ x: p.x, y: 2 * W - p.y }) },
      { name: 'left',   mirrorPocket: (p) => ({ x: -p.x, y: p.y }) },
      { name: 'right',  mirrorPocket: (p) => ({ x: 2 * L - p.x, y: p.y }) },
    ];

    for (const target of targets) {
      for (const pocket of table.pockets) {
        for (const rail of rails) {
          // Mirror the pocket AIM POINT (not the back corner) across this rail
          const aimPt = { x: pocket.aimX || pocket.x, y: pocket.aimY || pocket.y };
          const mirPocket = rail.mirrorPocket(aimPt);

          // Direction from target to mirrored aim point
          const tmx = mirPocket.x - target.x;
          const tmy = mirPocket.y - target.y;
          const tmDist = Math.sqrt(tmx * tmx + tmy * tmy);
          if (tmDist < 1) continue;
          const tmNX = tmx / tmDist;
          const tmNY = tmy / tmDist;

          // Find where the object ball's path hits the rail
          let bounceX, bounceY;
          if (rail.name === 'top') {
            if (tmNY >= 0) continue; // not heading toward top rail
            const t = -target.y / tmNY;
            bounceX = target.x + tmNX * t;
            bounceY = 0;
          } else if (rail.name === 'bottom') {
            if (tmNY <= 0) continue;
            const t = (W - target.y) / tmNY;
            bounceX = target.x + tmNX * t;
            bounceY = W;
          } else if (rail.name === 'left') {
            if (tmNX >= 0) continue;
            const t = -target.x / tmNX;
            bounceX = 0;
            bounceY = target.y + tmNY * t;
          } else {
            if (tmNX <= 0) continue;
            const t = (L - target.x) / tmNX;
            bounceX = L;
            bounceY = target.y + tmNY * t;
          }

          // Bounce point must not be near a pocket opening along the rail.
          // Only check the coordinate ALONG the rail, not perpendicular.
          if (rail.name === 'top' || rail.name === 'bottom') {
            // Horizontal rail: check x is not near corner pockets
            if (bounceX < R * 4 || bounceX > L - R * 4) continue;
            // Also not near the side pocket
            if (Math.abs(bounceX - L / 2) < R * 4) continue;
          } else {
            // Vertical rail: check y is not near corner pockets
            if (bounceY < R * 4 || bounceY > W - R * 4) continue;
          }

          // Check pocket approach angle at the REAL pocket after the bounce
          // The ball rebounds off the rail toward the pocket
          const rbx = pocket.x - bounceX;
          const rby = pocket.y - bounceY;
          const rbDist = Math.sqrt(rbx * rbx + rby * rby);
          if (rbDist < 1) continue;

          // Path 1: target ball to bounce point (clear of other balls?)
          const dist1 = Math.sqrt((bounceX - target.x) ** 2 + (bounceY - target.y) ** 2);
          if (this._isPathObstructed(target.x, target.y, bounceX, bounceY, balls,
              [target.id, cueBall.id], clearance)) continue;

          // Path 2: bounce point to pocket (clear?)
          if (this._isPathObstructed(bounceX, bounceY, pocket.x, pocket.y, balls,
              [target.id, cueBall.id], clearance)) continue;

          // Check that the rebounded ball approaches the pocket at an acceptable angle.
          // Create a fake "target" at the bounce point to check the approach angle
          // from the bounce point to the pocket (the rebound path).
          const reboundAngle = this._pocketApproachAngle(
            { x: bounceX, y: bounceY }, pocket);
          if (reboundAngle >= 999) continue; // impossible approach after rebound

          // Ghost ball position: send the object ball toward the mirrored pocket.
          // Ghost is 2R behind the target along the target->mirroredPocket direction.
          const ghostX = target.x - tmNX * 2 * R;
          const ghostY = target.y - tmNY * 2 * R;

          // Can the cue ball reach the ghost?
          const cgx = ghostX - cueBall.x;
          const cgy = ghostY - cueBall.y;
          const cgDist = Math.sqrt(cgx * cgx + cgy * cgy);
          if (cgDist < 0.1) continue;
          const aimDX = cgx / cgDist;
          const aimDY = cgy / cgDist;

          // Cut angle
          const dot = aimDX * tmNX + aimDY * tmNY;
          const cutAngle = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
          if (cutAngle > 60) continue; // bank shots with steep cuts are very hard

          // Cue ball path to ghost clear?
          if (this._isPathObstructed(cueBall.x, cueBall.y, ghostX, ghostY, balls,
              [cueBall.id, target.id], clearance)) continue;

          // Total object ball travel distance
          const totalObjDist = dist1 + rbDist;
          const force = this._computeRequiredForce(cgDist, totalObjDist, cutAngle, 0);

          // Score (lower than direct shots, comparable to kick shots)
          let score = 45;
          score -= (cutAngle / 60) ** 2 * 20;
          score -= (totalObjDist / (L * 1.5)) * 10;
          score -= (cgDist / L) * 10;
          score -= (reboundAngle / 50) * 10; // steep rebound approaches are harder
          if (rbDist < 30) score += 5;

          if (score < 10) continue;

          candidates.push({
            cueBallX: cueBall.x, cueBallY: cueBall.y,
            aimDX, aimDY, force,
            contactX: 0, contactY: 0, elevation: 0,
            ghostX, ghostY,
            targetPocketX: pocket.x, targetPocketY: pocket.y,
            targetId: target.id,
            pocketName: `${rail.name} bank -> ${pocket.name}`,
            score, cutAngle,
            // Bank-specific: store bounce point for rendering
            bankBounceX: bounceX, bankBounceY: bounceY,
          });
        }
      }
    }

    return candidates;
  }

  // -- Cue ball position estimation (lookahead) ----------------------

  _estimateCueBallPosition(ghostX, ghostY, aimDX, aimDY, force, cutAngleDeg, contactY, cgDist, target, pocket, pockets) {
    const R = C.BALL_RADIUS;
    const cutRad = cutAngleDeg * Math.PI / 180;
    const e = C.BALL_RESTITUTION;
    const vImpact = this._speedAfterDistance(force, cgDist);

    const cueRetainedSpeed = cutAngleDeg < 5
      ? vImpact * (1 - e) / 2
      : vImpact * Math.sin(cutRad);
    const cueRemainingDist = this._distanceForSpeed(cueRetainedSpeed);

    let cueEndX, cueEndY;

    if (cutAngleDeg < 5) {
      if (contactY < -0.1) {
        const drawDist = Math.min(cueRemainingDist, Math.abs(contactY) * vImpact * 0.4);
        cueEndX = ghostX - aimDX * drawDist;
        cueEndY = ghostY - aimDY * drawDist;
      } else if (contactY > 0.1) {
        const followDist = Math.min(cueRemainingDist, contactY * vImpact * 0.3);
        cueEndX = ghostX + aimDX * followDist;
        cueEndY = ghostY + aimDY * followDist;
      } else {
        cueEndX = ghostX;
        cueEndY = ghostY;
      }
    } else {
      // Cut shot: cue ball deflects roughly 90deg from aim direction for a stun shot.
      // After collision, the cue ball retains the tangential component of its velocity.
      // The deflection direction is perpendicular to the collision normal (which is
      // along the aim line for a ghost-ball hit).
      //
      // The collision normal is from ghost to target = target-to-pocket direction.
      // Tangential direction = perpendicular to this normal.
      const tpDx = pocket.x - target.x;
      const tpDy = pocket.y - target.y;
      const tpL = Math.sqrt(tpDx * tpDx + tpDy * tpDy);
      const tpNX = tpDx / (tpL || 1);
      const tpNY = tpDy / (tpL || 1);

      // The cue ball's tangential velocity after collision:
      // v_tangential = v_impact * sin(cutAngle), direction perpendicular to normal
      const sinCut = Math.sin(cutRad);
      const cosCut = Math.cos(cutRad);
      const tangentialSpeed = vImpact * sinCut;

      // Normal component retained: v_normal_retained = v_impact * cos(cut) * (1-e)/2 ~ 2%
      const normalRetained = vImpact * cosCut * (1 - e) / 2;

      // Tangential direction: perpendicular to the collision normal (tpNX, tpNY)
      // Choose the side the cue ball deflects to (away from the pocket line)
      const cross = aimDX * tpNY - aimDY * tpNX;
      const sign = cross > 0 ? 1 : -1;
      const tanDX = sign * (-tpNY);
      const tanDY = sign * tpNX;

      // Combined post-collision velocity direction
      let postVX = tanDX * tangentialSpeed + tpNX * normalRetained;
      let postVY = tanDY * tangentialSpeed + tpNY * normalRetained;

      // Spin adjustment: follow shifts the direction forward, draw shifts back
      if (contactY > 0.1) {
        // Follow: add a forward component along the aim direction
        postVX += aimDX * vImpact * contactY * 0.3;
        postVY += aimDY * vImpact * contactY * 0.3;
      } else if (contactY < -0.1) {
        // Draw: subtract forward component (slows/reverses along aim)
        postVX += aimDX * vImpact * contactY * 0.2;
        postVY += aimDY * vImpact * contactY * 0.2;
      }

      const postSpeed = Math.sqrt(postVX * postVX + postVY * postVY);
      if (postSpeed > 0.1) {
        const travelDist = this._distanceForSpeed(postSpeed) * 0.7; // friction reduces travel
        cueEndX = ghostX + (postVX / postSpeed) * travelDist;
        cueEndY = ghostY + (postVY / postSpeed) * travelDist;
      } else {
        cueEndX = ghostX;
        cueEndY = ghostY;
      }
    }

    cueEndX = Math.max(R, Math.min(C.TABLE_LENGTH - R, cueEndX));
    cueEndY = Math.max(R, Math.min(C.TABLE_WIDTH - R, cueEndY));

    // Check if the cue ball's post-collision path passes through any pocket (scratch risk).
    // Test the line from the ghost ball position to the estimated end position.
    let scratchRisk = false;
    for (const p of (pockets || [])) {
      // Distance from pocket center to the line ghostBall->cueEnd
      const pathDx = cueEndX - ghostX;
      const pathDy = cueEndY - ghostY;
      const pathLen = Math.sqrt(pathDx * pathDx + pathDy * pathDy);
      if (pathLen < 0.1) continue;
      const pnx = pathDx / pathLen;
      const pny = pathDy / pathLen;
      const toP_x = p.x - ghostX;
      const toP_y = p.y - ghostY;
      const proj = toP_x * pnx + toP_y * pny;
      if (proj < 0 || proj > pathLen) continue;
      const perpDist = Math.abs(-toP_x * pny + toP_y * pnx);
      if (perpDist < p.radius + R) {
        scratchRisk = true;
        break;
      }
    }

    return { x: cueEndX, y: cueEndY, scratchRisk };
  }

  // Score follow-up position
  _evaluateFollowUpPosition(px, py, balls, table, assignedGroup, justPocketedId, currentTarget) {
    const R = C.BALL_RADIUS;
    let bestFollowUp = 0;
    const clearance = 2 * R;

    // Determine follow-up target group
    let group = assignedGroup;
    const isNineBall = group && group.startsWith('nine-ball-');

    if (isNineBall) {
      // 9-ball: next target is the next-lowest ball after the just-pocketed one
      let nextLowestId = 99;
      for (const b of balls) {
        if (b.isPocketed || b.isCueBall || b.id === justPocketedId) continue;
        if (b.id < nextLowestId) nextLowestId = b.id;
      }
      group = nextLowestId < 99 ? 'nine-ball-' + nextLowestId : null;
    } else if (!group && currentTarget) {
      if (currentTarget.isSolid) group = 'solids';
      else if (currentTarget.isStripe) group = 'stripes';
    }

    const targets = balls.filter(b => {
      if (b.isPocketed || b.isCueBall || b.id === justPocketedId) return false;
      if (group === 'solids') return b.isSolid;
      if (group === 'stripes') return b.isStripe;
      if (group === 'eightball') return b.isEightBall;
      if (group && group.startsWith('nine-ball-')) {
        const nid = parseInt(group.split('-')[2]);
        return b.id === nid;
      }
      return false;
    });

    for (const target of targets) {
      for (const pocket of table.pockets) {
        const tpx = pocket.x - target.x;
        const tpy = pocket.y - target.y;
        const tpDist = Math.sqrt(tpx * tpx + tpy * tpy);
        if (tpDist < 0.1) continue;
        const tpNX = tpx / tpDist;
        const tpNY = tpy / tpDist;
        const ghostX = target.x - tpNX * 2 * R;
        const ghostY = target.y - tpNY * 2 * R;
        const cgx = ghostX - px;
        const cgy = ghostY - py;
        const cgDist = Math.sqrt(cgx * cgx + cgy * cgy);
        if (cgDist < 0.1) continue;
        const aDX = cgx / cgDist;
        const aDY = cgy / cgDist;
        const dot = aDX * tpNX + aDY * tpNY;
        const cutAngle = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
        if (cutAngle > 70) continue;
        if (this._isPathObstructed(target.x, target.y, pocket.x, pocket.y, balls,
            [target.id, justPocketedId], clearance)) continue;
        let fScore = 30;
        fScore -= (cutAngle / 70) * 15;
        fScore -= ((cgDist + tpDist) / C.TABLE_LENGTH) * 10;
        if (tpDist < 20) fScore += 5;
        if (cutAngle < 20) fScore += 5;
        bestFollowUp = Math.max(bestFollowUp, fScore);
      }
    }
    return bestFollowUp;
  }

  // -- Spin selection -------------------------------------------------
  // The key idea: identify the NEXT shot (best target ball + pocket after this one),
  // determine the ideal cue ball position for that next shot (the ghost ball position),
  // and pick the spin that puts the cue ball closest to that ideal position while
  // avoiding scratches.

  _computeSpin(cueBall, target, pocket, ghostX, ghostY, aimDX, aimDY, cgDist, balls, table, assignedGroup) {
    const R = C.BALL_RADIUS;
    const clearance = 2 * R;
    const tpDx = pocket.x - target.x;
    const tpDy = pocket.y - target.y;
    const tpL = Math.sqrt(tpDx * tpDx + tpDy * tpDy);
    const cutRad = Math.acos(Math.max(-1, Math.min(1,
      aimDX * tpDx / (tpL || 1) + aimDY * tpDy / (tpL || 1))));
    const cutAngleDeg = cutRad * 180 / Math.PI;

    // Step 1: Find the best NEXT shot from the remaining balls.
    // This tells us where the cue ball ideally needs to end up.
    // Determine the next target based on game mode:
    let nextGroup = assignedGroup;
    const isNineBall = assignedGroup && assignedGroup.startsWith('nine-ball-');

    if (isNineBall) {
      // 9-ball: next target is the next-lowest ball after the current target
      // Find the lowest-numbered ball on the table excluding the current target
      let nextLowestId = 99;
      for (const b of balls) {
        if (b.isPocketed || b.isCueBall || b.id === target.id) continue;
        if (b.id < nextLowestId) nextLowestId = b.id;
      }
      nextGroup = nextLowestId < 99 ? 'nine-ball-' + nextLowestId : null;
    } else if (!nextGroup || nextGroup === null) {
      // Open table (8-ball): infer group from the current target ball
      if (target.isSolid) nextGroup = 'solids';
      else if (target.isStripe) nextGroup = 'stripes';
    }

    const nextTargets = balls.filter(b => {
      if (b.isPocketed || b.isCueBall || b.id === target.id) return false;
      if (nextGroup === 'solids') return b.isSolid;
      if (nextGroup === 'stripes') return b.isStripe;
      if (nextGroup === 'eightball') return b.isEightBall;
      // 9-ball: match specific ball ID
      if (nextGroup && nextGroup.startsWith('nine-ball-')) {
        const nid = parseInt(nextGroup.split('-')[2]);
        return b.id === nid;
      }
      return false;
    });

    // 8-ball: if current target is last in group, next is the 8-ball
    if (nextTargets.length === 0 && !isNineBall) {
      const eight = balls.find(b => b.id === 8 && !b.isPocketed && b.id !== target.id);
      if (eight) nextTargets.push(eight);
    }

    // Find the best next shot (target + pocket) and its ideal cue ball position
    let bestNextGhostX = C.TABLE_LENGTH / 2;
    let bestNextGhostY = C.TABLE_WIDTH / 2;
    let bestNextScore = -Infinity;
    let bestNextBallId = null;
    let bestNextPocketName = null;
    let bestNextPocketX = null;
    let bestNextPocketY = null;

    // First, estimate roughly where the cue ball will end up for a center-hit
    // stun shot (baseline). This helps evaluate which next shots are reachable.
    const baseEst = this._estimateCueBallPosition(
      ghostX, ghostY, aimDX, aimDY, 60, cutAngleDeg, 0, cgDist, target, pocket, table.pockets);
    const estX = baseEst.x;
    const estY = baseEst.y;

    for (const nt of nextTargets) {
      for (const np of table.pockets) {
        // Use aim point for ghost ball direction
        const npAimX = np.aimX !== undefined ? np.aimX : np.x;
        const npAimY = np.aimY !== undefined ? np.aimY : np.y;
        const ntpx = npAimX - nt.x;
        const ntpy = npAimY - nt.y;
        const ntpDist = Math.sqrt(ntpx * ntpx + ntpy * ntpy);
        if (ntpDist < 0.1) continue;
        const ntpNX = ntpx / ntpDist;
        const ntpNY = ntpy / ntpDist;

        // Ghost ball position for the next shot
        const ngX = nt.x - ntpNX * 2 * R;
        const ngY = nt.y - ntpNY * 2 * R;

        // Is the next shot's target-to-pocket path clear?
        if (this._isPathObstructed(nt.x, nt.y, np.x, np.y, balls,
            [nt.id, target.id], clearance)) continue;

        // Check if the cue ball can reach the ghost from the estimated position
        const cgNextX = ngX - estX;
        const cgNextY = ngY - estY;
        const cgNextDist = Math.sqrt(cgNextX * cgNextX + cgNextY * cgNextY);
        if (cgNextDist < 0.1) continue;

        // Is the path from estimated cue ball to next ghost clear?
        if (this._isPathObstructed(estX, estY, ngX, ngY, balls,
            [0, nt.id, target.id], clearance)) continue;

        // Cut angle from the estimated cue ball position
        const nextAimDX = cgNextX / cgNextDist;
        const nextAimDY = cgNextY / cgNextDist;
        const nextDot = nextAimDX * ntpNX + nextAimDY * ntpNY;
        const nextCutAngle = Math.acos(Math.max(-1, Math.min(1, nextDot))) * 180 / Math.PI;
        if (nextCutAngle > 70) continue; // not makeable from this position

        // Score: a complete evaluation of the next shot's viability
        let ns = 40;
        // Cut angle penalty (steep cuts are hard)
        ns -= (nextCutAngle / 70) * 20;
        // Distance penalty (long shots are harder)
        ns -= ((cgNextDist + ntpDist) / C.TABLE_LENGTH) * 10;
        // Short target-to-pocket bonus
        if (ntpDist < 25) ns += 5;
        // Straight shot bonus
        if (nextCutAngle < 20) ns += 8;
        // Ghost position accessibility: prefer positions not jammed against rails
        if (ngX < 5 || ngX > C.TABLE_LENGTH - 5) ns -= 5;
        if (ngY < 5 || ngY > C.TABLE_WIDTH - 5) ns -= 5;

        if (ns > bestNextScore) {
          bestNextScore = ns;
          bestNextGhostX = ngX;
          bestNextGhostY = ngY;
          bestNextBallId = nt.id;
          bestNextPocketName = np.name;
          bestNextPocketX = np.x;
          bestNextPocketY = np.y;
        }
      }
    }

    // Fallback: if no next shot passed the strict evaluation (path from estimated
    // cue ball position was always blocked or cut angle too steep), do a relaxed pass
    // that only requires the target-to-pocket path to be clear. The cue ball position
    // estimate is rough, so we shouldn't let it prevent showing a next-shot plan.
    if (bestNextBallId === null) {
      for (const nt of nextTargets) {
        for (const np of table.pockets) {
          const ntpx = np.x - nt.x;
          const ntpy = np.y - nt.y;
          const ntpDist = Math.sqrt(ntpx * ntpx + ntpy * ntpy);
          if (ntpDist < 0.1) continue;
          const ntpNX = ntpx / ntpDist;
          const ntpNY = ntpy / ntpDist;
          const ngX = nt.x - ntpNX * 2 * R;
          const ngY = nt.y - ntpNY * 2 * R;

          // Only check target-to-pocket (not cue-to-ghost, since we don't know
          // where the cue ball will actually end up with spin)
          if (this._isPathObstructed(nt.x, nt.y, np.x, np.y, balls,
              [nt.id, target.id], clearance)) continue;

          // Check pocket approach angle
          const pa = this._pocketApproachAngle(nt, np);
          if (pa >= 999) continue;

          let ns = 25;
          ns -= (ntpDist / C.TABLE_LENGTH) * 10;
          ns -= (pa / 45) * 5;
          if (ntpDist < 25) ns += 5;
          // Prefer balls closer to the ghost position (more reachable)
          const distToGhost = Math.sqrt((ngX - ghostX) ** 2 + (ngY - ghostY) ** 2);
          ns -= (distToGhost / C.TABLE_LENGTH) * 5;

          if (ns > bestNextScore) {
            bestNextScore = ns;
            bestNextGhostX = ngX;
            bestNextGhostY = ngY;
            bestNextBallId = nt.id;
            bestNextPocketName = np.name;
            bestNextPocketX = np.x;
            bestNextPocketY = np.y;
          }
        }
      }
    }

    // Step 2: For each spin option, estimate where the cue ball ends up
    // and score by how close it gets to the ideal next-shot position.
    const DEG = Math.PI / 180;
    // Realistic spin options -- moderate draw/follow, no extreme mass_
    const spinOptions = [
      { cx: 0, cy:  0,    elev: 0 },          // stun (center hit)
      { cx: 0, cy:  0.25, elev: 0 },          // light follow
      { cx: 0, cy:  0.4,  elev: 0 },          // follow
      { cx: 0, cy: -0.25, elev: 3 * DEG },    // light draw
      { cx: 0, cy: -0.4,  elev: 5 * DEG },    // draw
      { cx: 0, cy: -0.55, elev: 8 * DEG },    // strong draw
      { cx:  0.2, cy: 0,  elev: 0 },          // right english
      { cx: -0.2, cy: 0,  elev: 0 },          // left english
      { cx:  0.15, cy: 0.2, elev: 0 },        // follow + right
      { cx: -0.15, cy: 0.2, elev: 0 },        // follow + left
      { cx:  0.15, cy: -0.2, elev: 3 * DEG }, // draw + right
      { cx: -0.15, cy: -0.2, elev: 3 * DEG }, // draw + left
    ];

    let bestSpin = { cx: 0, cy: -0.35, elev: 5 * DEG }; // default to draw (safest)
    let bestScore = -Infinity;
    let bestEst = null;

    for (const opt of spinOptions) {
      const est = this._estimateCueBallPosition(
        ghostX, ghostY, aimDX, aimDY, 60, cutAngleDeg, opt.cy, cgDist, target, pocket, table.pockets);

      // Absolutely reject any spin that risks a scratch
      if (est.scratchRisk) continue;

      let score = 0;

      // Primary criterion: distance to the ideal next-shot position
      const dNext = Math.sqrt((est.x - bestNextGhostX) ** 2 + (est.y - bestNextGhostY) ** 2);
      score += Math.max(0, 50 - dNext * 0.8); // up to 50 points for being on the ideal spot

      // Heavy penalty for ending near any pocket (scratch danger)
      for (const p of table.pockets) {
        const dp = Math.sqrt((est.x - p.x) ** 2 + (est.y - p.y) ** 2);
        if (dp < 10) score -= (10 - dp) * 10; // up to 100 point penalty
      }

      // Penalty for cushion proximity (limits shot options)
      if (est.x < 4 || est.x > C.TABLE_LENGTH - 4) score -= 10;
      if (est.y < 4 || est.y > C.TABLE_WIDTH - 4) score -= 10;

      // Small bonus for center-table (versatile position)
      const dc = Math.sqrt((est.x - C.TABLE_LENGTH / 2) ** 2 + (est.y - C.TABLE_WIDTH / 2) ** 2);
      score += Math.max(0, 10 - dc * 0.15);

      if (score > bestScore) { bestScore = score; bestSpin = opt; bestEst = est; }
    }

    console.log(`Spin selection: next ball #${bestNextBallId}->${bestNextPocketName} ` +
      `ideal=(${bestNextGhostX.toFixed(1)},${bestNextGhostY.toFixed(1)}) ` +
      `chose cy=${bestSpin.cy} elev=${((bestSpin.elev||0)*180/Math.PI).toFixed(0)}deg`);

    return {
      cx: bestSpin.cx, cy: bestSpin.cy, elevation: bestSpin.elev || 0,
      nextBallId: bestNextBallId,
      nextPocketName: bestNextPocketName,
      nextPocketX: bestNextPocketX,
      nextPocketY: bestNextPocketY,
      nextGhostX: bestNextGhostX,
      nextGhostY: bestNextGhostY,
      estCueBallX: bestEst ? bestEst.x : null,
      estCueBallY: bestEst ? bestEst.y : null,
    };
  }

  // -- Path obstruction check -----------------------------------------

  // Check if a ball traveling from (x1,y1) to (x2,y2) would fall into any pocket.
  // The ball has radius ballR; if its center passes within (pocket.radius) of a
  // pocket center, it would be pocketed (scratch).
  _pathCrossesPocket(x1, y1, x2, y2, pockets, ballR) {
    if (!pockets) return false;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.1) return false;
    const nx = dx / len;
    const ny = dy / len;

    for (const p of pockets) {
      const bx = p.x - x1;
      const by = p.y - y1;
      const proj = bx * nx + by * ny;
      // Only check within the segment (not before start or after end)
      if (proj < ballR || proj > len - ballR) continue;
      const perpDist = Math.abs(-bx * ny + by * nx);
      // The pocket "catches" the ball if the ball center passes within pocket radius
      if (perpDist < p.radius * 0.8) return true;
    }
    return false;
  }

  _isPathObstructed(x1, y1, x2, y2, balls, excludeIds, minClearance) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.1) return false;
    const nx = dx / len;
    const ny = dy / len;

    for (const ball of balls) {
      if (ball.isPocketed || excludeIds.includes(ball.id)) continue;

      const bx = ball.x - x1;
      const by = ball.y - y1;
      const proj = bx * nx + by * ny;
      const perpDist = Math.abs(-bx * ny + by * nx);

      // Ball center within segment range: check perpendicular distance
      if (proj >= 0 && proj <= len) {
        if (perpDist < minClearance) return true;
      }
      // Ball center near an endpoint: check actual distance to endpoint
      else if (proj > -minClearance && proj < 0) {
        const d = Math.sqrt(bx * bx + by * by);
        if (d < minClearance) return true;
      }
      else if (proj > len && proj < len + minClearance) {
        const ex = ball.x - x2;
        const ey = ball.y - y2;
        const d = Math.sqrt(ex * ex + ey * ey);
        if (d < minClearance) return true;
      }
    }
    return false;
  }

  // -- Safety shot ----------------------------------------------------
  // When no pocketing shot passes the main evaluation (often due to high
  // cut angles or score thresholds), try again with relaxed scoring but
  // STRICT obstruction checks. If still nothing, just nudge the nearest
  // reachable ball toward open table without claiming any pocket target.

  _playSafety(cueBall, targets, balls, table, assignedGroup) {
    const R = C.BALL_RADIUS;
    const L = C.TABLE_LENGTH;
    const W = C.TABLE_WIDTH;
    const clearance = 2 * R;
    let bestShot = null;
    let bestScore = -Infinity;

    const allTargets = targets.length > 0
      ? targets
      : balls.filter(b => !b.isPocketed && !b.isCueBall);

    // Pass 1: find a target/pocket with COMPLETELY clear paths
    for (const target of allTargets) {
      for (const pocket of table.pockets) {
        const aPX = pocket.aimX !== undefined ? pocket.aimX : pocket.x;
        const aPY = pocket.aimY !== undefined ? pocket.aimY : pocket.y;
        const tpx = aPX - target.x;
        const tpy = aPY - target.y;
        const tpDist = Math.sqrt(tpx * tpx + tpy * tpy);
        if (tpDist < 0.1) continue;
        const tpNX = tpx / tpDist;
        const tpNY = tpy / tpDist;

        const ghostX = target.x - tpNX * 2 * R;
        const ghostY = target.y - tpNY * 2 * R;

        const cgx = ghostX - cueBall.x;
        const cgy = ghostY - cueBall.y;
        const cgDist = Math.sqrt(cgx * cgx + cgy * cgy);
        if (cgDist < 0.1) continue;
        const aimDX = cgx / cgDist;
        const aimDY = cgy / cgDist;

        const dot = aimDX * tpNX + aimDY * tpNY;
        const cutAngle = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
        if (cutAngle > 80) continue; // slightly more lenient than normal

        // Check pocket approach angle
        const pAngle = this._pocketApproachAngle(target, pocket);
        if (pAngle >= 999) continue;

        // Both paths must be completely clear, and cue ball must not cross a pocket
        if (this._isPathObstructed(cueBall.x, cueBall.y, ghostX, ghostY, balls,
            [cueBall.id, target.id], clearance)) continue;
        if (this._pathCrossesPocket(cueBall.x, cueBall.y, ghostX, ghostY, table.pockets, R)) continue;
        if (this._isPathObstructed(target.x, target.y, pocket.x, pocket.y, balls,
            [target.id, cueBall.id], clearance)) continue;

        // Compute spin and next-shot plan
        const spin = this._computeSpin(cueBall, target, pocket, ghostX, ghostY,
                                        aimDX, aimDY, cgDist, balls, table, assignedGroup);
        const force = this._computeRequiredForce(cgDist, tpDist, cutAngle, spin.cy, spin.elevation);

        let score = 40;
        score -= (cutAngle / 80) * 20;
        score -= (pAngle / 45) * 10;
        score -= (cgDist / C.TABLE_LENGTH) * 10;
        if (tpDist < 30) score += 5;

        if (score > bestScore) {
          bestScore = score;
          bestShot = {
            cueBallX: cueBall.x, cueBallY: cueBall.y,
            aimDX, aimDY, force,
            contactX: spin.cx, contactY: spin.cy, elevation: spin.elevation,
            ghostX, ghostY,
            targetPocketX: pocket.x, targetPocketY: pocket.y,
            targetId: target.id,
            pocketName: pocket.name,
            score, cutAngle,
            nextBallId: spin.nextBallId,
            nextPocketName: spin.nextPocketName,
            nextPocketX: spin.nextPocketX,
            nextPocketY: spin.nextPocketY,
            nextGhostX: spin.nextGhostX,
            nextGhostY: spin.nextGhostY,
            estCueBallX: spin.estCueBallX,
            estCueBallY: spin.estCueBallY,
          };
        }
      }
    }

    if (bestShot) {
      this.lastShotInfo = `Safety pocket: ball ${bestShot.targetId} -> ${bestShot.pocketName}`;
      return bestShot;
    }

    // Pass 2: No clear pocket shot. Play smart defense.
    // Try multiple approach angles per target to find thin cuts that leave
    // the cue ball in a difficult position for the opponent.
    // CRITICAL: force must be high enough that a ball reaches a rail after contact.
    let bestDef = null;
    let bestDefScore = -Infinity;

    for (const target of allTargets) {
      const tdx = target.x - cueBall.x;
      const tdy = target.y - cueBall.y;
      const directDist = Math.sqrt(tdx * tdx + tdy * tdy);
      if (directDist < 0.5) continue;

      // Try several approach angles: head-on plus thin cuts left and right
      const directAngle = Math.atan2(tdy, tdx);
      const offsets = [0, -25, 25, -45, 45, -60, 60];

      for (const degOff of offsets) {
        const aimAngle = directAngle + degOff * Math.PI / 180;
        const aimDX = Math.cos(aimAngle);
        const aimDY = Math.sin(aimAngle);

        // Does this aim ray contact the target ball?
        const toCX = target.x - cueBall.x;
        const toCY = target.y - cueBall.y;
        const proj = toCX * aimDX + toCY * aimDY;
        if (proj < 0) continue;
        const perpDist = Math.abs(-toCX * aimDY + toCY * aimDX);
        if (perpDist > 2 * R) continue; // misses

        // Ghost ball position for this approach
        const contactOff = Math.sqrt(Math.max(0, 4 * R * R - perpDist * perpDist));
        const ghostDist = proj - contactOff;
        if (ghostDist < 0) continue;
        const ghostX = cueBall.x + aimDX * ghostDist;
        const ghostY = cueBall.y + aimDY * ghostDist;

        // Path clear to ghost? Also check pocket crossing and extended path.
        if (this._isPathObstructed(cueBall.x, cueBall.y, ghostX, ghostY, balls,
            [cueBall.id, target.id], clearance)) continue;
        if (this._pathCrossesPocket(cueBall.x, cueBall.y, ghostX, ghostY, table.pockets, R)) continue;
        if (this._isPathObstructed(cueBall.x, cueBall.y, target.x, target.y, balls,
            [cueBall.id, target.id], clearance)) continue;

        const cutAngle = Math.asin(Math.min(1, perpDist / (2 * R))) * 180 / Math.PI;

        // Force: must reach the ball AND ensure at least one ball hits a rail.
        // The target ball gets deflected; the cue ball continues. At minimum
        // one of them must travel far enough to reach the nearest rail.
        // Max distance to a rail from any point ~ TABLE_WIDTH/2 = 25".
        const minTravelAfterContact = 30; // inches -- guarantees rail contact
        const forceNeeded = this._speedForDistance(ghostDist + minTravelAfterContact) * 1.4;
        const force = Math.max(C.MIN_CUE_SPEED * 2, Math.min(C.MAX_CUE_SPEED * 0.55, forceNeeded));

        // Estimate cue ball end position
        const est = this._estimateCueBallPosition(
          ghostX, ghostY, aimDX, aimDY, force, cutAngle, 0, ghostDist,
          target, { x: ghostX + aimDX * 20, y: ghostY + aimDY * 20 }, table.pockets);
        if (est.scratchRisk) continue;

        let score = 20;

        // Bonus: cue ball near a cushion (harder for opponent to reach)
        const minCushX = Math.min(est.x, C.TABLE_LENGTH - est.x);
        const minCushY = Math.min(est.y, C.TABLE_WIDTH - est.y);
        if (minCushX < 8) score += 3;
        if (minCushY < 8) score += 3;

        // Bonus: cue ball hidden behind other balls (snooker)
        let snookerBonus = 0;
        for (const ob of balls) {
          if (ob.isPocketed || ob.isCueBall || ob.id === target.id) continue;
          for (const p of table.pockets) {
            const ex = p.x - est.x, ey = p.y - est.y;
            const el = Math.sqrt(ex * ex + ey * ey);
            if (el < 1) continue;
            const ox = ob.x - est.x, oy = ob.y - est.y;
            const oproj = (ox * ex + oy * ey) / el;
            if (oproj < 0 || oproj > el) continue;
            const operp = Math.abs((-ox * ey + oy * ex) / el);
            if (operp < 3 * R) snookerBonus += 2;
          }
        }
        score += Math.min(snookerBonus, 12);

        // Penalty: cue ball near a pocket
        for (const p of table.pockets) {
          const dp = Math.sqrt((est.x - p.x) ** 2 + (est.y - p.y) ** 2);
          if (dp < 10) score -= (10 - dp) * 5;
        }

        // Prefer thin cuts (cue ball deflects to the side)
        if (cutAngle > 15 && cutAngle < 55) score += 4;

        if (score > bestDefScore) {
          bestDefScore = score;
          bestDef = {
            cueBallX: cueBall.x, cueBallY: cueBall.y,
            aimDX, aimDY, force,
            contactX: 0, contactY: 0, elevation: 0,
            ghostX, ghostY,
            targetId: target.id,
            pocketName: 'defensive',
            score, cutAngle,
          };
        }
      }
    }

    if (bestDef) {
      this.lastShotInfo = `Defensive: ball ${bestDef.targetId} ` +
        `(cut ${bestDef.cutAngle.toFixed(0)}deg, force ${bestDef.force.toFixed(0)} in/s)`;
      return bestDef;
    }

    // Fallback: no direct defensive path found. Try a kick shot to reach a target
    // ball via one rail. This is better than shooting through a blocking ball.
    let kickDef = null;
    let bestKickScore = -Infinity;

    for (const target of allTargets) {
      // Try kicking off each rail to reach this target
      const kickRails = [
        { name: 'top',    mx: (x) => x,       my: (y) => -y },
        { name: 'bottom', mx: (x) => x,       my: (y) => 2 * W - y },
        { name: 'left',   mx: (x) => -x,      my: (y) => y },
        { name: 'right',  mx: (x) => 2 * L - x, my: (y) => y },
      ];

      for (const rail of kickRails) {
        // Mirror the cue ball across the rail
        const mirX = rail.mx(cueBall.x);
        const mirY = rail.my(cueBall.y);

        // Line from mirrored cue ball to target
        const mdx = target.x - mirX;
        const mdy = target.y - mirY;
        const mDist = Math.sqrt(mdx * mdx + mdy * mdy);
        if (mDist < 1) continue;
        const mnx = mdx / mDist;
        const mny = mdy / mDist;

        // Find bounce point on the rail
        let bounceX, bounceY;
        if (rail.name === 'top') {
          if (mny === 0) continue;
          const t = -mirY / mny;
          if (t < 0) continue;
          bounceX = mirX + mnx * t; bounceY = 0;
        } else if (rail.name === 'bottom') {
          if (mny === 0) continue;
          const t = (W - mirY) / mny;
          if (t < 0) continue;
          bounceX = mirX + mnx * t; bounceY = W;
        } else if (rail.name === 'left') {
          if (mnx === 0) continue;
          const t = -mirX / mnx;
          if (t < 0) continue;
          bounceX = 0; bounceY = mirY + mny * t;
        } else {
          if (mnx === 0) continue;
          const t = (L - mirX) / mnx;
          if (t < 0) continue;
          bounceX = L; bounceY = mirY + mny * t;
        }

        // Bounce in bounds?
        if (bounceX < R * 3 || bounceX > L - R * 3) {
          if (rail.name === 'top' || rail.name === 'bottom') continue;
        }
        if (bounceY < R * 3 || bounceY > W - R * 3) {
          if (rail.name === 'left' || rail.name === 'right') continue;
        }

        // Both segments clear (including pocket crossings)?
        const d1 = Math.sqrt((bounceX - cueBall.x) ** 2 + (bounceY - cueBall.y) ** 2);
        if (this._isPathObstructed(cueBall.x, cueBall.y, bounceX, bounceY, balls,
            [cueBall.id], clearance)) continue;
        if (this._pathCrossesPocket(cueBall.x, cueBall.y, bounceX, bounceY, table.pockets, R)) continue;
        if (this._isPathObstructed(bounceX, bounceY, target.x, target.y, balls,
            [cueBall.id, target.id], clearance)) continue;
        if (this._pathCrossesPocket(bounceX, bounceY, target.x, target.y, table.pockets, R)) continue;

        const totalDist = d1 + Math.sqrt((target.x - bounceX) ** 2 + (target.y - bounceY) ** 2);
        const force = Math.max(C.MIN_CUE_SPEED * 2,
          this._speedForDistance(totalDist + 30) * 1.4);

        const aimDX = (bounceX - cueBall.x) / d1;
        const aimDY = (bounceY - cueBall.y) / d1;

        let score = 15;
        score -= (totalDist / (L * 2)) * 10;

        if (score > bestKickScore) {
          bestKickScore = score;
          kickDef = {
            cueBallX: cueBall.x, cueBallY: cueBall.y,
            aimDX, aimDY,
            force: Math.min(force, C.MAX_CUE_SPEED * 0.6),
            contactX: 0, contactY: 0, elevation: 0,
            targetId: target.id,
            pocketName: `${rail.name} rail kick (defensive)`,
            score, cutAngle: 0,
            kickBounceX: bounceX, kickBounceY: bounceY,
          };
        }
      }
    }

    if (kickDef) {
      this.lastShotInfo = `Defensive kick: ball ${kickDef.targetId} via ${kickDef.pocketName}`;
      return kickDef;
    }

    // True last resort: find the nearest ball with a clear direct path
    let nearest = null;
    let nearestDist = Infinity;
    for (const t of allTargets) {
      const d = Math.sqrt((t.x - cueBall.x) ** 2 + (t.y - cueBall.y) ** 2);
      if (d > 0.5 && !this._isPathObstructed(cueBall.x, cueBall.y, t.x, t.y, balls,
          [cueBall.id, t.id], clearance)) {
        if (d < nearestDist) { nearestDist = d; nearest = t; }
      }
    }
    // If still nothing with clear path, just pick closest (rare edge case)
    if (!nearest) {
      for (const t of allTargets) {
        const d = Math.sqrt((t.x - cueBall.x) ** 2 + (t.y - cueBall.y) ** 2);
        if (d < nearestDist) { nearestDist = d; nearest = t; }
      }
    }
    if (!nearest) return null;
    const fdx = nearest.x - cueBall.x;
    const fdy = nearest.y - cueBall.y;
    const fdist = Math.sqrt(fdx * fdx + fdy * fdy);
    const fallbackForce = Math.max(C.MIN_CUE_SPEED * 2,
      this._speedForDistance(fdist + 40) * 1.4);
    this.lastShotInfo = `Defensive: ball ${nearest.id}`;
    return {
      cueBallX: cueBall.x, cueBallY: cueBall.y,
      aimDX: fdx / fdist, aimDY: fdy / fdist,
      force: Math.min(fallbackForce, C.MAX_CUE_SPEED * 0.5),
      contactX: 0, contactY: 0, elevation: 0,
      targetId: nearest.id,
      pocketName: 'defensive',
      score: 1, cutAngle: 0,
    };
  }

  // Count how many balls obstruct a path (instead of just boolean)
  _countObstructions(x1, y1, x2, y2, balls, excludeIds, minClearance) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.1) return 0;
    const nx = dx / len;
    const ny = dy / len;
    let count = 0;
    for (const ball of balls) {
      if (ball.isPocketed || excludeIds.includes(ball.id)) continue;
      const bx = ball.x - x1;
      const by = ball.y - y1;
      const proj = bx * nx + by * ny;
      if (proj < 0 || proj > len) continue;
      const perpDist = Math.abs(-bx * ny + by * nx);
      if (perpDist < minClearance) count++;
    }
    return count;
  }

  // Evaluate whether a shot path passes near clustered own-group balls,
  // potentially breaking them apart. Returns a bonus score (0 to ~15).
  // Clusters are groups of 2+ own-group balls within 3 ball diameters of each other
  // that are NOT near a pocket (i.e., they're "stuck" and need to be spread).
  _clusterBreakBonus(cueBallX, cueBallY, ghostX, ghostY, pocketX, pocketY, balls, group) {
    const R = C.BALL_RADIUS;
    if (!ghostX || !pocketX) return 0;

    // Find own-group balls that are clustered (close to other balls)
    const ownBalls = balls.filter(b => {
      if (b.isPocketed || b.isCueBall) return false;
      if (group === 'solids') return b.isSolid;
      if (group === 'stripes') return b.isStripe;
      return false;
    });

    let bonus = 0;
    for (const ob of ownBalls) {
      // Is this ball "stuck" (close to other balls, not near a pocket)?
      let nearbyCount = 0;
      for (const other of balls) {
        if (other.isPocketed || other.id === ob.id || other.isCueBall) continue;
        const d = Math.sqrt((ob.x - other.x) ** 2 + (ob.y - other.y) ** 2);
        if (d < 4 * R) nearbyCount++;
      }
      if (nearbyCount < 1) continue; // not clustered

      // Check if the cue ball's post-collision path (from ghost onward) passes
      // near this clustered ball. The cue ball deflects after the collision;
      // a rough estimate of its path is along the perpendicular or backward from the ghost.
      // Instead, check if the OBJECT ball's path to the pocket passes near the cluster.
      const tpDx = pocketX - ghostX;
      const tpDy = pocketY - ghostY;
      const tpLen = Math.sqrt(tpDx * tpDx + tpDy * tpDy);
      if (tpLen < 1) continue;
      const tpNx = tpDx / tpLen;
      const tpNy = tpDy / tpLen;

      // Distance from the clustered ball to the object ball's path
      const bx = ob.x - ghostX;
      const by = ob.y - ghostY;
      const proj = bx * tpNx + by * tpNy;
      if (proj < 0 || proj > tpLen) continue;
      const perpDist = Math.abs(-bx * tpNy + by * tpNx);

      // If the object ball would pass within ~3R of the clustered ball,
      // there's a chance of breaking up the cluster
      if (perpDist < 4 * R) {
        bonus += 5 + nearbyCount * 2;
      }
    }
    return Math.min(bonus, 15);
  }

  // Compute the approach angle relative to the pocket's acceptance cone.
  // Returns the angle in degrees; higher = worse. Returns 999 if the shot
  // is physically impossible (approach angle exceeds the pocket opening).
  //
  // Side pockets accept balls approaching roughly perpendicular to the long rail
  // (within -35deg). Corner pockets accept from a wider range -- roughly within
  // -55deg of the pocket's bisector (the diagonal into the corner).
  _pocketApproachAngle(target, pocket) {
    const L = C.TABLE_LENGTH;
    const W = C.TABLE_WIDTH;

    // Direction from target to pocket
    const bpx = pocket.x - target.x;
    const bpy = pocket.y - target.y;
    const bpLen = Math.sqrt(bpx * bpx + bpy * bpy);
    if (bpLen < 0.1) return 0;
    const bpNX = bpx / bpLen;
    const bpNY = bpy / bpLen;

    // Determine pocket type and its ideal approach direction
    const isSide = pocket.name.includes('side');
    let idealDX, idealDY, maxAngle;

    if (isSide) {
      // Side pockets: ideal approach is perpendicular to the long rail.
      // The ball-to-pocket vector for a good shot points INTO the pocket,
      // which is toward -y for top-side and toward +y for bottom-side.
      if (pocket.y < W / 2) {
        idealDX = 0; idealDY = -1;  // top-side: ball travels toward -y (into pocket)
      } else {
        idealDX = 0; idealDY = 1;   // bottom-side: ball travels toward +y (into pocket)
      }
      // BCA spec: side pocket entrance angle 103deg, acceptance ~50deg from perpendicular
      maxAngle = 50;
    } else {
      // Corner pockets: ideal approach is along the diagonal bisector INTO the corner.
      // BCA spec: corner pocket entrance angle 142deg, acceptance ~70deg from bisector
      if (pocket.x < L / 2 && pocket.y < W / 2) {
        idealDX = -1; idealDY = -1;
      } else if (pocket.x >= L / 2 && pocket.y < W / 2) {
        idealDX = 1; idealDY = -1;
      } else if (pocket.x < L / 2 && pocket.y >= W / 2) {
        idealDX = -1; idealDY = 1;
      } else {
        idealDX = 1; idealDY = 1;
      }
      const iLen = Math.sqrt(idealDX * idealDX + idealDY * idealDY);
      idealDX /= iLen; idealDY /= iLen;
      maxAngle = 65;
    }

    // Angle between the ball's approach and the pocket's ideal direction
    const dot = bpNX * idealDX + bpNY * idealDY;
    const angle = Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;

    // If the approach exceeds the pocket's acceptance cone, return 999 (impossible)
    if (angle > maxAngle) return 999;

    return angle;
  }
}
