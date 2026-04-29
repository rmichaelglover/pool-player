// game.js -- Game state machine with multiple game modes
// Base class handles common state, shot execution, physics events.
// Subclasses implement mode-specific rules.

class GameBase {
  constructor(table) {
    this.table = table;
    this.balls = [];
    this.state = 'IDLE';
    this.gameMode = 'eight-ball'; // overridden by subclass
    this.players = [
      { name: 'Player 1', group: null, score: 0, consecutiveFouls: 0 },
      { name: 'Player 2', group: null, score: 0, consecutiveFouls: 0 },
    ];
    this.currentPlayer = 0;
    this.tableOpen = true;
    this.pocketedThisTurn = [];
    this.foulThisTurn = null;
    this.ballInHand = false;
    this.ballInKitchen = false; // 14.1: cue ball in kitchen after scratch
    this.currentShot = null;
    this.turnNumber = 0;
    this.gameOverMessage = '';
    this.statusMessage = '';
    this._snapshotPocketed = [];
  }

  currentPlayerName() { return this.players[this.currentPlayer].name; }
  opponentIndex() { return 1 - this.currentPlayer; }
  opponentName() { return this.players[this.opponentIndex()].name; }
  getCueBall() { return this.balls.find(b => b.id === 0); }

  // Override in subclass
  init() {}
  getAIGroup() { return null; }
  _evaluateResult() {}

  planNextShot(ai) {
    this.pocketedThisTurn = [];
    this.foulThisTurn = null;

    const cueBall = this.getCueBall();
    if (!cueBall || cueBall.isPocketed) {
      this._respawnCueBall();
    }

    if (this.ballInHand) {
      const group = this.getAIGroup();
      const bestPos = ai.findBestPlacement(this.balls, this.table, group);
      const cb = this.getCueBall();
      cb.isPocketed = false;
      cb.reset(bestPos.x, bestPos.y);
      this.ballInHand = false;
      this.ballInKitchen = false;
    }

    let shot;
    if (this.state === 'BREAKING') {
      shot = ai.planBreak(this.getCueBall(), this.balls, this.table);
    } else {
      const group = this.getAIGroup();
      shot = ai.findBestShot(this.balls, this.table, group, null, this.getCueBall());
    }

    if (shot) {
      this.currentShot = shot;
      this.state = 'SHOT_PREVIEW';
      this.statusMessage = `${this.currentPlayerName()}'s turn. ${ai.lastShotInfo}. Click "Shoot".`;
    } else {
      this.statusMessage = `${this.currentPlayerName()}: No valid shot found!`;
      this.state = 'SHOT_RESULT';
    }
  }

  executeShot() {
    if (this.state !== 'SHOT_PREVIEW' || !this.currentShot) return;
    const cueBall = this.getCueBall();
    const shot = this.currentShot;
    this._snapshotPocketed = this.balls.filter(b => b.isPocketed).map(b => b.id);
    Physics.resetEvents();
    Physics.strikeCueBall(cueBall, shot.aimDX, shot.aimDY, shot.force,
                          shot.contactX, shot.contactY, shot.elevation || 0);
    this.state = 'SIMULATING';
    this.statusMessage = `${this.currentPlayerName()} shooting...`;
    this.turnNumber++;
  }

  onSimulationComplete() {
    this.pocketedThisTurn = this.balls.filter(b =>
      b.isPocketed && !this._snapshotPocketed.includes(b.id));
    this.state = 'SHOT_RESULT';
    this._evaluateResult();
  }

  // -- Common foul detection --
  _detectBasicFouls(pocketed, events, requiredFirstHitCheck) {
    const cueBallPocketed = pocketed.some(b => b.isCueBall);
    const offTableEvents = events.filter(e => e.type === 'off-table');
    const cueBallOffTable = offTableEvents.some(e => e.ballId === 0);
    const cueBallHits = events.filter(e => e.type === 'ball-hit' && e.hitterId === 0);
    const firstCueBallHit = cueBallHits.length > 0 ? cueBallHits[0] : null;
    const noBallContacted = !firstCueBallHit;

    let noRailAfterContact = false;
    if (firstCueBallHit && !cueBallPocketed) {
      const anyPocketed = pocketed.filter(b => !b.isCueBall).length > 0;
      const anyCushionAfterContact = events.some(e => e.type === 'cushion');
      if (!anyPocketed && !anyCushionAfterContact) noRailAfterContact = true;
    }

    // Re-spot off-table balls
    for (const e of offTableEvents) {
      if (e.ballId === 0) continue;
      const ball = this.balls.find(b => b.id === e.ballId);
      if (ball) this._respotBall(ball);
    }

    if (cueBallPocketed) return { type: 'scratch', detail: 'Cue ball pocketed', firstCueBallHit, cueBallPocketed, cueBallOffTable };
    if (cueBallOffTable) return { type: 'off-table', detail: 'Cue ball jumped off table', firstCueBallHit, cueBallPocketed, cueBallOffTable };
    if (noBallContacted) return { type: 'no-contact', detail: 'Cue ball did not hit any ball', firstCueBallHit, cueBallPocketed, cueBallOffTable };

    // Wrong ball first check (passed in by subclass)
    if (requiredFirstHitCheck && firstCueBallHit) {
      const hitBall = this.balls.find(b => b.id === firstCueBallHit.hitId);
      if (hitBall && !requiredFirstHitCheck(hitBall)) {
        return { type: 'wrong-ball', detail: `Hit ball ${hitBall.id} first (wrong ball)`, firstCueBallHit, cueBallPocketed, cueBallOffTable };
      }
    }

    if (noRailAfterContact) return { type: 'no-rail', detail: 'No ball reached a cushion or was pocketed', firstCueBallHit, cueBallPocketed, cueBallOffTable };

    return null; // no foul
  }

  // -- Ball placement helpers --
  _respawnCueBall() {
    const cueBall = this.getCueBall();
    cueBall.isPocketed = false;
    let x = C.HEAD_SPOT_X;
    let y = C.HEAD_SPOT_Y;
    let attempts = 0;
    while (this._isPositionOccupied(x, y, 0) && attempts < 50) {
      x = C.HEAD_SPOT_X + (Math.random() - 0.5) * 10;
      y = C.TABLE_WIDTH / 2 + (Math.random() - 0.5) * (C.TABLE_WIDTH - 4);
      x = Math.max(C.BALL_RADIUS + 1, Math.min(C.HEAD_SPOT_X, x));
      attempts++;
    }
    cueBall.reset(x, y);
  }

  _respotBall(ball) {
    ball.isPocketed = false;
    let x = C.FOOT_SPOT_X;
    let y = C.FOOT_SPOT_Y;
    let attempts = 0;
    while (this._isPositionOccupied(x, y, ball.id) && attempts < 50) {
      x = C.FOOT_SPOT_X + (attempts + 1) * C.BALL_RADIUS * 2.2;
      if (x > C.TABLE_LENGTH - C.BALL_RADIUS * 2) { x = C.FOOT_SPOT_X; y += C.BALL_RADIUS * 2.2; }
      attempts++;
    }
    ball.reset(x, y);
  }

  _isPositionOccupied(x, y, excludeId) {
    const minDist = C.BALL_RADIUS * 2.1;
    for (const b of this.balls) {
      if (b.id === excludeId || b.isPocketed) continue;
      const dx = b.x - x; const dy = b.y - y;
      if (dx * dx + dy * dy < minDist * minDist) return true;
    }
    return false;
  }

  restart() { this.init(); }
}

// =====================================================================
// 8-BALL
// =====================================================================
class EightBallGame extends GameBase {
  constructor(table) {
    super(table);
    this.gameMode = 'eight-ball';
  }

  init() {
    this.balls = [];
    const margin = C.TABLE_WIDTH * 0.15;
    const breakY = margin + Math.random() * (C.TABLE_WIDTH - 2 * margin);
    this.balls.push(new Ball(0, C.HEAD_SPOT_X, breakY));
    for (const rp of this.table.getRackPositions()) this.balls.push(new Ball(rp.id, rp.x, rp.y));
    this.players[0].group = null; this.players[1].group = null;
    this.players[0].score = 0; this.players[1].score = 0;
    this.currentPlayer = 0; this.tableOpen = true;
    this.state = 'BREAKING'; this.turnNumber = 0;
    this.foulThisTurn = null; this.ballInHand = false; this.ballInKitchen = false;
    this.gameOverMessage = '';
    this.statusMessage = `${this.currentPlayerName()} breaks. Click "Next Shot".`;
  }

  currentGroup() { return this.players[this.currentPlayer].group; }

  remainingInGroup(playerIdx) {
    const group = this.players[playerIdx].group;
    if (!group) return 99;
    if (group === 'solids') return this.balls.filter(b => b.isSolid && !b.isPocketed).length;
    if (group === 'stripes') return this.balls.filter(b => b.isStripe && !b.isPocketed).length;
    return 0;
  }

  shootingEightBall() {
    const group = this.currentGroup();
    if (!group) return false;
    return this.remainingInGroup(this.currentPlayer) === 0;
  }

  getAIGroup() {
    let group = this.currentGroup();
    if (this.tableOpen) return null;
    if (group && this.remainingInGroup(this.currentPlayer) === 0) return 'eightball';
    return group;
  }

  _evaluateResult() {
    const events = Physics.events;
    const pocketed = this.pocketedThisTurn;
    const isBreak = this.turnNumber === 1;
    const player = this.players[this.currentPlayer];

    // Wrong ball first check for 8-ball
    let firstHitCheck = null;
    if (!this.tableOpen && player.group) {
      firstHitCheck = (hitBall) => {
        const ownPocketed = pocketed.filter(b => {
          if (player.group === 'solids') return b.isSolid;
          if (player.group === 'stripes') return b.isStripe;
          return false;
        }).length;
        const remainingBefore = this.remainingInGroup(this.currentPlayer) + ownPocketed;
        if (remainingBefore === 0) return hitBall.isEightBall;
        if (player.group === 'solids') return hitBall.isSolid;
        if (player.group === 'stripes') return hitBall.isStripe;
        return true;
      };
    }

    const foulResult = this._detectBasicFouls(pocketed, events, firstHitCheck);
    if (foulResult) this.foulThisTurn = foulResult;
    const isFoul = this.foulThisTurn !== null;

    // 8-ball pocketed
    const eightBall = this.balls.find(b => b.id === 8);
    if (pocketed.some(b => b.isEightBall)) {
      if (isBreak) { this._respotBall(eightBall); this.statusMessage = '8-ball pocketed on break -- re-spotted.'; }
      else if (isFoul) { this.state = 'GAME_OVER'; this.gameOverMessage = `${this.currentPlayerName()} loses! 8-ball on a foul.`; this.statusMessage = this.gameOverMessage; return; }
      else if (this.shootingEightBall()) { this.state = 'GAME_OVER'; this.gameOverMessage = `${this.currentPlayerName()} wins!`; this.statusMessage = this.gameOverMessage; return; }
      else { this.state = 'GAME_OVER'; this.gameOverMessage = `${this.currentPlayerName()} loses! 8-ball pocketed early.`; this.statusMessage = this.gameOverMessage; return; }
    }

    // Group assignment
    if (this.tableOpen && !isBreak && !isFoul) {
      const legallyPocketed = pocketed.filter(b => !b.isCueBall && !b.isEightBall);
      if (legallyPocketed.length > 0) {
        const first = legallyPocketed[0];
        if (first.isSolid) { this.players[this.currentPlayer].group = 'solids'; this.players[this.opponentIndex()].group = 'stripes'; }
        else if (first.isStripe) { this.players[this.currentPlayer].group = 'stripes'; this.players[this.opponentIndex()].group = 'solids'; }
        this.tableOpen = false;
      }
    }

    if (isFoul) {
      this.ballInHand = true;
      if (foulResult.cueBallPocketed || foulResult.cueBallOffTable) { const cb = this.getCueBall(); cb.isPocketed = false; cb.reset(C.HEAD_SPOT_X, C.HEAD_SPOT_Y); }
      this.currentPlayer = this.opponentIndex();
      this.statusMessage = `FOUL: ${this.foulThisTurn.detail}! ${this.currentPlayerName()} gets ball-in-hand.`;
      return;
    }

    // Turn logic
    const objPocketed = pocketed.filter(b => !b.isCueBall && !b.isEightBall);
    let pocketedOwn = false;
    if (this.tableOpen || isBreak) { pocketedOwn = objPocketed.length > 0; }
    else if (player.group) { pocketedOwn = objPocketed.some(b => (player.group === 'solids' ? b.isSolid : b.isStripe)); }

    if (pocketedOwn) {
      const nowOnEight = this.remainingInGroup(this.currentPlayer) === 0;
      this.statusMessage = `${this.currentPlayerName()} pocketed ${objPocketed.map(b=>'#'+b.id).join(', ')}. ` + (nowOnEight ? 'Now shooting the 8-ball! ' : 'Shoots again! ');
    } else if (objPocketed.length > 0) {
      this.currentPlayer = this.opponentIndex();
      this.statusMessage = `Pocketed opponent's ball. ${this.currentPlayerName()}'s turn.`;
    } else {
      this.currentPlayer = this.opponentIndex();
      this.statusMessage = `No ball pocketed. ${this.currentPlayerName()}'s turn.`;
    }
  }
}

// =====================================================================
// 9-BALL
// =====================================================================
class NineBallGame extends GameBase {
  constructor(table) {
    super(table);
    this.gameMode = 'nine-ball';
  }

  init() {
    this.balls = [];
    const margin = C.TABLE_WIDTH * 0.15;
    const breakY = margin + Math.random() * (C.TABLE_WIDTH - 2 * margin);
    this.balls.push(new Ball(0, C.HEAD_SPOT_X, breakY));
    for (const rp of this.table.get9BallRackPositions()) this.balls.push(new Ball(rp.id, rp.x, rp.y));
    this.players[0].group = null; this.players[1].group = null;
    this.players[0].score = 0; this.players[1].score = 0;
    this.currentPlayer = 0; this.tableOpen = false;
    this.state = 'BREAKING'; this.turnNumber = 0;
    this.foulThisTurn = null; this.ballInHand = false; this.ballInKitchen = false;
    this.gameOverMessage = '';
    this.statusMessage = `${this.currentPlayerName()} breaks. Click "Next Shot".`;
  }

  // Lowest numbered ball on the table
  lowestBall() {
    let lowest = null;
    for (const b of this.balls) {
      if (b.isPocketed || b.isCueBall) continue;
      if (!lowest || b.id < lowest.id) lowest = b;
    }
    return lowest;
  }

  getAIGroup() {
    // In 9-ball, must hit the lowest ball first. AI group = specific ball ID
    const low = this.lowestBall();
    return low ? 'nine-ball-lowest' : null;
  }

  planNextShot(ai) {
    this.pocketedThisTurn = [];
    this.foulThisTurn = null;
    const cueBall = this.getCueBall();
    if (!cueBall || cueBall.isPocketed) this._respawnCueBall();
    if (this.ballInHand) {
      // 9-ball: ball in hand anywhere, target the lowest ball
      const low = this.lowestBall();
      const nineBallGroup = low ? 'nine-ball-' + low.id : null;
      const bestPos = ai.findBestPlacement(this.balls, this.table, nineBallGroup);
      const cb = this.getCueBall();
      cb.isPocketed = false; cb.reset(bestPos.x, bestPos.y);
      this.ballInHand = false;
    }

    let shot;
    if (this.state === 'BREAKING') {
      shot = ai.planBreak(this.getCueBall(), this.balls, this.table);
    } else {
      // 9-ball: must hit the lowest ball first.
      // The AI targets ONLY the lowest ball for direct/bank/kick shots.
      // (Combo shots where you hit lowest then pocket another are not yet supported.)
      const low = this.lowestBall();
      const nineBallGroup = low ? 'nine-ball-' + low.id : null;
      shot = ai.findBestShot(this.balls, this.table, nineBallGroup, null, this.getCueBall());
    }

    if (shot) {
      this.currentShot = shot;
      this.state = 'SHOT_PREVIEW';
      const low = this.lowestBall();
      this.statusMessage = `${this.currentPlayerName()}'s turn. Must hit ${low ? '#'+low.id : '?'} first. ${ai.lastShotInfo}. Click "Shoot".`;
    } else {
      this.statusMessage = `${this.currentPlayerName()}: No valid shot found!`;
      this.state = 'SHOT_RESULT';
    }
  }

  _evaluateResult() {
    const events = Physics.events;
    const pocketed = this.pocketedThisTurn;
    const isBreak = this.turnNumber === 1;
    const lowestBefore = this.lowestBall();

    // Must hit lowest ball first
    const firstHitCheck = (hitBall) => {
      if (isBreak) return true; // on break, just need to hit the rack (1 ball at apex)
      // Find the lowest ball that was on the table at the start of the shot
      // (before any were pocketed this turn)
      let lowestId = 99;
      for (const b of this.balls) {
        if (b.isCueBall) continue;
        // Was it on the table before this shot?
        if (!this._snapshotPocketed.includes(b.id)) {
          if (b.id < lowestId) lowestId = b.id;
        }
      }
      return hitBall.id === lowestId;
    };

    const foulResult = this._detectBasicFouls(pocketed, events, firstHitCheck);
    if (foulResult) this.foulThisTurn = foulResult;
    const isFoul = this.foulThisTurn !== null;

    // 9-ball pocketed on a legal shot = win
    if (pocketed.some(b => b.id === 9) && !isFoul) {
      this.state = 'GAME_OVER';
      this.gameOverMessage = `${this.currentPlayerName()} wins! 9-ball pocketed!`;
      this.statusMessage = this.gameOverMessage;
      return;
    }
    // 9-ball pocketed on a foul: re-spot it
    if (pocketed.some(b => b.id === 9) && isFoul) {
      const nine = this.balls.find(b => b.id === 9);
      this._respotBall(nine);
    }

    // All balls pocketed check
    if (!this.balls.some(b => !b.isPocketed && !b.isCueBall)) {
      this.state = 'GAME_OVER';
      this.gameOverMessage = 'All balls pocketed!';
      this.statusMessage = this.gameOverMessage;
      return;
    }

    if (isFoul) {
      this.ballInHand = true;
      if (foulResult.cueBallPocketed || foulResult.cueBallOffTable) { const cb = this.getCueBall(); cb.isPocketed = false; cb.reset(C.HEAD_SPOT_X, C.HEAD_SPOT_Y); }
      this.currentPlayer = this.opponentIndex();
      this.statusMessage = `FOUL: ${this.foulThisTurn.detail}! ${this.currentPlayerName()} gets ball-in-hand.`;
      return;
    }

    const objPocketed = pocketed.filter(b => !b.isCueBall);
    if (objPocketed.length > 0) {
      this.statusMessage = `${this.currentPlayerName()} pocketed ${objPocketed.map(b=>'#'+b.id).join(', ')}. Shoots again!`;
    } else {
      this.currentPlayer = this.opponentIndex();
      this.statusMessage = `No ball pocketed. ${this.currentPlayerName()}'s turn.`;
    }
  }
}

// =====================================================================
// 14.1 CONTINUOUS (STRAIGHT POOL)
// =====================================================================
class StraightPoolGame extends GameBase {
  constructor(table) {
    super(table);
    this.gameMode = 'fourteen-one';
    this.targetScore = 50;
  }

  init() {
    this.balls = [];
    this.balls.push(new Ball(0, C.HEAD_SPOT_X, C.HEAD_SPOT_Y));
    for (const rp of this.table.get141RackPositions()) this.balls.push(new Ball(rp.id, rp.x, rp.y));
    this.players[0].group = null; this.players[1].group = null;
    this.players[0].score = 0; this.players[1].score = 0;
    this.players[0].consecutiveFouls = 0; this.players[1].consecutiveFouls = 0;
    this.currentPlayer = 0; this.tableOpen = false;
    this.state = 'BREAKING'; this.turnNumber = 0;
    this.foulThisTurn = null; this.ballInHand = false; this.ballInKitchen = false;
    this.gameOverMessage = '';
    this.statusMessage = `${this.currentPlayerName()} breaks. First to ${this.targetScore} wins. Click "Next Shot".`;
  }

  getAIGroup() { return null; } // any ball is legal in 14.1

  _evaluateResult() {
    const events = Physics.events;
    const pocketed = this.pocketedThisTurn;
    const isBreak = this.turnNumber === 1;
    const player = this.players[this.currentPlayer];

    // No wrong-ball-first rule in 14.1 (any ball is legal)
    const foulResult = this._detectBasicFouls(pocketed, events, null);
    if (foulResult) this.foulThisTurn = foulResult;
    const isFoul = this.foulThisTurn !== null;

    const objPocketed = pocketed.filter(b => !b.isCueBall);

    if (isFoul) {
      player.score -= 1; // -1 for foul
      player.consecutiveFouls++;

      // Three consecutive fouls = -15 points
      if (player.consecutiveFouls >= 3) {
        player.score -= 15; // additional -15 (total -16 for the third foul)
        player.consecutiveFouls = 0;
        this.statusMessage = `FOUL: ${this.foulThisTurn.detail}! THREE CONSECUTIVE FOULS! ${this.currentPlayerName()} loses 16 points (now ${player.score}).`;
      } else {
        this.statusMessage = `FOUL: ${this.foulThisTurn.detail}! ${this.currentPlayerName()}: -1 point (now ${player.score}).`;
      }

      // 14.1 foul handling: NOT ball-in-hand unless scratch
      if (foulResult.cueBallPocketed || foulResult.cueBallOffTable) {
        // Scratch: cue ball in kitchen
        this.ballInHand = true;
        this.ballInKitchen = true;
        const cb = this.getCueBall();
        cb.isPocketed = false;
        cb.reset(C.HEAD_SPOT_X, C.HEAD_SPOT_Y);
        this.statusMessage += ` Cue ball in kitchen.`;
      }
      // Non-scratch foul: cue ball stays where it is, turn switches
      this.currentPlayer = this.opponentIndex();
      return;
    }

    // Legal shot -- reset consecutive foul counter
    player.consecutiveFouls = 0;

    // Score points for pocketed balls
    if (objPocketed.length > 0) {
      player.score += objPocketed.length;
      this.statusMessage = `${this.currentPlayerName()} pocketed ${objPocketed.map(b=>'#'+b.id).join(', ')} (+${objPocketed.length}, total: ${player.score}).`;

      // Check win
      if (player.score >= this.targetScore) {
        this.state = 'GAME_OVER';
        this.gameOverMessage = `${this.currentPlayerName()} wins with ${player.score} points!`;
        this.statusMessage = this.gameOverMessage;
        return;
      }

      // Check for re-rack: if 14 balls are pocketed, re-rack
      const onTable = this.balls.filter(b => !b.isPocketed && !b.isCueBall);
      if (onTable.length <= 1) {
        this._doRerack(onTable.length === 1 ? onTable[0] : null);
        this.statusMessage += ' Re-rack!';
      }

      // Player continues shooting
      this.statusMessage += ' Shoots again!';
    } else {
      // Nothing pocketed -- turn switches
      this.currentPlayer = this.opponentIndex();
      this.statusMessage = `No ball pocketed. ${this.currentPlayerName()}'s turn.`;
    }
  }

  // Re-rack 14 balls, leaving the 15th ball and cue ball in place
  _doRerack(remainingBall) {
    const R = C.BALL_RADIUS;
    const cueBall = this.getCueBall();
    const excludeId = remainingBall ? remainingBall.id : null;

    // Check if remaining ball is in the rack area
    if (remainingBall) {
      const dx = remainingBall.x - C.FOOT_SPOT_X;
      const dy = remainingBall.y - C.FOOT_SPOT_Y;
      if (Math.sqrt(dx * dx + dy * dy) < R * 6) {
        // Ball is in the rack area -- spot it on the head spot
        remainingBall.reset(C.HEAD_SPOT_X, C.HEAD_SPOT_Y);
      }
    }

    // Check if cue ball is in the rack area
    if (cueBall) {
      const dx = cueBall.x - C.FOOT_SPOT_X;
      const dy = cueBall.y - C.FOOT_SPOT_Y;
      if (Math.sqrt(dx * dx + dy * dy) < R * 6) {
        cueBall.reset(C.HEAD_SPOT_X, C.TABLE_WIDTH / 2);
      }
    }

    // Re-rack the pocketed balls (all except the remaining ball)
    const positions = this.table.get141RackPositions(excludeId);
    for (const rp of positions) {
      const ball = this.balls.find(b => b.id === rp.id);
      if (ball && ball.isPocketed) {
        ball.isPocketed = false;
        ball.reset(rp.x, rp.y);
      }
    }
  }
}

// Factory function
function createGame(mode, table) {
  switch (mode) {
    case 'nine-ball': return new NineBallGame(table);
    case 'fourteen-one': return new StraightPoolGame(table);
    default: return new EightBallGame(table);
  }
}
