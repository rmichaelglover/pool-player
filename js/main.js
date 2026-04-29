// main.js -- Entry point, game loop, UI handlers
(function () {
  'use strict';

  const canvas = document.getElementById('poolCanvas');
  const spinCanvas = document.getElementById('spinCanvas');
  const shootBtn = document.getElementById('shootBtn');
  const nextBtn = document.getElementById('nextBtn');
  const restartBtn = document.getElementById('restartBtn');
  const statusEl = document.getElementById('status');
  const shotInfoEl = document.getElementById('shotInfo');
  const gameOverOverlay = document.getElementById('gameOverOverlay');
  const gameOverText = document.getElementById('gameOverText');
  const foulDisplay = document.getElementById('foulDisplay');
  const player1El = document.getElementById('player1');
  const player2El = document.getElementById('player2');
  const tableStatusEl = document.getElementById('tableStatus');
  const gameModeSelect = document.getElementById('gameModeSelect');

  const table = new Table();
  const renderer = new Renderer(canvas, spinCanvas, table);
  const ai = new AI();

  let game = createGame(gameModeSelect ? gameModeSelect.value : 'eight-ball', table);
  let lastTime = 0;
  let physicsAccumulator = 0;
  let loopError = false;

  function updateUI() {
    statusEl.textContent = game.statusMessage;
    shotInfoEl.textContent = ai.lastShotInfo;

    shootBtn.disabled = game.state !== 'SHOT_PREVIEW';
    nextBtn.disabled = game.state === 'SIMULATING' || game.state === 'GAME_OVER';

    // Player bar
    player1El.classList.toggle('active', game.currentPlayer === 0);
    player2El.classList.toggle('active', game.currentPlayer === 1);

    const p1Group = game.players[0].group;
    const p2Group = game.players[1].group;
    const mode = game.gameMode;

    if (mode === 'eight-ball') {
      player1El.querySelector('.player-group').textContent = p1Group
        ? (p1Group === 'solids' ? 'Solids (1-7)' : 'Stripes (9-15)') : '';
      player2El.querySelector('.player-group').textContent = p2Group
        ? (p2Group === 'solids' ? 'Solids (1-7)' : 'Stripes (9-15)') : '';
      if (game.remainingInGroup) {
        player1El.querySelector('.player-indicator').textContent = p1Group
          ? (game.remainingInGroup(0) > 0 ? `${game.remainingInGroup(0)} left` : '8-ball') : '';
        player2El.querySelector('.player-indicator').textContent = p2Group
          ? (game.remainingInGroup(1) > 0 ? `${game.remainingInGroup(1)} left` : '8-ball') : '';
      }
    } else if (mode === 'nine-ball') {
      player1El.querySelector('.player-group').textContent = '';
      player2El.querySelector('.player-group').textContent = '';
      const low = game.lowestBall ? game.lowestBall() : null;
      player1El.querySelector('.player-indicator').textContent = '';
      player2El.querySelector('.player-indicator').textContent = '';
    } else if (mode === 'fourteen-one') {
      player1El.querySelector('.player-group').textContent = `Score: ${game.players[0].score}`;
      player2El.querySelector('.player-group').textContent = `Score: ${game.players[1].score}`;
      player1El.querySelector('.player-indicator').textContent =
        game.players[0].consecutiveFouls > 0 ? `${game.players[0].consecutiveFouls} foul(s)` : '';
      player2El.querySelector('.player-indicator').textContent =
        game.players[1].consecutiveFouls > 0 ? `${game.players[1].consecutiveFouls} foul(s)` : '';
    }

    // Table status
    if (mode === 'eight-ball' && game.tableOpen) {
      tableStatusEl.textContent = 'Table Open';
      tableStatusEl.className = 'open';
    } else if (mode === 'nine-ball') {
      const low = game.lowestBall ? game.lowestBall() : null;
      tableStatusEl.textContent = low ? `Hit #${low.id} first` : '';
      tableStatusEl.className = '';
    } else if (mode === 'fourteen-one') {
      tableStatusEl.textContent = `First to ${game.targetScore || 50}`;
      tableStatusEl.className = '';
    } else {
      tableStatusEl.textContent = '';
      tableStatusEl.className = '';
    }

    // Foul display
    if (game.foulThisTurn) {
      foulDisplay.textContent = `FOUL: ${game.foulThisTurn.detail}`;
      foulDisplay.style.display = 'block';
    } else {
      foulDisplay.style.display = 'none';
    }

    // Game over
    if (game.state === 'GAME_OVER') {
      gameOverOverlay.style.display = 'block';
      gameOverText.textContent = game.gameOverMessage;
      shootBtn.disabled = true;
      nextBtn.disabled = true;
    } else {
      gameOverOverlay.style.display = 'none';
    }
  }

  function gameLoop(timestamp) {
    if (loopError) return;
    try {
      const deltaTime = Math.min((timestamp - lastTime) / 1000, 0.05);
      lastTime = timestamp;
      if (game.state === 'SIMULATING') {
        physicsAccumulator += deltaTime;
        const maxSteps = 150;
        let steps = 0;
        while (physicsAccumulator >= C.PHYSICS_DT && steps < maxSteps) {
          Physics.simulateStep(game.balls, table, C.PHYSICS_DT);
          physicsAccumulator -= C.PHYSICS_DT;
          steps++;
        }
        if (steps >= maxSteps) physicsAccumulator = 0;
        if (Physics.allStopped(game.balls)) {
          game.onSimulationComplete();
          updateUI();
        }
      }
      renderer.render(game.balls, game, game.currentShot);
      requestAnimationFrame(gameLoop);
    } catch (err) {
      loopError = true;
      console.error('GAME LOOP ERROR:', err);
      statusEl.textContent = 'Error: ' + err.message;
    }
  }

  shootBtn.addEventListener('click', () => {
    if (game.state === 'SHOT_PREVIEW') {
      game.executeShot();
      physicsAccumulator = 0;
      updateUI();
    }
  });

  nextBtn.addEventListener('click', () => {
    if (game.state !== 'SIMULATING' && game.state !== 'GAME_OVER') {
      try {
        game.planNextShot(ai);
      } catch (err) {
        console.error('Error in planNextShot:', err);
        game.statusMessage = `Error: ${err.message}. Click "Next Shot" to retry.`;
        game.state = 'SHOT_RESULT';
      }
      updateUI();
    }
  });

  restartBtn.addEventListener('click', () => {
    game.restart();
    ai.lastShotInfo = '';
    loopError = false;
    updateUI();
    if (loopError) { loopError = false; requestAnimationFrame(gameLoop); }
  });

  const viewToggle = document.getElementById('viewToggle');
  viewToggle.addEventListener('click', () => {
    if (renderer.viewMode === 'overhead') {
      renderer.setViewMode('shooter');
      viewToggle.textContent = 'Overhead View';
    } else {
      renderer.setViewMode('overhead');
      viewToggle.textContent = 'Shooter View';
    }
  });

  // Game mode selector
  if (gameModeSelect) {
    gameModeSelect.addEventListener('change', () => {
      game = createGame(gameModeSelect.value, table);
      game.init();
      ai.lastShotInfo = '';
      loopError = false;
      updateUI();
    });
  }

  try {
    game.init();
    updateUI();
    requestAnimationFrame(gameLoop);
  } catch (err) {
    console.error('Initialization error:', err);
    statusEl.textContent = 'Error: ' + err.message;
  }
})();
