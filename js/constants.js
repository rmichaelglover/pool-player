// constants.js -- All physics and game constants
const C = Object.freeze({
  // Table dimensions (inches) -- 9-ft regulation table
  TABLE_LENGTH: 100,
  TABLE_WIDTH: 50,
  RAIL_WIDTH: 5,
  POCKET_RADIUS: 2.5,        // corner pocket opening radius (~5.0" mouth, BCA spec 4.875-5.125")
  POCKET_RADIUS_SIDE: 2.75,  // side pockets wider (~5.5" mouth, BCA spec 5.375-5.625")

  // Display
  SCALE: 8,                   // pixels per inch

  // Ball
  BALL_RADIUS: 1.125,         // 2.25" diameter
  BALL_MASS: 0.17,            // kg (6 oz)
  BALL_INERTIA_COEFF: 2 / 5,  // solid sphere: I = (2/5) m R^2

  // Friction
  MU_SLIDE: 0.2,              // sliding friction coefficient (ball-cloth)
  MU_ROLL: 0.01,              // rolling friction coefficient
  MU_SPIN_DECEL: 5.0,         // sidespin deceleration (rad/s^2)
  G: 386.09,                  // gravity in/s^2

  // Cushion
  CUSHION_RESTITUTION: 0.92,
  CUSHION_FRICTION: 0.14,

  // Ball-ball collision
  BALL_RESTITUTION: 0.96,
  SPIN_TRANSFER_COEFF: 0.05,

  // Simulation
  PHYSICS_DT: 1 / 600,
  VELOCITY_THRESHOLD: 0.1,
  ANGULAR_VEL_THRESHOLD: 0.2,

  // AI
  MAX_CUE_SPEED: 150,         // in/s
  MIN_CUE_SPEED: 20,

  // Positions
  FOOT_SPOT_X: 75,            // foot spot for rack
  FOOT_SPOT_Y: 25,
  HEAD_SPOT_X: 25,            // head spot for cue ball
  HEAD_SPOT_Y: 25,

  // Ball colors
  BALL_COLORS: [
    '#FFFFFF',  // 0: cue ball (white)
    '#FFD700',  // 1: yellow
    '#0000CC',  // 2: blue
    '#CC0000',  // 3: red
    '#660099',  // 4: purple
    '#FF6600',  // 5: orange
    '#007700',  // 6: green
    '#800000',  // 7: maroon
    '#111111',  // 8: black
    '#FFD700',  // 9: yellow stripe
    '#0000CC',  // 10: blue stripe
    '#CC0000',  // 11: red stripe
    '#660099',  // 12: purple stripe
    '#FF6600',  // 13: orange stripe
    '#007700',  // 14: green stripe
    '#800000',  // 15: maroon stripe
  ],
});
