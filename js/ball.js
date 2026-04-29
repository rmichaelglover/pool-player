// ball.js -- Ball state class
class Ball {
  constructor(id, x, y) {
    this.id = id;
    this.x = x;
    this.y = y;
    this.vx = 0;
    this.vy = 0;
    // Angular velocity (rad/s)
    // wx: spin around x-axis (affects y-motion rolling)
    // wy: spin around y-axis (affects x-motion rolling)
    // wz: sidespin (english) around vertical axis
    this.wx = 0;
    this.wy = 0;
    this.wz = 0;

    this.radius = C.BALL_RADIUS;
    this.mass = C.BALL_MASS;
    this.isPocketed = false;

    // Visual rotation tracking (accumulated angles for rendering)
    this.orientAngle = 0;  // rotation around vertical (wz), visible from above
    this.rollPhase = 0;    // rolling phase for the stripe/number visual rotation

    this.number = id;
    this.isSolid = id >= 1 && id <= 7;
    this.isStripe = id >= 9 && id <= 15;
    this.isEightBall = id === 8;
    this.isCueBall = id === 0;
    this.color = C.BALL_COLORS[id];
  }

  reset(x, y) {
    this.x = x;
    this.y = y;
    this.vx = 0;
    this.vy = 0;
    this.wx = 0;
    this.wy = 0;
    this.wz = 0;
    this.isPocketed = false;
    this.orientAngle = 0;
    this.rollPhase = 0;
  }

  speed() {
    return Math.sqrt(this.vx * this.vx + this.vy * this.vy);
  }

  angularSpeed() {
    return Math.sqrt(this.wx * this.wx + this.wy * this.wy + this.wz * this.wz);
  }

  isMoving() {
    return this.speed() > C.VELOCITY_THRESHOLD ||
           this.angularSpeed() > C.ANGULAR_VEL_THRESHOLD;
  }

  clone() {
    const b = new Ball(this.id, this.x, this.y);
    b.vx = this.vx;
    b.vy = this.vy;
    b.wx = this.wx;
    b.wy = this.wy;
    b.wz = this.wz;
    b.isPocketed = this.isPocketed;
    return b;
  }
}
