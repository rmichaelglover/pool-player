/*
 * Pool physics simulation with sliding + rolling friction.
 *
 * Spin is modeled via angular velocity (wx, wy). The sliding phase
 * (mu ~0.2) converts spin into linear motion — this is how draw,
 * follow, and stop actually work. Once slip velocity drops to zero,
 * the ball transitions to natural roll with low rolling friction.
 *
 * Compiled: gcc -O2 -shared -fPIC -o libpool_sim.so pool_sim.c -lm
 */
#include <math.h>
#include <string.h>

#define MAX_BALLS 16
#define R         1.125
#define TL        100.0
#define TW        50.0
#define MU_SLIDE  0.20     /* sliding friction (ball on cloth) */
#define MU_ROLL   0.01     /* rolling resistance */
#define GRAV      386.09   /* gravity in/s^2 */
#define CUSH_R    0.90     /* cushion restitution */
#define BALL_R    0.96     /* ball-ball restitution */
#define DT        (1.0/300.0)
#define VT        0.10     /* velocity threshold for stopping */
#define SLIP_VT   0.05     /* slip velocity threshold for rolling transition */
#define MAX_STEPS 3000     /* 10 seconds of sim time */

static const double PX[6] = {0, TL/2, TL, 0, TL/2, TL};
static const double PY[6] = {0, 0, 0, TW, TW, TW};
static const double PR[6] = {2.5, 2.75, 2.5, 2.5, 2.75, 2.5};

typedef struct {
    double x, y;       /* position */
    double vx, vy;     /* linear velocity */
    double wx, wy;     /* angular velocity (wx = spin about x-axis, affects y-motion) */
    int pocketed;
} Ball;

/*
 * Apply friction: sliding (high, converts spin↔linear) or rolling (low).
 *
 * Slip velocity = linear velocity - R × angular velocity (at contact point).
 * If slip > threshold: sliding friction opposes slip, changes both v and w.
 * If slip ≈ 0: ball is in natural roll, apply rolling resistance only.
 */
static void apply_friction(Ball *b) {
    /* Slip velocity at contact point with cloth */
    /* For a ball rolling on a surface: v_contact = v - R*w (cross product in 2D) */
    /* wx (spin about x-axis) affects vy: contact_y = vy - R*wx */
    /* wy (spin about y-axis) affects vx: contact_x = vx + R*wy */
    /* v_contact = v_linear + (omega x r_contact), r_contact = (0,0,-R)
     * slip_x = vx - R*wy,  slip_y = vy + R*wx */
    double slip_x = b->vx - R * b->wy;
    double slip_y = b->vy + R * b->wx;
    double slip = sqrt(slip_x * slip_x + slip_y * slip_y);

    double sp = sqrt(b->vx * b->vx + b->vy * b->vy);

    if (slip > SLIP_VT) {
        /* Sliding phase: high friction opposes slip direction */
        double f = MU_SLIDE * GRAV * DT;
        double sx = slip_x / slip;
        double sy = slip_y / slip;
        double wf = 2.5 * MU_SLIDE * GRAV * DT / R;
        /* Check if friction would overshoot slip past zero (numerical oscillation).
         * If so, snap directly to natural roll instead. */
        double new_slip_x = (b->vx - f*sx) - R * (b->wy + wf*sx);
        double new_slip_y = (b->vy - f*sy) + R * (b->wx - wf*sy);
        if (slip_x * new_slip_x < 0 || slip_y * new_slip_y < 0) {
            /* Overshoot → transition to natural roll now */
            /* Natural roll: vx = R*wy, vy = -R*wx
             * Conserve: vx + (2/5)*R*wy = const during friction
             * Solve: vx_roll = (5*vx + 2*R*wy) / 7 */
            double vx_roll = (5.0*b->vx + 2.0*R*b->wy) / 7.0;
            double vy_roll = (5.0*b->vy - 2.0*R*b->wx) / 7.0;
            b->vx = vx_roll;
            b->vy = vy_roll;
            b->wy = vx_roll / R;
            b->wx = -vy_roll / R;
        } else {
            b->vx -= f * sx;
            b->vy -= f * sy;
            b->wy += wf * sx;
            b->wx -= wf * sy;
        }
    } else if (sp > VT) {
        /* Rolling phase: natural roll, low friction */
        /* Natural roll: wy = vx/R, wx = -vy/R */
        b->wy =  b->vx / R;
        b->wx = -b->vy / R;
        double f = MU_ROLL * GRAV * DT;
        if (f > sp) f = sp;
        b->vx -= (b->vx / sp) * f;
        b->vy -= (b->vy / sp) * f;
        b->wy =  b->vx / R;
        b->wx = -b->vy / R;
    } else {
        /* Stopped */
        b->vx = 0; b->vy = 0;
        b->wx = 0; b->wy = 0;
    }
}

int simulate_shot(
    const double *pos_in, int n_balls,
    double cue_vx, double cue_vy,
    int spin_type, double aim_dx, double aim_dy,
    double *pos_out, int *pocketed_out,
    int *hit_ball, int *hit_rail)
{
    if (n_balls > MAX_BALLS) n_balls = MAX_BALLS;

    Ball b[MAX_BALLS];
    *hit_ball = 0;
    *hit_rail = 0;

    for (int i = 0; i < n_balls; i++) {
        b[i].x  = pos_in[i*2];
        b[i].y  = pos_in[i*2+1];
        b[i].vx = (i == 0) ? cue_vx : 0.0;
        b[i].vy = (i == 0) ? cue_vy : 0.0;
        b[i].wx = 0.0;
        b[i].wy = 0.0;
        b[i].pocketed = 0;
    }

    /* Set cue ball spin based on spin type.
     * The spin is set as angular velocity that will interact with
     * cloth friction during the sliding phase.
     * Speed factor: spin proportional to shot speed for realistic behavior. */
    double cue_speed = sqrt(cue_vx * cue_vx + cue_vy * cue_vy);
    if (cue_speed > 0.1) {
        double spin_factor = cue_speed / R;  /* natural roll angular velocity */
        /* Natural roll: wy = vx/R, wx = -vy/R (slip = 0) */
        if (spin_type == 0) {
            /* Stop: moderate backspin (hit below center).
             * Friction consumes the backspin during travel so the ball
             * arrives near zero spin. Collision kills velocity → dead stop.
             * More realistic than zero spin (which gains topspin and creeps). */
            b[0].wy = -0.7 * b[0].vx / R;
            b[0].wx =  0.7 * b[0].vy / R;
        } else if (spin_type == 1) {
            /* Follow: topspin (wy > natural roll).
             * Negative slip → friction pushes ball forward after collision. */
            b[0].wy =  2.5 * b[0].vx / R;
            b[0].wx = -2.5 * b[0].vy / R;
        } else if (spin_type == 2) {
            /* Draw: backspin (wy opposite to natural roll).
             * After collision kills v, backspin pulls ball backward. */
            b[0].wy = -2.0 * b[0].vx / R;
            b[0].wx =  2.0 * b[0].vy / R;
        }
    }

    for (int step = 0; step < MAX_STEPS; step++) {
        /* ── Friction + move ── */
        int all_stopped = 1;
        for (int i = 0; i < n_balls; i++) {
            if (b[i].pocketed) continue;
            apply_friction(&b[i]);
            double sp = sqrt(b[i].vx*b[i].vx + b[i].vy*b[i].vy);
            double aw = sqrt(b[i].wx*b[i].wx + b[i].wy*b[i].wy);
            if (sp > VT || aw > SLIP_VT) all_stopped = 0;
            b[i].x += b[i].vx * DT;
            b[i].y += b[i].vy * DT;
        }
        if (all_stopped) break;

        /* ── Ball-ball collisions (multi-pass) ── */
        for (int pass = 0; pass < 5; pass++) {
            int any_col = 0;
            for (int i = 0; i < n_balls; i++) {
                if (b[i].pocketed) continue;
                for (int j = i + 1; j < n_balls; j++) {
                    if (b[j].pocketed) continue;
                    double sv = b[i].vx*b[i].vx + b[i].vy*b[i].vy
                              + b[j].vx*b[j].vx + b[j].vy*b[j].vy;
                    if (sv < 0.001) continue;

                    double dx = b[j].x - b[i].x;
                    double dy = b[j].y - b[i].y;
                    double d2 = dx*dx + dy*dy;
                    double md = 2.0 * R;
                    if (d2 >= md*md || d2 < 0.0001) continue;

                    double d = sqrt(d2);
                    double ol = md - d;

                    double dvx = b[i].vx - b[j].vx;
                    double dvy = b[i].vy - b[j].vy;
                    double cs = (dvx*dx + dvy*dy) / d;
                    double dt2 = (cs > 0.01) ? ol / cs : 0.0;
                    double cx1 = b[i].x - b[i].vx*dt2;
                    double cy1 = b[i].y - b[i].vy*dt2;
                    double cx2 = b[j].x - b[j].vx*dt2;
                    double cy2 = b[j].y - b[j].vy*dt2;
                    double cdx = cx2 - cx1, cdy = cy2 - cy1;
                    double cd = sqrt(cdx*cdx + cdy*cdy);
                    if (cd < 0.001) cd = 0.001;
                    double nx = cdx/cd, ny = cdy/cd;

                    b[i].x -= nx*ol/2;  b[i].y -= ny*ol/2;
                    b[j].x += nx*ol/2;  b[j].y += ny*ol/2;

                    double dvn = (b[i].vx - b[j].vx)*nx + (b[i].vy - b[j].vy)*ny;
                    if (dvn <= 0) continue;

                    any_col = 1;
                    int is_cue = (i == 0 || j == 0);

                    double jj = (1.0 + BALL_R) * dvn / 2.0;
                    b[i].vx -= jj*nx;  b[i].vy -= jj*ny;
                    b[j].vx += jj*nx;  b[j].vy += jj*ny;

                    if (is_cue) {
                        *hit_ball = 1;
                        /* Spin transfers naturally through the friction model —
                         * no need to apply spin impulse at collision.
                         * The cue ball's angular velocity persists through the
                         * collision and cloth friction handles the rest. */
                    }
                }
            }
            if (!any_col) break;
        }

        /* ── Cushion bounces ── */
        for (int i = 0; i < n_balls; i++) {
            if (b[i].pocketed) continue;
            int bounced = 0;
            if (b[i].x < R  && b[i].vx < 0) { b[i].vx = -b[i].vx*CUSH_R; b[i].x = R;     bounced = 1; }
            if (b[i].x > TL-R && b[i].vx > 0) { b[i].vx = -b[i].vx*CUSH_R; b[i].x = TL-R; bounced = 1; }
            if (b[i].y < R  && b[i].vy < 0) { b[i].vy = -b[i].vy*CUSH_R; b[i].y = R;     bounced = 1; }
            if (b[i].y > TW-R && b[i].vy > 0) { b[i].vy = -b[i].vy*CUSH_R; b[i].y = TW-R; bounced = 1; }
            if (bounced && *hit_ball) *hit_rail = 1;
        }

        /* ── Pocket detection ── */
        for (int i = 0; i < n_balls; i++) {
            if (b[i].pocketed) continue;
            for (int p = 0; p < 6; p++) {
                double pdx = b[i].x - PX[p];
                double pdy = b[i].y - PY[p];
                if (pdx*pdx + pdy*pdy < PR[p]*PR[p]) {
                    b[i].pocketed = 1;
                    b[i].vx = 0; b[i].vy = 0;
                    b[i].wx = 0; b[i].wy = 0;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < n_balls; i++) {
        pos_out[i*2]   = b[i].x;
        pos_out[i*2+1] = b[i].y;
        pocketed_out[i] = b[i].pocketed;
    }
    return 0;
}
