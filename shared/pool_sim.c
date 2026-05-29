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
#define MU_ROLL   0.015    /* rolling resistance */
#define GRAV      386.09   /* gravity in/s^2 */
#define CUSH_R    0.70     /* cushion restitution (real cushions ≈ 0.55-0.70) */
#define BALL_R    0.96     /* ball-ball restitution */
#define DT        (1.0/300.0)
#define VT        0.10     /* velocity threshold for stopping */
#define SLIP_VT   0.05     /* slip velocity threshold for rolling transition */
#define MAX_STEPS 3000     /* 10 seconds of sim time */
#define EPS_FROZEN 0.10    /* a ball within this gap of a rail is treated as
                              "frozen" — at collision, the rail provides an
                              instantaneous normal reaction that kills any
                              velocity component pushing into it. Without this,
                              rail-frozen object balls bounce off the cushion
                              at an angle when struck and miss the pocket. */

/* ── Table geometry (must match rl/table_geometry.py) ──
 * Diamond pocket spec: corner mouth 4.5″, side mouth 5″, cushion 2″,
 * slate overhang 3.5″ (total slate 107×57).
 *
 * Cushion's pocket-side edges:
 *   Corner: 45° slant (parallel to corner bisector). Cushion fabric ends
 *           with two slants converging at the cushion-back inner corner.
 *   Side:   12° from perpendicular to rail, converging inward (cushion-line
 *           mouth 5″, cushion-back mouth ≈4.15″).
 *
 * Drop pocket capture (slate cutouts):
 *   Corner: circle centered at slate corner ((-3.5, -3.5) for TL),
 *           radius ≈6.21″ — passes through the 50% midpoint of each slant.
 *   Side:   circle centered 12 5/8″ outside the cushion line on the
 *           bisector; radius 12 3/8″.
 */
#define CORNER_MOUTH       4.5
#define SIDE_MOUTH         5.0
#define CUSH_DEPTH         2.0
#define SLATE_OVERHANG     3.5
#define CORNER_RAIL_OFFSET 3.181980515339464   /* CORNER_MOUTH / sqrt(2) */
#define SIDE_HALF          2.5                  /* SIDE_MOUTH / 2 */
#define SIDE_SHIFT         0.4251131233400443  /* CUSH_DEPTH * tan(12°) */
#define CORNER_POCKET_R    6.207648715632781
#define SIDE_POCKET_R      12.375
#define SIDE_POCKET_OFFSET 12.625

/* Cushion segments along the rails — straight, broken by pocket gaps.
 * Each: (x0,y0,x1,y1,nx,ny) with inward bounce normal (nx,ny). */
static const double CUSHIONS[6][6] = {
    /* Top rail (y=0): split by top side pocket. Inward normal +y. */
    { CORNER_RAIL_OFFSET, 0.0,  TL/2 - SIDE_HALF, 0.0,   0.0, +1.0 },
    { TL/2 + SIDE_HALF,   0.0,  TL - CORNER_RAIL_OFFSET, 0.0,  0.0, +1.0 },
    /* Bottom rail (y=TW). Inward -y. */
    { CORNER_RAIL_OFFSET, TW,   TL/2 - SIDE_HALF, TW,    0.0, -1.0 },
    { TL/2 + SIDE_HALF,   TW,   TL - CORNER_RAIL_OFFSET, TW,   0.0, -1.0 },
    /* Left rail (x=0): single segment between corners. Inward +x. */
    { 0.0, CORNER_RAIL_OFFSET,  0.0, TW - CORNER_RAIL_OFFSET,  +1.0, 0.0 },
    /* Right rail (x=TL): single segment. Inward -x. */
    { TL,  CORNER_RAIL_OFFSET,  TL,  TW - CORNER_RAIL_OFFSET,  -1.0, 0.0 },
};

/* Facing segments — slanted cushion ends at each pocket. 8 corner facings
 * (45° from rail) + 4 side facings (12° from perpendicular).
 * Bounce normals point toward the playing-surface side of the segment.
 * Generated from rl/table_geometry.py; regenerate if geometry changes. */
static const double FACINGS[12][6] = {
    /* Corner facings (8). Each goes from cushion-line endpoint to the
     * cushion-back endpoint along a 45° slant. */
    { 3.181980515339464, 0, 1.181980515339464, -2,
      -0.7071067811865475,  0.7071067811865475 },   /* TL top  */
    { 0, 3.181980515339464, -2, 1.181980515339464,
       0.7071067811865475, -0.7071067811865475 },   /* TL left */
    { 96.81801948466054, 0, 98.81801948466054, -2,
       0.7071067811865475,  0.7071067811865475 },   /* TR top  */
    { 100, 3.181980515339464, 102, 1.181980515339464,
      -0.7071067811865475, -0.7071067811865475 },   /* TR right */
    { 3.181980515339464, 50, 1.181980515339464, 52,
      -0.7071067811865475, -0.7071067811865475 },   /* BL bot  */
    { 0, 46.81801948466054, -2, 48.81801948466054,
       0.7071067811865475,  0.7071067811865475 },   /* BL left */
    { 96.81801948466054, 50, 98.81801948466054, 52,
       0.7071067811865475, -0.7071067811865475 },   /* BR bot  */
    { 100, 46.81801948466054, 102, 48.81801948466054,
      -0.7071067811865475,  0.7071067811865475 },   /* BR right */
    /* Side facings (4) — 12° slant from perpendicular. */
    { 47.5, 0, 47.92511312334005, -2,
       0.9781476007338057,  0.2079116908177593 },   /* T-side L */
    { 52.5, 0, 52.07488687665995, -2,
      -0.9781476007338057,  0.2079116908177593 },   /* T-side R */
    { 47.5, 50, 47.92511312334005, 52,
       0.9781476007338057, -0.2079116908177593 },   /* B-side L */
    { 52.5, 50, 52.07488687665995, 52,
      -0.9781476007338057, -0.2079116908177593 },   /* B-side R */
};

/* Drop-pocket capture circles (slate cutouts).
 *   POCKETS[i] = { cx, cy, r }.
 *   Order matches POCKET_NAMES = ['TL','T-side','TR','BL','B-side','BR']. */
static const double POCKET_CIRCLES[6][3] = {
    { -3.5,    -3.5,     6.207648715632781 },  /* TL */
    { 50.0,   -12.625,  12.375              },  /* T-side */
    { 103.5,  -3.5,      6.207648715632781 },  /* TR */
    { -3.5,    53.5,     6.207648715632781 },  /* BL */
    { 50.0,    62.625,  12.375              },  /* B-side */
    { 103.5,   53.5,     6.207648715632781 },  /* BR */
};

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
    double spin_factor, double aim_dx, double aim_dy,
    double *pos_out, int *pocketed_out,
    int *hit_ball, int *hit_rail,
    /* Optional trajectory recording: if traj_out is non-NULL, every
     * TRAJ_STRIDE internal steps we write a frame of (x0,y0,x1,y1,...)
     * into traj_out, up to traj_max_frames. On return, *traj_n_out holds
     * the number of frames actually written (0 if traj_out is NULL).
     *
     * Optional cue-ball path length: if cue_path_len_out is non-NULL,
     * accumulates total distance traveled by the cue ball (sum of
     * |v|*DT each step while not pocketed) and writes the result. */
    double *traj_out, int *traj_n_out, int traj_max_frames,
    double *cue_path_len_out,
    int *cue_contacts_out,
    /* Optional: if non-NULL, receives the ball-array index (matching pos_in
     * order) of the FIRST object ball the cue ball contacts, or -1 if the
     * cue never touches an object ball. Used for foul detection — this is
     * the true first contact from the physics, so it is correct for kicks
     * and caroms where a straight cue->aim ray would be wrong. */
    int *first_hit_idx_out)
{
    const int TRAJ_STRIDE = 6;  /* 300 Hz / 6 = 50 frames per second */
    int traj_n = 0;
    double cue_path_len = 0.0;
    int cue_contacts = 0;
    if (n_balls > MAX_BALLS) n_balls = MAX_BALLS;

    Ball b[MAX_BALLS];
    *hit_ball = 0;
    *hit_rail = 0;
    int first_hit_idx = -1;

    for (int i = 0; i < n_balls; i++) {
        b[i].x  = pos_in[i*2];
        b[i].y  = pos_in[i*2+1];
        b[i].vx = (i == 0) ? cue_vx : 0.0;
        b[i].vy = (i == 0) ? cue_vy : 0.0;
        b[i].wx = 0.0;
        b[i].wy = 0.0;
        b[i].pocketed = 0;
    }

    /* Set cue ball spin from the continuous spin_factor.
     *
     * spin_factor is the ratio of initial angular velocity to the
     * natural-roll angular velocity (vx/R). Physical tip-contact
     * model: a cue tip striking at vertical offset b below/above
     * ball center imparts ω = (5/2)(b/R)(v/R) → spin_factor = 5b/(2R).
     * Slider ey ∈ [-1, +1] → spin_factor = 2.5·ey.
     *
     *   spin_factor =  0    → no spin (true center hit; slides, builds
     *                          to natural roll over distance)
     *   spin_factor =  1    → natural roll (lag shot)
     *   spin_factor =  2    → max follow (2× natural roll)
     *   spin_factor = -1    → full backspin
     *   spin_factor = -2    → max draw */
    if (sqrt(cue_vx * cue_vx + cue_vy * cue_vy) > 0.1) {
        /* Spin coupling factor: dampens follow/draw effects to better match
         * real-world cue/cloth/ball interaction. Empirically calibrated:
         * 1.0 = full theoretical spin (model says spin_factor=-1 → backspin
         * matching natural roll). 0.7 = realistic skilled-player draw range
         * across all spin values without exaggerating low-draw effects. */
        const double SPIN_COUPLE = 0.7;
        double sf = spin_factor * SPIN_COUPLE;
        b[0].wy =  sf * b[0].vx / R;
        b[0].wx = -sf * b[0].vy / R;
    }

    for (int step = 0; step < MAX_STEPS; step++) {
        /* Record a trajectory frame every TRAJ_STRIDE steps (and always
         * the first step so clients see the initial configuration). */
        if (traj_out != NULL && traj_n < traj_max_frames &&
            (step % TRAJ_STRIDE == 0)) {
            for (int i = 0; i < n_balls; i++) {
                traj_out[traj_n * 2 * n_balls + i*2    ] = b[i].x;
                traj_out[traj_n * 2 * n_balls + i*2 + 1] = b[i].y;
            }
            traj_n++;
        }

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
            /* Cue-ball path length: only counts AFTER first OB contact, since
             * pre-contact travel is forced by shot geometry, not a control
             * choice. *hit_ball is set to 1 at the cue→OB collision below. */
            if (i == 0 && *hit_ball) cue_path_len += sp * DT;
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

                    /* 3-body constraint: if either ball is rail-frozen, the
                     * rail absorbs any inward impulse (instantaneous reaction).
                     * Project the post-collision velocity onto the rail-parallel
                     * direction so the ball travels ALONG the rail rather than
                     * bouncing off it at an angle. */
                    for (int k = 0; k < 2; k++) {
                        int idx = (k == 0) ? i : j;
                        if (b[idx].y < R + EPS_FROZEN && b[idx].vy < 0)
                            b[idx].vy = 0;
                        if (b[idx].y > TW - R - EPS_FROZEN && b[idx].vy > 0)
                            b[idx].vy = 0;
                        if (b[idx].x < R + EPS_FROZEN && b[idx].vx < 0)
                            b[idx].vx = 0;
                        if (b[idx].x > TL - R - EPS_FROZEN && b[idx].vx > 0)
                            b[idx].vx = 0;
                    }

                    if (is_cue) {
                        *hit_ball = 1;
                        if (first_hit_idx < 0) first_hit_idx = (i == 0) ? j : i;
                        cue_contacts++;
                        /* Spin transfers naturally through the friction model —
                         * no need to apply spin impulse at collision.
                         * The cue ball's angular velocity persists through the
                         * collision and cloth friction handles the rest. */
                    }
                }
            }
            if (!any_col) break;
        }

        /* ── Cushion + facing bounces ──
         * Iterate over all 18 segments (6 cushion + 12 facing). For each,
         * find the closest point on the segment to the ball center; if
         * within R and the ball is on the bounce-normal side moving toward
         * the segment, reflect velocity and push the ball out along the
         * segment's bounce normal. */
        for (int i = 0; i < n_balls; i++) {
            if (b[i].pocketed) continue;
            int bounced = 0;
            for (int seg = 0; seg < 6 + 12; seg++) {
                int is_facing = (seg >= 6);
                const double *S = is_facing ? FACINGS[seg - 6]
                                             : CUSHIONS[seg];
                double sx0 = S[0], sy0 = S[1], sx1 = S[2], sy1 = S[3];
                double snx = S[4], sny = S[5];
                double sdx = sx1 - sx0, sdy = sy1 - sy0;
                double seg_len_sq = sdx * sdx + sdy * sdy;
                double t = ((b[i].x - sx0) * sdx + (b[i].y - sy0) * sdy)
                            / seg_len_sq;
                /* Cushion segments include endpoint cap zones (the cushion-
                 * points at pocket corners are real bounce points). Facing
                 * segments skip the caps: the OUTER endpoint is the same
                 * cushion-point handled by the cushion above, and the INNER
                 * endpoint represents the tip of the facing inside the
                 * pocket — in real tables the ball drops into the hole
                 * before reaching it, so we don't model a bounce there. */
                if (is_facing) {
                    if (t <= 0.0 || t >= 1.0) continue;
                } else {
                    if (t < 0.0) t = 0.0;
                    if (t > 1.0) t = 1.0;
                }
                double cpx = sx0 + t * sdx;
                double cpy = sy0 + t * sdy;
                double pdx = b[i].x - cpx;
                double pdy = b[i].y - cpy;
                double pdist_sq = pdx * pdx + pdy * pdy;
                if (pdist_sq >= R * R) continue;
                /* Ball must be on the bounce-normal side (positive dot of
                 * (ball-closest_point) with normal). Otherwise the ball is
                 * on the rail-wood side and we don't bounce (it shouldn't
                 * physically be there). */
                double side = pdx * snx + pdy * sny;
                if (side < 0.0) continue;
                /* Push out along the bounce normal */
                double pdist = sqrt(pdist_sq);
                double overlap = R - pdist;
                b[i].x += overlap * snx;
                b[i].y += overlap * sny;
                /* Reflect velocity along the bounce normal */
                double vdotn = b[i].vx * snx + b[i].vy * sny;
                if (vdotn < 0.0) {
                    b[i].vx -= (1.0 + CUSH_R) * vdotn * snx;
                    b[i].vy -= (1.0 + CUSH_R) * vdotn * sny;
                    /* Spin is preserved across the bounce (angular momentum
                     * doesn't instantly reorient). Post-bounce, the ball
                     * usually slides briefly because spin no longer matches
                     * the new linear velocity — sliding friction handles the
                     * realignment, dissipating extra energy. This is the
                     * physically correct behavior. */
                    bounced = 1;
                }
            }
            if (bounced && *hit_ball) *hit_rail = 1;
        }

        /* ── Pocket detection (drop-pocket circle + mouth-line constraint) ──
         * A ball is captured when its center is INSIDE a pocket's drop
         * circle AND past that pocket's mouth/cushion line (so corner
         * circles' overhang into the playing surface only counts once a
         * ball has crossed the mouth chord). */
        for (int i = 0; i < n_balls; i++) {
            if (b[i].pocketed) continue;
            double bx = b[i].x, by = b[i].y;
            int captured = 0;
            /* Helper: distance² from (bx,by) to circle center. */
            #define IN_CIRCLE(idx) ( \
                (bx - POCKET_CIRCLES[idx][0]) * (bx - POCKET_CIRCLES[idx][0]) + \
                (by - POCKET_CIRCLES[idx][1]) * (by - POCKET_CIRCLES[idx][1]) < \
                POCKET_CIRCLES[idx][2] * POCKET_CIRCLES[idx][2] )

            /* TL corner: inside circle AND past throat (x + y < off) */
            if (!captured && IN_CIRCLE(0) &&
                bx + by < CORNER_RAIL_OFFSET) captured = 1;
            /* T-side: inside circle AND past cushion line (y<0) within mouth */
            if (!captured && IN_CIRCLE(1) && by < 0.0 &&
                bx > TL/2 - SIDE_HALF && bx < TL/2 + SIDE_HALF) captured = 1;
            /* TR corner */
            if (!captured && IN_CIRCLE(2) &&
                (TL - bx) + by < CORNER_RAIL_OFFSET) captured = 1;
            /* BL corner */
            if (!captured && IN_CIRCLE(3) &&
                bx + (TW - by) < CORNER_RAIL_OFFSET) captured = 1;
            /* B-side */
            if (!captured && IN_CIRCLE(4) && by > TW &&
                bx > TL/2 - SIDE_HALF && bx < TL/2 + SIDE_HALF) captured = 1;
            /* BR corner */
            if (!captured && IN_CIRCLE(5) &&
                (TL - bx) + (TW - by) < CORNER_RAIL_OFFSET) captured = 1;
            #undef IN_CIRCLE
            if (captured) {
                b[i].pocketed = 1;
                b[i].vx = 0; b[i].vy = 0;
                b[i].wx = 0; b[i].wy = 0;
            }
        }
    }

    /* Capture a final frame at the stopping position (so the animation
     * ends on the final resting positions, not just the last stride tick). */
    if (traj_out != NULL && traj_n < traj_max_frames) {
        for (int i = 0; i < n_balls; i++) {
            traj_out[traj_n * 2 * n_balls + i*2    ] = b[i].x;
            traj_out[traj_n * 2 * n_balls + i*2 + 1] = b[i].y;
        }
        traj_n++;
    }
    if (traj_n_out != NULL) *traj_n_out = traj_n;
    if (cue_path_len_out != NULL) *cue_path_len_out = cue_path_len;
    if (cue_contacts_out != NULL) *cue_contacts_out = cue_contacts;
    if (first_hit_idx_out != NULL) *first_hit_idx_out = first_hit_idx;

    for (int i = 0; i < n_balls; i++) {
        pos_out[i*2]   = b[i].x;
        pos_out[i*2+1] = b[i].y;
        pocketed_out[i] = b[i].pocketed;
    }
    return 0;
}
