"""
Phase 9: Key-ball / break-ball curriculum.

End-of-rack drill that teaches the agent the canonical 14.1 closing
sequence: pocket the *key* ball with shape on the *break* ball, then
pocket the break ball such that the cue scatters the newly-racked balls.

Initial state on each reset:
  - 1 break ball positioned alongside (would-be) rack apex (Phase8-style
    distribution: upper or lower, 8-15" from apex so the auto-rerack
    that fires after pocketing the key ball doesn't relocate it)
  - 1 key ball at a random position from which the cue has at least one
    legal shot
  - cue ball in the head kitchen

Episode flow:
  Shot 1 — agent calls a shot. If the call is the key ball and it goes
           in, +key_ball_reward. If the call is the break ball, regardless
           of outcome → wrong-sequence penalty.
  Shot 2 — by now Phase7Env's auto-rerack has placed 14 balls in the rack
           (apex empty), break ball remains where it was. Agent calls a
           shot. If the call is the break ball and it goes in:
             +break_ball_reward
             +scatter_bonus if ≥ scatter_threshold rack balls have moved
                              off their rack positions
             +valid_after_bonus if at least one legal shot exists in the
                                  resulting state
  Shots 3+ — regular play. max_shots caps the episode length.

Inherits Phase 7 step machinery (call-shot rule, rerack, shape bonus,
movement penalty) — only adds the curriculum reward shaping on top.
"""
from __future__ import annotations

import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
from shot_enumerator import R, generate_legal_shots
from rack_geometry import RACK_POSITIONS, RACK_APEX, TABLE_LENGTH, TABLE_WIDTH
from train_phase7 import Phase7Env


class Phase9CurriculumEnv(Phase7Env):
    """End-of-rack curriculum env. Inherits Phase 7 step/rerack logic; adds
    a curriculum reward on the first two shots and overrides the initial
    state sampler.

    Curriculum rewards (added on top of regular pocket/scratch/shape rewards):

        Shot 1 (the "key ball" shot):
          +key_ball_reward       if called=key_ball AND key_ball pocketed
          +wrong_sequence_penalty if called=break_ball  (break called too early)

        Shot 2 (the "break shot" — fires after rerack auto-spawns the rack):
          +break_ball_reward     if called=break_ball AND break_ball pocketed
          +scatter_bonus         if ≥ scatter_threshold rack balls moved
                                   off their rack positions
          +valid_after_bonus     if obs.shot_meta non-empty after the break
          +wrong_sequence_penalty if called != break_ball
    """

    def __init__(self, pocket_reward=10.0, max_shots=5,
                 opening_break_force=240.0, scratch_penalty=-10.0,
                 aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0,
                 shape_bonus_max=0.0, movement_penalty_weight=1.0,
                 cue_movement_penalty_weight=0.0,
                 cue_ricochet_penalty_weight=0.0,
                 force_efficiency_penalty_weight=0.0,
                 rail_shot_bonus_weight=0.0,
                 next_shape_bonus_max=0.0,
                 eor_bonus_max=0.0,
                 # Curriculum-specific:
                 key_ball_reward=5.0, break_ball_reward=10.0,
                 wrong_sequence_penalty=-5.0,
                 scatter_bonus=5.0, scatter_threshold=4,
                 valid_after_bonus=5.0):
        # super().__init__ calls reset() which uses _sample_curriculum_setup.
        # Curriculum reward params aren't needed during reset, only step.
        self.key_ball_reward = key_ball_reward
        self.break_ball_reward = break_ball_reward
        self.wrong_sequence_penalty = wrong_sequence_penalty
        self.scatter_bonus = scatter_bonus
        self.scatter_threshold = scatter_threshold
        self.valid_after_bonus = valid_after_bonus
        super().__init__(
            pocket_reward=pocket_reward, max_shots=max_shots,
            opening_break_force=opening_break_force,
            scratch_penalty=scratch_penalty,
            aim_noise_deg=aim_noise_deg, force_noise_pct=force_noise_pct,
            spin_noise=spin_noise,
            shape_bonus_max=shape_bonus_max,
            movement_penalty_weight=movement_penalty_weight,
            cue_movement_penalty_weight=cue_movement_penalty_weight,
            cue_ricochet_penalty_weight=cue_ricochet_penalty_weight,
            force_efficiency_penalty_weight=force_efficiency_penalty_weight,
            rail_shot_bonus_weight=rail_shot_bonus_weight,
            next_shape_bonus_max=next_shape_bonus_max,
            eor_bonus_max=eor_bonus_max,
        )

    def reset(self):
        self.cue, self.balls = self._sample_curriculum_setup()
        self.shot_idx = 0
        self.done = False
        self.rerack_count = 0
        self.total_pocketed = 0
        self.pending_rerack = False
        # EOR tracking attrs (set in Phase7Env.reset, must mirror here).
        self._break_ball_id_after_rerack = None
        self._post_rerack_break_pending = False
        # Curriculum tracking
        self.curriculum_shot = 0
        self.key_pocketed = False
        self.break_pocketed = False
        self.curriculum_success = False   # set True if shot 2 succeeds
        return self.get_obs()

    def _sample_curriculum_setup(self):
        """Place break ball alongside the rack apex (Phase8-style positions
        outside the rerack relocation radius), key ball at a random position
        with a legal shot to it, and cue in the head kitchen."""
        for trial in range(60):
            # Break ball: alongside the rack, 8-15" from apex so that after
            # the agent pockets the key ball and rerack auto-fires, the break
            # ball stays where it was placed (rerack relocates only if within
            # 8" of apex).
            scenario = random.random()
            if scenario < 0.5:
                # Upper alongside: by 11-16
                bx = 75.0 + random.random() * 9.0
                by = 11.0 + random.random() * 5.0
            else:
                # Lower alongside: by 34-39
                bx = 75.0 + random.random() * 9.0
                by = 34.0 + random.random() * 5.0

            # Key ball: random makeable spot, well away from break ball
            kb_x = 25.0 + random.random() * 50.0
            kb_y = 8.0 + random.random() * 34.0
            if math.hypot(kb_x - bx, kb_y - by) < 4 * R:
                continue

            # Cue: head kitchen
            cx = 5.0 + random.random() * 20.0
            cy = 5.0 + random.random() * 40.0
            if math.hypot(cx - kb_x, cy - kb_y) < 4 * R:
                continue
            if math.hypot(cx - bx, cy - by) < 4 * R:
                continue

            # Verify cue actually has a legal shot on the key ball
            balls = {1: [bx, by], 2: [kb_x, kb_y]}    # 1=break, 2=key
            shots = generate_legal_shots([cx, cy], balls)
            if not any(s.ball_id == 2 for s in shots):
                continue

            self._key_ball_id = 2
            self._break_ball_id = 1
            return [cx, cy], balls

        # Fallback (should rarely fire after 60 tries)
        balls = {1: [80.0, 13.0], 2: [50.0, 25.0]}
        self._key_ball_id = 2
        self._break_ball_id = 1
        return [15.0, 25.0], balls

    def step(self, *args, **kwargs):
        is_first = (self.curriculum_shot == 0)
        is_second = (self.curriculum_shot == 1)
        # Snapshot rack ball positions before the shot — needed to measure
        # scatter on shot 2.
        pre_rack_positions = {bid: tuple(pos)
                               for bid, pos in self.balls.items()
                               if bid != self._break_ball_id
                                  and bid != self._key_ball_id}

        obs, reward, done, info = super().step(*args, **kwargs)

        called_shot = info.get('shot')
        called_id = called_shot.ball_id if called_shot is not None else None
        pocketed = info.get('pocketed_ids', [])

        bonus = 0.0
        if is_first:
            if called_id == self._key_ball_id:
                if self._key_ball_id in pocketed:
                    bonus += self.key_ball_reward
                    self.key_pocketed = True
            elif called_id == self._break_ball_id:
                bonus += self.wrong_sequence_penalty
        elif is_second:
            if called_id == self._break_ball_id:
                if self._break_ball_id in pocketed:
                    bonus += self.break_ball_reward
                    self.break_pocketed = True
                    self.curriculum_success = True
                    # Scatter — count rack balls displaced from their rack positions
                    scatter = self._count_rack_scatter()
                    if scatter >= self.scatter_threshold:
                        bonus += self.scatter_bonus
                    info['scatter_count'] = scatter
                    # Valid follow-up shot in the resulting state
                    if obs is not None and getattr(obs, 'shot_meta', None):
                        bonus += self.valid_after_bonus
            else:
                bonus += self.wrong_sequence_penalty

        reward += bonus
        info['curriculum_shot'] = self.curriculum_shot
        info['curriculum_bonus'] = bonus
        self.curriculum_shot += 1
        return obs, reward, done, info

    def _count_rack_scatter(self):
        """Count rack balls (anything that's not the break ball) currently
        more than 2R from any standard rack position. After a successful
        break, this should be ≥ scatter_threshold."""
        rack_positions = RACK_POSITIONS[1:]   # 14 positions, apex empty
        scatter = 0
        for bid, pos in self.balls.items():
            if bid == self._break_ball_id:
                continue
            # Is this ball at any rack position?
            at_rack = any(math.hypot(pos[0] - rp[0], pos[1] - rp[1]) < 2 * R
                          for rp in rack_positions)
            if not at_rack:
                scatter += 1
        # Pocketed rack balls (no longer in self.balls) also count as scattered
        # — they didn't stay at rack positions.
        # Total racked at start of break-shot = 14 (apex empty rerack).
        # Surviving on table at rack positions = 14 - scatter (balls still
        # at rack positions). Pocketed/scattered count = 14 - (still racked).
        # Simpler: count balls NOT at rack = scatter. Then add (14 - on_table_count)
        # for pocketed.
        if self._key_ball_id not in self.balls:
            on_table_non_break = sum(1 for bid in self.balls
                                      if bid != self._break_ball_id)
            pocketed_count = 14 - on_table_non_break
        else:
            pocketed_count = 0   # key ball still on table — pre-rerack
        return scatter + pocketed_count


# ── Mixed env: most episodes are regular Phase 7, some are curriculum ──

class Phase9MixedEnv(Phase9CurriculumEnv):
    """Mixed training env: per-reset random choice between curriculum and
    regular Phase 7 setup. Avoids the overfitting that pure-curriculum
    training caused — most episodes still practice the full game while
    a fraction (mix_ratio) get curriculum-specific signal.

    Differences from Phase9CurriculumEnv:
      - On reset, with probability mix_ratio: curriculum setup (key+break ball).
        Otherwise: regular Phase7 setup (full rack + auto-break).
      - Curriculum reward bonuses fire ONLY in curriculum episodes.
      - Default reward magnitudes scaled up so that, when the curriculum
        bonus does fire, it's strong enough to shape behavior:
          key_ball_reward       5  → 15
          break_ball_reward    10  → 25
          wrong_sequence_penalty −5 → −20
      - max_shots stays at the regular Phase 7 default; curriculum episodes
        continue into post-break play, so the agent practices "after the
        break shot" too rather than the episode ending at shot 5.
    """

    def __init__(self, pocket_reward=10.0, max_shots=60,
                 opening_break_force=240.0, scratch_penalty=-10.0,
                 aim_noise_deg=0.0, force_noise_pct=0.0, spin_noise=0.0,
                 shape_bonus_max=0.0, movement_penalty_weight=1.0,
                 cue_movement_penalty_weight=0.0,
                 cue_ricochet_penalty_weight=0.0,
                 force_efficiency_penalty_weight=0.0,
                 rail_shot_bonus_weight=0.0,
                 next_shape_bonus_max=0.0,
                 eor_bonus_max=0.0,
                 mix_ratio=0.3,
                 rail_drill_share=0.0,
                 threeball_drill_share=0.0,
                 railbreak_drill_share=0.0,
                 key_ball_reward=15.0, break_ball_reward=25.0,
                 wrong_sequence_penalty=-20.0,
                 scatter_bonus=5.0, scatter_threshold=4,
                 valid_after_bonus=5.0):
        self.mix_ratio = mix_ratio
        # Within the curriculum slice, drill type shares (rest goes to
        # original key+break drill):
        #   rail_drill_share       — single rail-frozen ball + cue (v11)
        #   threeball_drill_share  — key1 + key2 + break + cue (v12 NEW)
        # Sum should be <= 1.0. Remaining share is original key+break drill.
        self.rail_drill_share = rail_drill_share
        self.threeball_drill_share = threeball_drill_share
        self.railbreak_drill_share = railbreak_drill_share
        # _is_curriculum_episode is set per-reset; init it here so the
        # initial reset() (called by super().__init__) can read it.
        self._is_curriculum_episode = False
        super().__init__(
            pocket_reward=pocket_reward, max_shots=max_shots,
            opening_break_force=opening_break_force,
            scratch_penalty=scratch_penalty,
            aim_noise_deg=aim_noise_deg, force_noise_pct=force_noise_pct,
            spin_noise=spin_noise,
            shape_bonus_max=shape_bonus_max,
            movement_penalty_weight=movement_penalty_weight,
            cue_movement_penalty_weight=cue_movement_penalty_weight,
            cue_ricochet_penalty_weight=cue_ricochet_penalty_weight,
            force_efficiency_penalty_weight=force_efficiency_penalty_weight,
            rail_shot_bonus_weight=rail_shot_bonus_weight,
            next_shape_bonus_max=next_shape_bonus_max,
            eor_bonus_max=eor_bonus_max,
            key_ball_reward=key_ball_reward,
            break_ball_reward=break_ball_reward,
            wrong_sequence_penalty=wrong_sequence_penalty,
            scatter_bonus=scatter_bonus,
            scatter_threshold=scatter_threshold,
            valid_after_bonus=valid_after_bonus,
        )

    def reset(self):
        if random.random() < self.mix_ratio:
            # Curriculum episode — pick drill type by share weights.
            self._is_curriculum_episode = True
            r = random.random()
            if r < self.rail_drill_share:
                self._drill_type = 'rail'
                self.cue, self.balls = self._sample_rail_drill_setup()
            elif r < self.rail_drill_share + self.threeball_drill_share:
                self._drill_type = '3ball'
                self.cue, self.balls = self._sample_3ball_drill_setup()
            elif r < (self.rail_drill_share + self.threeball_drill_share
                      + self.railbreak_drill_share):
                self._drill_type = 'railbreak'
                self.cue, self.balls = self._sample_railbreak_drill_setup()
            else:
                self._drill_type = 'key_break'
                self.cue, self.balls = self._sample_curriculum_setup()
            self._is_rail_drill = (self._drill_type == 'rail')
            self.shot_idx = 0
            self.done = False
            self.rerack_count = 0
            self.total_pocketed = 0
            self.pending_rerack = False
            # EOR tracking attrs (mirror Phase7Env.reset).
            self._break_ball_id_after_rerack = None
            self._post_rerack_break_pending = False
            # For the railbreak drill, the setup function assigned the
            # break ball id (id 15) before the EOR-tracking reset above
            # wiped it. Re-assign here and mark next shot as the
            # post-rerack break so the existing scatter bonus fires.
            if self._drill_type == 'railbreak':
                self._post_rerack_break_pending = True
                self._break_ball_id_after_rerack = 15
            self.curriculum_shot = 0
            self.key_pocketed = False
            self.break_pocketed = False
            self.curriculum_success = False
            return self.get_obs()
        # Regular Phase 7 episode — full rack + auto-break.
        self._is_curriculum_episode = False
        self._is_rail_drill = False
        self._drill_type = 'regular'
        # Skip Phase9CurriculumEnv.reset; go straight to Phase7Env.reset.
        # (Phase7Env's reset does sample_phase6_setup + _execute_opening_break.)
        return Phase7Env.reset(self)

    def _sample_3ball_drill_setup(self):
        """Place 3 balls + cue for end-of-rack drill: key1 → key2 → break.

        Layout:
        - break ball (id 3): 8-15″ from rack apex along ±y (Phase9-style)
        - key2 (id 2): 15-25″ from break ball, positioned to allow shape
          between them; placed left of rack so cue can travel naturally
        - key1 (id 1): 15-25″ from key2, accessible from cue's start
        - cue: head kitchen area, with verified legal shot to key1

        Existing rewards (pocket_reward, next_shape_bonus, eor_bonus) will
        teach the right sequence:
        - Pocket key1, leave easy shot on key2 → +next_shape_bonus
        - Pocket key2, leave easy shot on break ball → +next_shape_bonus
        - Pocket key2 triggers rerack (len==1 after) → rack placed
        - Shoot break ball at rerack → +eor_bonus for scatter
        """
        from shot_enumerator import generate_legal_shots
        apex_x, apex_y = RACK_APEX
        R_BALL = R
        for _trial in range(80):
            # Break ball: alongside the rack
            offset_dir = random.choice([+1, -1])
            break_dist = random.uniform(8, 14)
            break_x = apex_x + random.uniform(-2, 2)
            break_y = apex_y + offset_dir * break_dist
            break_y = max(R_BALL + 3, min(TABLE_WIDTH - R_BALL - 3, break_y))
            # Key2: between rack and the cue start area, 15-25″ from break
            for _t2 in range(20):
                k2x = random.uniform(35, 70)
                k2y = random.uniform(R_BALL + 5, TABLE_WIDTH - R_BALL - 5)
                if math.hypot(k2x - break_x, k2y - break_y) > 14:
                    break
            # Key1: closer to head, 15-25″ from key2
            for _t1 in range(20):
                k1x = random.uniform(15, 45)
                k1y = random.uniform(R_BALL + 5, TABLE_WIDTH - R_BALL - 5)
                if math.hypot(k1x - k2x, k1y - k2y) > 12:
                    break
            # Cue: head kitchen
            for _tc in range(20):
                cx = random.uniform(10, 30)
                cy = random.uniform(R_BALL + 5, TABLE_WIDTH - R_BALL - 5)
                balls = {1: [k1x, k1y], 2: [k2x, k2y], 3: [break_x, break_y]}
                # Need a legal shot at key1 (ball id 1) from this cue position.
                legal = generate_legal_shots([cx, cy], balls, max_cut_deg=80.0)
                if any(s.ball_id == 1 for s in legal):
                    return [cx, cy], balls
        # Fallback if all trials failed — return last attempt.
        return [cx, cy], balls

    def _sample_railbreak_drill_setup(self):
        """Place the 14 reracked balls + 1 break ball near a rail + cue
        positioned for the break shot. Used by v23 to teach high-force
        commitment when the break ball is on a rail — the network's
        learned rail-shot prior (low force = safe pocket) competes with
        the post-rerack-break scatter bonus, and without targeted
        examples the rail prior wins.

        Sets self._break_ball_id_after_rerack so the existing
        _eor_bonus(was_post_rerack_break=True) scatter calculation runs
        correctly on the first shot.
        """
        from shot_enumerator import generate_legal_shots
        R_BALL = R
        rack_positions = RACK_POSITIONS[1:]   # 14 positions
        break_ball_id = 15
        for _trial in range(80):
            rail = random.choice(['top', 'bottom', 'left', 'right'])
            if rail == 'top':
                bb_x = random.uniform(20, 80)
                bb_y = R_BALL + random.uniform(0.1, 3.0)
                cx = bb_x + random.uniform(-15, 15)
                cy = random.uniform(20, 40)
            elif rail == 'bottom':
                bb_x = random.uniform(20, 80)
                bb_y = TABLE_WIDTH - R_BALL - random.uniform(0.1, 3.0)
                cx = bb_x + random.uniform(-15, 15)
                cy = random.uniform(10, 30)
            elif rail == 'left':
                bb_x = R_BALL + random.uniform(0.1, 3.0)
                bb_y = random.uniform(15, 35)
                cx = random.uniform(20, 55)
                cy = bb_y + random.uniform(-12, 12)
            else:  # right
                bb_x = TABLE_LENGTH - R_BALL - random.uniform(0.1, 3.0)
                bb_y = random.uniform(15, 35)
                cx = random.uniform(35, 65)
                cy = bb_y + random.uniform(-12, 12)
            # Skip if break ball would overlap a rack position.
            if any(math.hypot(bb_x - rx, bb_y - ry) < 2 * R_BALL + 0.1
                   for rx, ry in rack_positions):
                continue
            # Clamp cue inside playable area, away from rack and break ball.
            cx = max(R_BALL + 3, min(TABLE_LENGTH - R_BALL - 3, cx))
            cy = max(R_BALL + 3, min(TABLE_WIDTH - R_BALL - 3, cy))
            if any(math.hypot(cx - rx, cy - ry) < 3 * R_BALL
                   for rx, ry in rack_positions):
                continue
            if math.hypot(cx - bb_x, cy - bb_y) < 3 * R_BALL:
                continue
            balls = {}
            rack_ids = [i for i in range(1, 16) if i != break_ball_id]
            for bid, pos in zip(rack_ids, rack_positions):
                balls[bid] = list(pos)
            balls[break_ball_id] = [bb_x, bb_y]
            cue = [cx, cy]
            legal = generate_legal_shots(cue, balls, max_cut_deg=80.0)
            if any(s.ball_id == break_ball_id for s in legal):
                self._break_ball_id_after_rerack = break_ball_id
                return cue, balls
        # Fallback — return last attempt regardless.
        self._break_ball_id_after_rerack = break_ball_id
        return cue, balls

    def _sample_rail_drill_setup(self):
        """Place 1 ball within ~1.5″ of a randomly chosen rail, with the cue
        positioned for a moderate-cut shot to the closer corner pocket.
        Verifies that at least one legal shot exists; retries up to 60 times.
        Used by v11 to break the rail-shot aversion observed in v8-v10."""
        from shot_enumerator import generate_legal_shots
        R_BALL = R
        for _trial in range(60):
            rail = random.choice(['top', 'bottom', 'left', 'right'])
            if rail == 'top':
                x_ob = random.uniform(15, 85)
                y_ob = R_BALL + random.uniform(0.1, 1.5)
                cue = (x_ob + random.uniform(-15, 15),
                       random.uniform(15, 40))
            elif rail == 'bottom':
                x_ob = random.uniform(15, 85)
                y_ob = TABLE_WIDTH - R_BALL - random.uniform(0.1, 1.5)
                cue = (x_ob + random.uniform(-15, 15),
                       random.uniform(10, 35))
            elif rail == 'left':
                x_ob = R_BALL + random.uniform(0.1, 1.5)
                y_ob = random.uniform(10, 40)
                cue = (random.uniform(15, 35),
                       y_ob + random.uniform(-12, 12))
            else:  # right
                x_ob = TABLE_LENGTH - R_BALL - random.uniform(0.1, 1.5)
                y_ob = random.uniform(10, 40)
                cue = (random.uniform(65, 85),
                       y_ob + random.uniform(-12, 12))
            # Clamp cue inside the table.
            cx = max(R_BALL + 3, min(TABLE_LENGTH - R_BALL - 3, cue[0]))
            cy = max(R_BALL + 3, min(TABLE_WIDTH - R_BALL - 3, cue[1]))
            cue = (cx, cy)
            ob_pos = (x_ob, y_ob)
            balls = {1: list(ob_pos)}
            # Verify at least one legal shot exists.
            legal = generate_legal_shots(list(cue), balls, max_cut_deg=80.0)
            if legal:
                return list(cue), balls
        # Fallback (rare) — return last attempt regardless.
        return list(cue), balls

    def step(self, *args, **kwargs):
        if not self._is_curriculum_episode:
            # Pure Phase 7 step — no curriculum bonuses.
            return Phase7Env.step(self, *args, **kwargs)
        drill = getattr(self, '_drill_type', 'key_break')
        if drill in ('rail', '3ball', 'railbreak'):
            # Rail-drill and 3-ball-EOR drill use plain Phase 7 step. The
            # existing reward signals (pocket_reward, next_shape_bonus,
            # eor_bonus, rail_shot_bonus) provide the curriculum signal
            # through the strategically-placed starting state. No new
            # reward shaping needed — exposure to these states is the lift.
            return Phase7Env.step(self, *args, **kwargs)
        # Original key+break curriculum: delegate to Phase9CurriculumEnv.step
        # which applies the key/break/scatter/valid_after bonuses.
        return super().step(*args, **kwargs)
