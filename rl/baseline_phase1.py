"""
Measure actual random-policy hit rate for Phase 1. No learning, just rollouts.

Four conditions (2x2):
  action:  (A) uniform angle in [0, 2pi)     (B) (sin, cos) ~ Normal(0, 0.6), arctan2'd
  env:     (1) current (ball drifts, reset on pocket)   (2) reset every shot

If A-2 ~ 15% but A-1 << 15%, the env drift kills the observed hit rate.
If A-2 ~ 15% but B-2 << 15%, the (sin, cos) -> arctan2 parametrization is the issue.
"""
import numpy as np
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_curriculum import Phase1Env


def rollout(action_mode, env_mode, num_shots=3200, seed=0):
    rng = np.random.default_rng(seed)
    env = Phase1Env()
    env.reset()
    hits = 0
    for _ in range(num_shots):
        if action_mode == 'uniform':
            aim_angle = rng.uniform(-math.pi, math.pi)
        elif action_mode == 'sincos':
            s = rng.normal(0, 0.6)
            c = rng.normal(0, 0.6)
            aim_angle = math.atan2(s, c)
        else:
            raise ValueError(action_mode)
        _, done, info = env.step(aim_angle, 30.0, 0.0)
        if info['hit']:
            hits += 1
        if env_mode == 'reset_each':
            env.reset()
        elif done:  # env_mode == 'current'
            env.reset()
    return hits / num_shots


if __name__ == '__main__':
    print('Measured random hit rates (3200 shots each):')
    print()
    for action_mode in ('uniform', 'sincos'):
        for env_mode in ('reset_each', 'current'):
            hr = rollout(action_mode, env_mode)
            print(f'  action={action_mode:8s} env={env_mode:11s} -> {hr:.1%}')
