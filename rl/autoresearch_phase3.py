"""
Overnight orchestrator for Phase 3 hyperparameter/reward experiments.

Design:
- A fixed list of configs is declared below. Each config is one trial.
- For each config: run train_phase3.py as a subprocess with those CLI args,
  then deterministic-eval the resulting checkpoint, then log a TSV row.
- Results go to results_phase3.tsv (append-only). On restart, trials whose
  tag already appears in the TSV are skipped, so the script is resumable.
- All checkpoints are saved to checkpoints/phase3_<tag>_best.pt etc.

Each trial targets ~25-35 min (1500 iters) so an ~8-hour overnight gets
through 12-16 trials comfortably.

Usage:
    python autoresearch_phase3.py

The process prints progress as it goes; kill with Ctrl-C and results so far
are preserved in the TSV. Individual training logs are at autoresearch/<tag>.log.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / 'autoresearch'
LOG_DIR.mkdir(exist_ok=True)
TSV = HERE / 'results_phase3.tsv'


@dataclass
class Config:
    tag: str
    description: str
    iters: int = 1500
    warm: str = 'checkpoints/phase3approach1_best.pt'  # current best Phase 3
    cut: float = 60.0
    log_std_min: float = -3.0
    entropy_coef: float = 0.01
    lr: float = 3e-4
    steps_per_update: int = 32
    envs: int = 32
    pocket_reward: float = 10.0
    proximity_reward: float = 1.5
    miss_shape: str = 'approach'          # new default (beat linear by +14pp)
    gauss_sigma_deg: float = 2.0
    approach_sigma_in: float = 0.3
    hit_shape: str = 'straight_line'      # new default
    hit_sigma_in: float = 2.5

    def cli_args(self) -> list[str]:
        return [
            '--tag', self.tag,
            '--iters', str(self.iters),
            '--warm', self.warm,
            '--cut', str(self.cut),
            '--log_std_min', str(self.log_std_min),
            '--entropy_coef', str(self.entropy_coef),
            '--lr', str(self.lr),
            '--steps_per_update', str(self.steps_per_update),
            '--envs', str(self.envs),
            '--pocket_reward', str(self.pocket_reward),
            '--proximity_reward', str(self.proximity_reward),
            '--miss_shape', self.miss_shape,
            '--gauss_sigma_deg', str(self.gauss_sigma_deg),
            '--approach_sigma_in', str(self.approach_sigma_in),
            '--hit_shape', self.hit_shape,
            '--hit_sigma_in', str(self.hit_sigma_in),
        ]


# ─── Experiment queue ─────────────────────────────────────────────────────
#
# Axes we want to explore (based on diagnosis that policy is stuck at ~1 deg
# aim precision and signal/noise in the middle regime is the bottleneck):
#   - variance reduction: more samples per update, slower LR
#   - reward reshaping: Gaussian miss + larger pocket reward
#   - exploration floor: tighter or looser log_std_min
#   - warm-start source
# Each trial ~25-40 min on CPU; 12 trials ~6-8 hours overnight.

CONFIGS = [
    # Reference: reproduce approach1 settings (our 75.1% baseline).
    Config(tag='or_ref',
           description='approach+straight_line reproduce, warm from approach1 best'),

    # --- Axis 1: tighten sigmas (sub-inch / sub-2-inch precision) ---
    Config(tag='or_sig_0_20',
           description='approach_sigma=0.20 in (tighter cue-to-ghost)',
           approach_sigma_in=0.20),
    Config(tag='or_sig_0_25',
           description='approach_sigma=0.25 in',
           approach_sigma_in=0.25),
    Config(tag='or_hit_sig_1_5',
           description='hit_sigma=1.5 in (tighter obj-traj-to-pocket)',
           hit_sigma_in=1.5),
    Config(tag='or_hit_sig_4',
           description='hit_sigma=4.0 in (looser; is current too sharp?)',
           hit_sigma_in=4.0),

    # --- Axis 2: exploration floor ---
    Config(tag='or_floor_3_5',
           description='log_std_min=-3.5 (std ~0.030)',
           log_std_min=-3.5),
    Config(tag='or_floor_2_5',
           description='log_std_min=-2.5 (looser exploration)',
           log_std_min=-2.5),

    # --- Axis 3: LR and batch ---
    Config(tag='or_slowlr',
           description='lr=1e-4',
           lr=1e-4),
    Config(tag='or_bigbatch',
           description='4x samples per update (128 steps)',
           steps_per_update=128),

    # --- Axis 4: pocket reward scaling ---
    Config(tag='or_pocket_20',
           description='Pocket reward 20 (2x)',
           pocket_reward=20.0),

    # --- Axis 5: duration & warm source ---
    Config(tag='or_longer',
           description='3000 iters with baseline settings',
           iters=3000),
    Config(tag='or_fresh',
           description='Warm from phase2_best (fresh start on cut geometry)',
           warm='checkpoints/phase2_best.pt'),

    # --- Moderate combined improvements ---
    Config(tag='or_combo_gentle',
           description='sigma=0.2 + pkt=15 + longer',
           approach_sigma_in=0.20, pocket_reward=15.0, iters=2500),
    Config(tag='or_combo_batch_pocket',
           description='bigbatch + pocket=20',
           steps_per_update=128, pocket_reward=20.0),
]


# ─── Helpers ──────────────────────────────────────────────────────────────

TSV_HEADER = (
    'tag\tdescription\tstoch_best_pocket\tdet_pocket_60\tdet_hr_60\t'
    'det_pocket_30\tmedian_aim_err_deg\tpeak_ent\twall_secs\tckpt_file\n'
)


def read_done_tags() -> set[str]:
    if not TSV.exists():
        return set()
    done = set()
    with TSV.open() as f:
        next(f, None)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts:
                done.add(parts[0])
    return done


def ensure_tsv():
    if not TSV.exists():
        TSV.write_text(TSV_HEADER)


def log_row(fields: dict):
    cols = ['tag', 'description', 'stoch_best_pocket', 'det_pocket_60',
            'det_hr_60', 'det_pocket_30', 'median_aim_err_deg', 'peak_ent',
            'wall_secs', 'ckpt_file']
    row = '\t'.join(str(fields.get(c, '')) for c in cols) + '\n'
    with TSV.open('a') as f:
        f.write(row)


def parse_stoch_best(train_log: Path) -> tuple[float, float]:
    """Return (best_stoch_avg_pocket, peak_entropy) from a training log."""
    best = 0.0
    peak_ent = float('inf')
    if not train_log.exists():
        return (0.0, 0.0)
    for line in train_log.read_text().splitlines():
        # Final "Done. Best avg pocket rate: 38.0%" or "Best hit rate:"
        if line.startswith('Done.') and 'pocket rate' in line.lower():
            try:
                pct = line.split(':')[1].strip().rstrip('%').split()[0]
                best = max(best, float(pct) / 100.0)
            except Exception:
                pass
        # Track minimum entropy we saw (negative = more confident)
        if 'Ent=' in line:
            try:
                ent = float(line.split('Ent=')[1].split()[0])
                peak_ent = min(peak_ent, ent)
            except Exception:
                pass
    if peak_ent == float('inf'):
        peak_ent = 0.0
    return best, peak_ent


def run_trial(cfg: Config) -> dict:
    """Run one trial end-to-end. Returns fields for TSV row."""
    log_path = LOG_DIR / f'{cfg.tag}.log'
    ckpt = HERE / 'checkpoints' / f'phase3{cfg.tag}_best.pt'
    latest = HERE / 'checkpoints' / f'phase3{cfg.tag}_latest.pt'
    t0 = time.time()
    print(f'\n[{time.strftime("%H:%M:%S")}] >>> Running {cfg.tag}: {cfg.description}', flush=True)

    # Spawn training subprocess. Stream stdout to the log file.
    cmd = [sys.executable, str(HERE / 'train_phase3.py'),
           '--device', 'cpu'] + cfg.cli_args()
    with log_path.open('w') as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=HERE)
    wall = time.time() - t0

    if proc.returncode != 0:
        print(f'  TRIAL FAILED with exit {proc.returncode}', flush=True)
        return dict(tag=cfg.tag, description=cfg.description,
                    stoch_best_pocket=0.0, det_pocket_60=0.0,
                    det_hr_60=0.0, det_pocket_30=0.0,
                    median_aim_err_deg=0.0, peak_ent=0.0,
                    wall_secs=int(wall), ckpt_file='CRASH')

    stoch_best, peak_ent = parse_stoch_best(log_path)
    print(f'  training done in {wall:.0f}s. stoch_best={stoch_best:.1%} peak_ent={peak_ent:.2f}', flush=True)

    # Evaluate the LATEST checkpoint (end of training). The "best" is saved
    # at stochastic peak which we already know is unreliable — using
    # latest gives a consistent final-state view.
    eval_ckpt = latest if latest.exists() else ckpt
    if not eval_ckpt.exists():
        print(f'  no checkpoint found', flush=True)
        return dict(tag=cfg.tag, description=cfg.description,
                    stoch_best_pocket=stoch_best, det_pocket_60=0.0,
                    det_hr_60=0.0, det_pocket_30=0.0,
                    median_aim_err_deg=0.0, peak_ent=peak_ent,
                    wall_secs=int(wall), ckpt_file='MISSING')

    det_metrics = evaluate_checkpoint(eval_ckpt)
    return dict(
        tag=cfg.tag, description=cfg.description,
        stoch_best_pocket=round(stoch_best, 4),
        det_pocket_60=round(det_metrics['pocket_60'], 4),
        det_hr_60=round(det_metrics['hr_60'], 4),
        det_pocket_30=round(det_metrics['pocket_30'], 4),
        median_aim_err_deg=round(det_metrics['median_err'], 2),
        peak_ent=round(peak_ent, 3),
        wall_secs=int(wall),
        ckpt_file=eval_ckpt.name,
    )


def evaluate_checkpoint(ckpt_path: Path, num_shots: int = 800) -> dict:
    """Deterministic eval on Phase3Env at both cut=60 and cut=30."""
    # Lazy-import heavy stuff so orchestrator import is cheap
    import numpy as np
    import torch
    sys.path.insert(0, str(HERE))
    from pool_attention_net import PoolAttentionNet
    from train_phase3 import Phase3Env, ghost_ball

    device = torch.device('cpu')
    net = PoolAttentionNet(embed_dim=96, num_heads=6, num_layers=4, act_dim=2).to(device)
    net.log_std = torch.nn.Parameter(torch.full((2,), -0.5).to(device))
    try:
        net.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
    except Exception as e:
        print(f'  eval load failed: {e}', flush=True)
        return dict(pocket_60=0.0, hr_60=0.0, pocket_30=0.0, median_err=90.0)
    net.eval()

    def run_env(cut):
        env = Phase3Env(max_cut_deg=cut)
        h = p = 0
        errs = []
        with torch.no_grad():
            for _ in range(num_shots):
                obs = env.reset()
                t = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _ = net.get_action(t, deterministic=True)
                a = action[0].cpu().numpy()
                aim_angle = math.atan2(float(a[0]), float(a[1]))
                g = ghost_ball(env.ball_pos, env.target_pocket)
                ghost_angle = math.atan2(g[1] - env.cue[1], g[0] - env.cue[0])
                err = abs(aim_angle - ghost_angle)
                if err > math.pi:
                    err = 2 * math.pi - err
                errs.append(math.degrees(err))
                _, _, info = env.step(aim_angle)
                if info['hit']: h += 1
                if info['pocketed']: p += 1
        return h / num_shots, p / num_shots, float(np.median(errs))

    hr60, p60, med60 = run_env(60.0)
    _, p30, _ = run_env(30.0)
    return dict(pocket_60=p60, hr_60=hr60, pocket_30=p30, median_err=med60)


# ─── Main loop ────────────────────────────────────────────────────────────

def main():
    ensure_tsv()
    done = read_done_tags()
    configs_to_run = [c for c in CONFIGS if c.tag not in done]

    print(f'[{time.strftime("%H:%M:%S")}] autoresearch starting')
    print(f'  {len(CONFIGS)} total configs, {len(done)} done, '
          f'{len(configs_to_run)} to run')
    print(f'  TSV: {TSV}')
    print(f'  per-trial logs: {LOG_DIR}')

    t_start = time.time()
    for i, cfg in enumerate(configs_to_run, 1):
        print(f'\n=== Trial {i}/{len(configs_to_run)}: {cfg.tag} ===', flush=True)
        try:
            row = run_trial(cfg)
            log_row(row)
            elapsed_h = (time.time() - t_start) / 3600
            print(f'  logged: det_p60={row["det_pocket_60"]} det_p30={row["det_pocket_30"]} '
                  f'aim_err={row["median_aim_err_deg"]} deg  (elapsed {elapsed_h:.1f}h)',
                  flush=True)
        except KeyboardInterrupt:
            print('  interrupted', flush=True)
            raise
        except Exception as e:
            print(f'  orchestrator error: {e}', flush=True)
            log_row(dict(tag=cfg.tag, description=cfg.description,
                         stoch_best_pocket=0, det_pocket_60=0, det_hr_60=0,
                         det_pocket_30=0, median_aim_err_deg=0, peak_ent=0,
                         wall_secs=0, ckpt_file=f'ERROR: {e}'))

    print(f'\n[{time.strftime("%H:%M:%S")}] DONE. Total {(time.time()-t_start)/3600:.1f}h. '
          f'Results: {TSV}')


if __name__ == '__main__':
    main()
