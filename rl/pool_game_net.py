"""
Token-based pool game net. Attention over balls, pockets, and legal shots.

Tokens:
  - Cue ball: (x, y)
  - Object balls: (x, y)  — one per ball on the table, positions only
  - Pockets: (x, y, is_corner)
  - Legal shots: (ghost_x, ghost_y, target_ball_x, target_ball_y,
                  target_pocket_x, target_pocket_y, cut_angle_norm,
                  cue_to_ghost_dist_norm, ball_to_pocket_dist_norm)

Every ball token has the same shape. The "cue vs object" distinction is only
the type embedding (no strategic features baked in). Same principle for
legal shots: only raw geometry, no pre-labeled "difficulty" or "break ball"
features — those concepts are left to the network to learn if they're useful.

Output per legal shot token:
  - score logit (softmax across legal shots)
  - force_mean (pre-sigmoid)
  - spin_mean (pre-tanh)

Plus a value head pooled from non-shot tokens.

Action:
  shot_idx ∈ {0 .. N_legal - 1}   discrete
  force_raw, spin_raw              continuous, sampled from Normal(means[shot_idx], std)
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0

# Action output scales
FORCE_LO = 50.0
FORCE_HI = 250.0
SPIN_MAX = 1.5    # was 2.0; lowered to limit max-draw + max-force combinations

# Padding limits
MAX_BALLS = 16   # cue + 15 object
MAX_POCKETS = 6
MAX_SHOTS = 60   # 15 balls × 6 pockets is 90 theoretical; legal set is much smaller


def decode_force(raw):
    return FORCE_LO + (FORCE_HI - FORCE_LO) / (1.0 + math.exp(-float(raw)))


def decode_spin(raw):
    return SPIN_MAX * math.tanh(float(raw))


# ── Observation struct (Python-side, converted to tensors in the net) ─────

@dataclass
class Phase7Obs:
    balls: np.ndarray          # (MAX_BALLS, 2)  — normalized (x/TL, y/TW); -1,-1 for absent
    ball_mask: np.ndarray      # (MAX_BALLS,)    — 1 where token is valid (on table)
    ball_is_cue: np.ndarray    # (MAX_BALLS,)    — 1 for cue, 0 otherwise
    pockets: np.ndarray        # (MAX_POCKETS, 3) — (x/TL, y/TW, is_corner)
    shots: np.ndarray          # (MAX_SHOTS, 10)  — per-shot features
    shot_mask: np.ndarray      # (MAX_SHOTS,)    — 1 where shot is valid
    shot_meta: list            # list of LegalShot objects (for decode/debug)


# ── Network ───────────────────────────────────────────────────────────────

class PoolGameNet(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, ff_dim=None):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 2
        self.embed_dim = embed_dim

        # Encoders per token type.
        self.ball_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),   # x, y, is_cue
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.pocket_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.shot_encoder = nn.Sequential(
            nn.Linear(10, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        # Type embedding: 0=ball, 1=pocket, 2=shot
        self.type_embed = nn.Embedding(3, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            activation='gelu', batch_first=True, norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-shot output head: score, force_mean, spin_mean.
        self.shot_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        # Value head: consumes mean-pooled non-shot token embedding.
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        # Learnable log std for force and spin (shared across shots).
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for the per-shot output (so initial scores are ~uniform
        # and initial force/spin means are ~0).
        nn.init.orthogonal_(self.shot_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.shot_head[-1].bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, balls, ball_mask, ball_is_cue, pockets,
                shots, shot_mask):
        """
        Args (all batched):
            balls:       (B, MAX_BALLS, 2)       float
            ball_mask:   (B, MAX_BALLS)          bool — True where valid
            ball_is_cue: (B, MAX_BALLS)          float 1/0
            pockets:     (B, MAX_POCKETS, 3)     float
            shots:       (B, MAX_SHOTS, 9)       float
            shot_mask:   (B, MAX_SHOTS)          bool — True where valid

        Returns:
            shot_scores: (B, MAX_SHOTS) — logits, -inf where mask is False
            force_means: (B, MAX_SHOTS)
            spin_means:  (B, MAX_SHOTS)
            value:       (B,)
        """
        B = balls.shape[0]
        device = balls.device

        # Ball tokens: concatenate (x, y, is_cue) and embed.
        ball_in = torch.cat([balls, ball_is_cue.unsqueeze(-1)], dim=-1)      # (B, MAX_BALLS, 3)
        ball_tok = self.ball_encoder(ball_in) + self.type_embed(
            torch.zeros(MAX_BALLS, dtype=torch.long, device=device))

        pocket_tok = self.pocket_encoder(pockets) + self.type_embed(
            torch.full((MAX_POCKETS,), 1, dtype=torch.long, device=device))

        shot_tok = self.shot_encoder(shots) + self.type_embed(
            torch.full((MAX_SHOTS,), 2, dtype=torch.long, device=device))

        tokens = torch.cat([ball_tok, pocket_tok, shot_tok], dim=1)          # (B, T, D)

        # Build the padding mask. Pockets are always valid. src_key_padding_mask
        # is True where positions should be IGNORED by attention.
        pad_all = torch.cat([
            ~ball_mask,
            torch.zeros(B, MAX_POCKETS, dtype=torch.bool, device=device),
            ~shot_mask,
        ], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=pad_all)

        # Split outputs.
        ball_enc = encoded[:, :MAX_BALLS, :]
        pocket_enc = encoded[:, MAX_BALLS:MAX_BALLS + MAX_POCKETS, :]
        shot_enc = encoded[:, MAX_BALLS + MAX_POCKETS:, :]                   # (B, MAX_SHOTS, D)

        # Per-shot outputs.
        shot_out = self.shot_head(shot_enc)                                  # (B, MAX_SHOTS, 3)
        shot_scores = shot_out[..., 0]
        force_means = shot_out[..., 1]
        spin_means = shot_out[..., 2]
        shot_scores = shot_scores.masked_fill(~shot_mask, -1e9)

        # Value head: pool over valid non-shot tokens (balls + pockets).
        nonshot_enc = torch.cat([ball_enc, pocket_enc], dim=1)               # (B, MAX_B+MAX_P, D)
        nonshot_valid = torch.cat([
            ball_mask,
            torch.ones(B, MAX_POCKETS, dtype=torch.bool, device=device),
        ], dim=1).unsqueeze(-1).float()
        pooled = (nonshot_enc * nonshot_valid).sum(1) / nonshot_valid.sum(1).clamp(min=1)
        value = self.value_head(pooled).squeeze(-1)

        return shot_scores, force_means, spin_means, value

    # ── Action sampling / evaluation ─────────────────────────────────────

    def get_action(self, obs_batch, deterministic=False):
        """
        Args:
            obs_batch: dict of batched tensors (see forward args)
            deterministic: if True, argmax shot + mean force/spin
        Returns:
            shot_idx: (B,) long
            force_raw: (B,) — raw (pre-sigmoid) force output for chosen shot
            spin_raw:  (B,) — raw (pre-tanh) spin output for chosen shot
            log_prob: (B,) — joint log-prob
            value:    (B,)
        """
        scores, f_means, s_means, value = self.forward(**obs_batch)
        force_std = torch.exp(self.log_std[0])
        spin_std = torch.exp(self.log_std[1])

        if deterministic:
            shot_idx = scores.argmax(dim=-1)
            # Gather per-shot means.
            f_mu = f_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            s_mu = s_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            force_raw, spin_raw = f_mu, s_mu
            log_prob = torch.zeros_like(value)
        else:
            cat = Categorical(logits=scores)
            shot_idx = cat.sample()
            log_p_shot = cat.log_prob(shot_idx)
            f_mu = f_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            s_mu = s_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            force_dist = Normal(f_mu, force_std)
            spin_dist = Normal(s_mu, spin_std)
            force_raw = force_dist.sample()
            spin_raw = spin_dist.sample()
            log_prob = log_p_shot + force_dist.log_prob(force_raw) + spin_dist.log_prob(spin_raw)
        return shot_idx, force_raw, spin_raw, log_prob, value

    def evaluate_actions(self, obs_batch, shot_idx, force_raw, spin_raw):
        """For PPO update. Returns (log_prob, entropy, value) for given actions."""
        scores, f_means, s_means, value = self.forward(**obs_batch)
        force_std = torch.exp(self.log_std[0])
        spin_std = torch.exp(self.log_std[1])

        cat = Categorical(logits=scores)
        log_p_shot = cat.log_prob(shot_idx)

        f_mu = f_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
        s_mu = s_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
        force_dist = Normal(f_mu, force_std)
        spin_dist = Normal(s_mu, spin_std)

        log_prob = (log_p_shot
                    + force_dist.log_prob(force_raw)
                    + spin_dist.log_prob(spin_raw))
        entropy = cat.entropy() + force_dist.entropy() + spin_dist.entropy()
        return log_prob, entropy, value


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    net = PoolGameNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Params: {n_params:,}')
    B = 4
    balls = torch.randn(B, MAX_BALLS, 2)
    ball_mask = torch.ones(B, MAX_BALLS, dtype=torch.bool)
    ball_is_cue = torch.zeros(B, MAX_BALLS)
    ball_is_cue[:, 0] = 1.0
    pockets = torch.randn(B, MAX_POCKETS, 3)
    shots = torch.randn(B, MAX_SHOTS, 10)
    # Randomly set some shots invalid to test masking
    shot_mask = torch.ones(B, MAX_SHOTS, dtype=torch.bool)
    shot_mask[:, 10:] = False  # only first 10 valid
    obs = dict(balls=balls, ball_mask=ball_mask, ball_is_cue=ball_is_cue,
               pockets=pockets, shots=shots, shot_mask=shot_mask)
    idx, f, s, lp, v = net.get_action(obs)
    print(f'shot_idx: {idx.tolist()}  force_raw: {f.tolist()}  spin_raw: {s.tolist()}')
    print(f'log_prob: {lp.tolist()}  value: {v.tolist()}')
    lp2, ent, v2 = net.evaluate_actions(obs, idx, f, s)
    print(f'eval log_prob close: {torch.allclose(lp, lp2)}  entropy: {ent.tolist()}')
