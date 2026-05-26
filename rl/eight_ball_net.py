"""
Token-based 8-ball pool network. Adapted from PoolGameNet (14.1) with:
  - Ball group encoding (mine/theirs/8-ball) instead of is_cue
  - Game-state context token (remaining counts, open table, ball-in-hand, etc.)
  - Safety action head (one extra logit appended to shot scores)
  - Sigmoid value head (win probability, not unbounded run length)

The observation is always from the current player's perspective: "mine"
means the acting player's group regardless of solids/stripes.
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

FORCE_LO = 50.0
FORCE_HI = 250.0
SPIN_MAX = 1.5

MAX_BALLS = 16   # cue + 15 object
MAX_POCKETS = 6
MAX_SHOTS = 60

# Ball group encoding (float feature on each ball token)
GROUP_CUE = 0.0
GROUP_MINE = 0.33
GROUP_NEUTRAL = 0.5   # open table
GROUP_THEIRS = 0.67
GROUP_8BALL = 1.0

# Game-state feature count
GAME_STATE_DIM = 8


def decode_force(raw):
    return FORCE_LO + (FORCE_HI - FORCE_LO) / (1.0 + math.exp(-float(raw)))


def decode_spin(raw):
    return SPIN_MAX * math.tanh(float(raw))


@dataclass
class EightBallObs:
    balls: np.ndarray          # (MAX_BALLS, 2)  — normalized positions; -1,-1 for absent
    ball_mask: np.ndarray      # (MAX_BALLS,)    — True where on table
    ball_group: np.ndarray     # (MAX_BALLS,)    — group encoding float
    pockets: np.ndarray        # (MAX_POCKETS, 3) — (x/TL, y/TW, is_corner)
    game_state: np.ndarray     # (GAME_STATE_DIM,) — global context features
    shots: np.ndarray          # (MAX_SHOTS, 10) — per-shot features (9 geom + is_bank)
    shot_mask: np.ndarray      # (MAX_SHOTS,)    — True where shot is valid
    shot_meta: list            # list of LegalShot objects


class EightBallNet(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, ff_dim=None):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 2
        self.embed_dim = embed_dim

        # Ball token: (x, y, ball_group) → 3 features + 1 group = 4
        # We concatenate (x, y) with (ball_group,) to get 3 input features,
        # but ball_group replaces is_cue, so input is (x, y, group) = 3.
        # Actually: we use 4 features: (x, y, is_cue_flag, ball_group)
        # No — keep it minimal: (x, y, ball_group) = 3 features.
        # The group float already encodes cue (0.0) vs mine (0.33) etc.
        self.ball_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
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
        self.game_state_encoder = nn.Sequential(
            nn.Linear(GAME_STATE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        # Type embedding: 0=ball, 1=pocket, 2=shot, 3=game_state
        self.type_embed = nn.Embedding(4, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            activation='gelu', batch_first=True, norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-shot output: score logit, force_mean, spin_mean
        self.shot_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        # Safety head: single logit from pooled non-shot tokens
        self.safety_head = nn.Linear(embed_dim, 1)
        # Value head: win probability via sigmoid
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        # Placement head: (x, y) in [0, 1] via sigmoid for ball-in-hand
        self.placement_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )
        # Learnable log std for force and spin
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        # Learnable log std for placement (x, y) — starts wider for exploration
        self.placement_log_std = nn.Parameter(torch.full((2,), -1.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.shot_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.shot_head[-1].bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.orthogonal_(self.safety_head.weight, gain=0.01)
        nn.init.constant_(self.safety_head.bias, -5.0)
        nn.init.orthogonal_(self.placement_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.placement_head[-1].bias)

    def forward(self, balls, ball_mask, ball_group, pockets,
                game_state, shots, shot_mask):
        B = balls.shape[0]
        device = balls.device

        # Ball tokens: (x, y, group)
        ball_in = torch.cat([balls, ball_group.unsqueeze(-1)], dim=-1)
        ball_tok = self.ball_encoder(ball_in) + self.type_embed(
            torch.zeros(MAX_BALLS, dtype=torch.long, device=device))

        pocket_tok = self.pocket_encoder(pockets) + self.type_embed(
            torch.full((MAX_POCKETS,), 1, dtype=torch.long, device=device))

        shot_tok = self.shot_encoder(shots) + self.type_embed(
            torch.full((MAX_SHOTS,), 2, dtype=torch.long, device=device))

        # Game state token: (B, 1, D)
        gs_tok = self.game_state_encoder(game_state).unsqueeze(1) + self.type_embed(
            torch.full((1,), 3, dtype=torch.long, device=device))

        # Total tokens: MAX_BALLS + MAX_POCKETS + 1(game_state) + MAX_SHOTS
        tokens = torch.cat([ball_tok, pocket_tok, gs_tok, shot_tok], dim=1)

        # Padding mask: True = ignore
        pad_all = torch.cat([
            ~ball_mask,
            torch.zeros(B, MAX_POCKETS, dtype=torch.bool, device=device),
            torch.zeros(B, 1, dtype=torch.bool, device=device),  # game_state always valid
            ~shot_mask,
        ], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=pad_all)

        # Split outputs
        n_nonshot = MAX_BALLS + MAX_POCKETS + 1
        nonshot_enc = encoded[:, :n_nonshot, :]
        shot_enc = encoded[:, n_nonshot:, :]

        # Per-shot outputs
        shot_out = self.shot_head(shot_enc)
        shot_scores = shot_out[..., 0]
        force_means = shot_out[..., 1]
        spin_means = shot_out[..., 2]
        shot_scores = shot_scores.masked_fill(~shot_mask, -1e9)

        # Pool non-shot tokens for value + safety heads
        nonshot_valid = torch.cat([
            ball_mask,
            torch.ones(B, MAX_POCKETS, dtype=torch.bool, device=device),
            torch.ones(B, 1, dtype=torch.bool, device=device),
        ], dim=1).unsqueeze(-1).float()
        pooled = (nonshot_enc * nonshot_valid).sum(1) / nonshot_valid.sum(1).clamp(min=1)

        # Safety logit
        safety_logit = self.safety_head(pooled)  # (B, 1)

        # Value: win probability
        value = torch.sigmoid(self.value_head(pooled).squeeze(-1))

        return shot_scores, force_means, spin_means, safety_logit, value, pooled

    def get_placement(self, pooled, deterministic=False):
        raw = self.placement_head(pooled)
        mu = torch.sigmoid(raw)
        std = torch.exp(self.placement_log_std)
        if deterministic:
            xy = mu
            log_prob = torch.zeros(mu.shape[0], device=mu.device)
        else:
            x_dist = Normal(mu[:, 0], std[0])
            y_dist = Normal(mu[:, 1], std[1])
            x_samp = x_dist.sample().clamp(0.0, 1.0)
            y_samp = y_dist.sample().clamp(0.0, 1.0)
            xy = torch.stack([x_samp, y_samp], dim=-1)
            log_prob = x_dist.log_prob(x_samp) + y_dist.log_prob(y_samp)
        return xy, log_prob

    def _evaluate_placement(self, pooled, x_norm, y_norm):
        raw = self.placement_head(pooled)
        mu = torch.sigmoid(raw)
        std = torch.exp(self.placement_log_std)
        x_dist = Normal(mu[:, 0], std[0])
        y_dist = Normal(mu[:, 1], std[1])
        log_prob = x_dist.log_prob(x_norm) + y_dist.log_prob(y_norm)
        entropy = x_dist.entropy() + y_dist.entropy()
        return log_prob, entropy

    def get_action(self, obs_batch, deterministic=False):
        scores, f_means, s_means, safety_logit, value, pooled = self.forward(**obs_batch)

        is_placement = obs_batch['game_state'][:, 5] > 0.5
        if is_placement.any() and is_placement.all():
            xy, log_prob = self.get_placement(pooled, deterministic)
            action_idx = torch.full((xy.shape[0],), -1, dtype=torch.long,
                                    device=xy.device)
            return action_idx, xy[:, 0], xy[:, 1], log_prob, value

        force_std = torch.exp(self.log_std[0])
        spin_std = torch.exp(self.log_std[1])

        combined_logits = torch.cat([scores, safety_logit], dim=-1)

        if deterministic:
            action_idx = combined_logits.argmax(dim=-1)
            is_safety = (action_idx == MAX_SHOTS)
            shot_idx = action_idx.clone()
            shot_idx[is_safety] = 0
            f_mu = f_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            s_mu = s_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            f_mu[is_safety] = 0.0
            s_mu[is_safety] = 0.0
            force_raw, spin_raw = f_mu, s_mu
            log_prob = torch.zeros_like(value)
        else:
            cat = Categorical(logits=combined_logits)
            action_idx = cat.sample()
            log_p_action = cat.log_prob(action_idx)

            is_safety = (action_idx == MAX_SHOTS)
            shot_idx = action_idx.clone()
            shot_idx[is_safety] = 0

            f_mu = f_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            s_mu = s_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            f_mu[is_safety] = 0.0
            s_mu[is_safety] = 0.0

            force_dist = Normal(f_mu, force_std)
            spin_dist = Normal(s_mu, spin_std)
            force_raw = force_dist.sample()
            spin_raw = spin_dist.sample()
            log_prob = log_p_action + force_dist.log_prob(force_raw) + spin_dist.log_prob(spin_raw)

        return action_idx, force_raw, spin_raw, log_prob, value

    def evaluate_actions(self, obs_batch, action_idx, force_raw, spin_raw,
                         is_placement=None):
        scores, f_means, s_means, safety_logit, value, pooled = self.forward(**obs_batch)

        if is_placement is None:
            is_placement = obs_batch['game_state'][:, 5] > 0.5

        B = scores.shape[0]
        device = scores.device
        log_prob = torch.zeros(B, device=device)
        entropy = torch.zeros(B, device=device)

        p_mask = is_placement
        s_mask = ~is_placement

        if p_mask.any():
            p_lp, p_ent = self._evaluate_placement(
                pooled[p_mask], force_raw[p_mask], spin_raw[p_mask])
            log_prob[p_mask] = p_lp
            entropy[p_mask] = p_ent

        if s_mask.any():
            force_std = torch.exp(self.log_std[0])
            spin_std = torch.exp(self.log_std[1])

            s_scores = scores[s_mask]
            s_safety = safety_logit[s_mask]
            combined_logits = torch.cat([s_scores, s_safety], dim=-1)
            cat = Categorical(logits=combined_logits)
            s_action_idx = action_idx[s_mask]
            log_p_action = cat.log_prob(s_action_idx)

            is_safety = (s_action_idx == MAX_SHOTS)
            shot_idx = s_action_idx.clone()
            shot_idx[is_safety] = 0

            s_f_means = f_means[s_mask]
            s_s_means = s_means[s_mask]
            f_mu = s_f_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            s_mu = s_s_means.gather(1, shot_idx.unsqueeze(-1)).squeeze(-1)
            f_mu[is_safety] = 0.0
            s_mu[is_safety] = 0.0

            force_dist = Normal(f_mu, force_std)
            spin_dist = Normal(s_mu, spin_std)

            s_lp = (log_p_action
                     + force_dist.log_prob(force_raw[s_mask])
                     + spin_dist.log_prob(spin_raw[s_mask]))
            s_ent = cat.entropy() + force_dist.entropy() + spin_dist.entropy()
            log_prob[s_mask] = s_lp
            entropy[s_mask] = s_ent

        return log_prob, entropy, value


if __name__ == '__main__':
    net = EightBallNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Params: {n_params:,}')
    B = 4
    balls = torch.randn(B, MAX_BALLS, 2)
    ball_mask = torch.ones(B, MAX_BALLS, dtype=torch.bool)
    ball_group = torch.zeros(B, MAX_BALLS)
    ball_group[:, 0] = GROUP_CUE
    ball_group[:, 1:8] = GROUP_MINE
    ball_group[:, 8] = GROUP_8BALL
    ball_group[:, 9:] = GROUP_THEIRS
    pockets = torch.randn(B, MAX_POCKETS, 3)
    game_state = torch.zeros(B, GAME_STATE_DIM)
    shots = torch.randn(B, MAX_SHOTS, 10)
    shot_mask = torch.ones(B, MAX_SHOTS, dtype=torch.bool)
    shot_mask[:, 10:] = False
    obs = dict(balls=balls, ball_mask=ball_mask, ball_group=ball_group,
               pockets=pockets, game_state=game_state,
               shots=shots, shot_mask=shot_mask)
    idx, f, s, lp, v = net.get_action(obs)
    print(f'action_idx: {idx.tolist()}  (safety={MAX_SHOTS})')
    print(f'force_raw: {[round(x, 3) for x in f.tolist()]}')
    print(f'spin_raw: {[round(x, 3) for x in s.tolist()]}')
    print(f'log_prob: {[round(x, 3) for x in lp.tolist()]}')
    print(f'value (win prob): {[round(x, 3) for x in v.tolist()]}')
    lp2, ent, v2 = net.evaluate_actions(obs, idx, f, s)
    print(f'eval log_prob close: {torch.allclose(lp, lp2, atol=1e-5)}')
    print(f'entropy: {[round(x, 3) for x in ent.tolist()]}')

    # Test placement head
    obs_plc = {k: v.clone() for k, v in obs.items()}
    obs_plc['game_state'][:, 5] = 1.0  # BIH flag
    obs_plc['shot_mask'][:] = False
    idx_p, xn, yn, lp_p, v_p = net.get_action(obs_plc)
    print(f'\nPlacement test (BIH):')
    print(f'action_idx: {idx_p.tolist()} (should be -1)')
    print(f'x_norm: {[round(x, 3) for x in xn.tolist()]}')
    print(f'y_norm: {[round(x, 3) for x in yn.tolist()]}')
    print(f'log_prob: {[round(x, 3) for x in lp_p.tolist()]}')
