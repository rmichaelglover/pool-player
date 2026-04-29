"""
Attention-based Actor-Critic for 14.1 Continuous pool.

Each ball is a "token" (like words in an LLM). Self-attention layers learn
ball-ball relationships: obstruction, clustering, pocket alignment, key ball
positioning. Everything from raw positions -- no hand-coded heuristics.

Architecture:
  16 ball tokens (7 features each) + 6 pocket tokens (4 features each)
  -> entity encoders -> type embeddings -> 4 transformer layers
  -> mean pool + cue token + game state -> actor (5 continuous) + critic (1)

Input: 38-dim flat observation (cue pos + 15 ball pos + 6 game state)
Output: 5 continuous actions (aim_sin, aim_cos, force, contact_x, contact_y)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

TABLE_LENGTH = 100.0
TABLE_WIDTH = 50.0

# Fixed pocket positions and properties (normalized)
_mo = 2.5 * 0.45
_smo = 2.75 * 0.15
POCKET_FEATURES = torch.tensor([
    [_mo / TABLE_LENGTH,          _mo / TABLE_WIDTH,          1.0, 2.5 / 2.75],   # TL corner
    [0.5,                         _smo / TABLE_WIDTH,         0.0, 1.0],            # TS side
    [(TABLE_LENGTH - _mo) / TABLE_LENGTH, _mo / TABLE_WIDTH,  1.0, 2.5 / 2.75],   # TR corner
    [_mo / TABLE_LENGTH,          (TABLE_WIDTH - _mo) / TABLE_WIDTH, 1.0, 2.5 / 2.75],  # BL corner
    [0.5,                         (TABLE_WIDTH - _smo) / TABLE_WIDTH, 0.0, 1.0],   # BS side
    [(TABLE_LENGTH - _mo) / TABLE_LENGTH, (TABLE_WIDTH - _mo) / TABLE_WIDTH, 1.0, 2.5 / 2.75],  # BR corner
], dtype=torch.float32)  # (6, 4)


class PoolAttentionNet(nn.Module):
    """
    Transformer-based pool AI.

    Every ball attends to every other ball and every pocket.
    The network learns shot selection, position play, safety play,
    and break ball management -- all from self-play.

    Like an LLM learns language from tokens, this learns pool from ball positions.
    """

    def __init__(self, embed_dim=96, num_heads=6, num_layers=4,
                 ff_dim=192, act_dim=5, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.act_dim = act_dim

        # --- Entity encoders ---
        # Ball: 7 features -> embed_dim
        #   (x, y, on_table, is_cue, sin(2pi*x), cos(2pi*x), sin(2pi*y))
        self.ball_encoder = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        # Pocket: 4 features -> embed_dim
        #   (x, y, is_corner, radius_norm)
        self.pocket_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        # Learnable type embedding: 0=cue, 1=object ball, 2=pocket
        self.type_embed = nn.Embedding(3, embed_dim)

        # --- Transformer encoder (pre-norm for stability) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- Game state encoder ---
        self.game_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
        )

        # Trunk output: mean_pool(embed_dim) + cue_token(embed_dim) + game(64)
        trunk_dim = embed_dim * 2 + 64

        # --- Actor head (continuous actions) ---
        self.actor = nn.Sequential(
            nn.Linear(trunk_dim, 192),
            nn.GELU(),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, act_dim),
        )

        # --- Critic head (state value) ---
        self.critic = nn.Sequential(
            nn.Linear(trunk_dim, 192),
            nn.GELU(),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Learnable log std for action distribution
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Register pocket features as buffer (auto-moves to device)
        self.register_buffer('pocket_features', POCKET_FEATURES)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for actor output (start near zero actions)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        # Standard init for critic output
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def _parse_obs(self, obs):
        """
        Parse flat observation (batch, 38) into structured entity features.

        Observation layout:
          [0:2]    cue ball (x/TL, y/TW), or -1 if pocketed
          [2:32]   object balls 1-15, each (x/TL, y/TW), or -1 if pocketed
          [32:38]  game state: my_score/target, opp_score/target,
                   balls_remaining/15, consec_fouls/3, rerack_flag, is_break

        Returns:
          ball_features: (batch, 16, 7)
          mask: (batch, 22) bool -- True = ignore (pocketed ball)
          game_state: (batch, 6)
        """
        B = obs.shape[0]
        device = obs.device

        ball_features = torch.zeros(B, 16, 7, device=device)
        mask = torch.zeros(B, 22, dtype=torch.bool, device=device)

        for b in range(16):
            idx = b * 2
            bx = obs[:, idx]
            by = obs[:, idx + 1]
            on = (bx >= 0).float()
            bx_safe = bx.clamp(min=0)
            by_safe = by.clamp(min=0)

            ball_features[:, b, 0] = bx_safe                          # x
            ball_features[:, b, 1] = by_safe                          # y
            ball_features[:, b, 2] = on                                # on_table
            ball_features[:, b, 3] = 1.0 if b == 0 else 0.0           # is_cue
            ball_features[:, b, 4] = torch.sin(2 * np.pi * bx_safe)   # fourier x
            ball_features[:, b, 5] = torch.cos(2 * np.pi * bx_safe)   # fourier x
            ball_features[:, b, 6] = torch.sin(2 * np.pi * by_safe)   # fourier y
            mask[:, b] = (bx < 0)  # mask pocketed balls

        # Pockets (indices 16-21) are never masked
        game_state = obs[:, 32:38]

        return ball_features, mask, game_state

    def forward(self, obs):
        """
        Forward pass.

        Args:
            obs: (batch, 38) flat observation

        Returns:
            action_mean: (batch, 5) -- [aim_sin, aim_cos, force, contact_x, contact_y]
            value: (batch,) -- state value estimate
        """
        ball_feat, mask, game_state = self._parse_obs(obs)
        B = obs.shape[0]
        device = obs.device

        # Encode balls -> (B, 16, embed_dim)
        ball_tokens = self.ball_encoder(ball_feat)

        # Encode pockets -> (B, 6, embed_dim)
        pocket_feat = self.pocket_features.unsqueeze(0).expand(B, -1, -1)
        pocket_tokens = self.pocket_encoder(pocket_feat)

        # Type embeddings
        cue_type = torch.zeros(B, 1, dtype=torch.long, device=device)
        obj_type = torch.ones(B, 15, dtype=torch.long, device=device)
        pocket_type = torch.full((B, 6), 2, dtype=torch.long, device=device)

        ball_type_ids = torch.cat([cue_type, obj_type], dim=1)
        ball_tokens = ball_tokens + self.type_embed(ball_type_ids)
        pocket_tokens = pocket_tokens + self.type_embed(pocket_type)

        # All tokens: (B, 22, embed_dim)
        tokens = torch.cat([ball_tokens, pocket_tokens], dim=1)

        # Transformer with attention masking for pocketed balls
        encoded = self.transformer(tokens, src_key_padding_mask=mask)

        # --- Aggregate ---
        # 1. Masked mean pool (exclude pocketed balls)
        active = (~mask).unsqueeze(-1).float()  # (B, 22, 1)
        pooled = (encoded * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)

        # 2. Cue ball token (index 0) -- always extracted, even if scratched
        cue_token = encoded[:, 0, :]

        # 3. Game state encoding
        game_repr = self.game_encoder(game_state)

        # Shared trunk
        trunk = torch.cat([pooled, cue_token, game_repr], dim=-1)

        # Output heads
        action_mean = self.actor(trunk)
        value = self.critic(trunk).squeeze(-1)

        return action_mean, value

    def get_action(self, obs, deterministic=False):
        """Sample action for environment interaction (inference)."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """Evaluate log_prob and entropy for PPO update."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    @staticmethod
    def action_to_physics(action_np):
        """
        Convert network output to physics parameters.

        Input: (batch, 5) raw network output
        Output: dict with aim_angle, force, spin_x, spin_y
        """
        aim_angle = np.arctan2(action_np[:, 0], action_np[:, 1])
        force = 1.0 / (1.0 + np.exp(-action_np[:, 2]))  # sigmoid -> [0, 1]
        force = force * 70 + 10  # map to [10, 80] in/s
        contact_x = np.tanh(action_np[:, 3])  # english: [-1, 1]
        contact_y = np.tanh(action_np[:, 4])  # draw/follow: [-1, 1]
        return {
            'aim_angle': aim_angle,
            'force': force,
            'contact_x': contact_x,
            'contact_y': contact_y,
        }


# ─── Quick test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    net = PoolAttentionNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Parameters: {n_params:,}')

    # Test forward pass
    obs = torch.randn(4, 38)  # batch of 4
    action_mean, value = net(obs)
    print(f'Action mean shape: {action_mean.shape}')  # (4, 5)
    print(f'Value shape: {value.shape}')                # (4,)

    # Test action sampling
    action, log_prob, value = net.get_action(obs)
    print(f'Action shape: {action.shape}')              # (4, 5)
    print(f'Log prob shape: {log_prob.shape}')           # (4,)

    # Test evaluate
    lp, ent, val = net.evaluate_actions(obs, action)
    print(f'Entropy shape: {ent.shape}')                 # (4,)

    # Test action conversion
    physics = PoolAttentionNet.action_to_physics(action.detach().numpy())
    print(f'Aim angles: {physics["aim_angle"]}')
    print(f'Forces: {physics["force"]}')
    print(f'Contact Y (draw/follow): {physics["contact_y"]}')
