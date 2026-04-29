"""
Transformer-based Actor-Critic policy for pool RL.

The table layout is treated as a SET of entities:
- 16 ball tokens (cue ball + 15 object balls)
- 6 pocket tokens

Each token has features (position, type, status). Self-attention
learns ball-ball relationships (blocking, combos, clusters) and
ball-pocket relationships (which shots are viable).

This is analogous to how Set Transformer or DETR handle sets of
objects -- much more natural for pool than a flat MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


class BallEncoder(nn.Module):
    """Encode raw ball features into token embeddings."""

    def __init__(self, embed_dim=128):
        super().__init__()
        # Ball features: x, y (2) + on_table (1) + ball_type (1) = 4
        self.ball_embed = nn.Linear(4, embed_dim)
        # Pocket features: x, y (2) = 2
        self.pocket_embed = nn.Linear(2, embed_dim)
        # Learnable type embeddings: cue ball, solid, stripe, 8-ball, pocket
        self.type_embed = nn.Embedding(5, embed_dim)

    def forward(self, ball_features, pocket_features):
        """
        ball_features: (batch, 16, 4) - [x, y, on_table, ball_type_id]
        pocket_features: (batch, 6, 2) - [x, y]
        Returns: (batch, 22, embed_dim) - all entity tokens
        """
        ball_tokens = self.ball_embed(ball_features)
        pocket_tokens = self.pocket_embed(pocket_features)

        # Add type embeddings
        ball_type_ids = ball_features[:, :, 3].long()  # 0=cue, 1=solid, 2=stripe, 3=eight
        ball_tokens = ball_tokens + self.type_embed(ball_type_ids)
        pocket_type = torch.full((pocket_features.shape[0], 6), 4, dtype=torch.long,
                                  device=pocket_features.device)
        pocket_tokens = pocket_tokens + self.type_embed(pocket_type)

        # Concatenate all tokens
        tokens = torch.cat([ball_tokens, pocket_tokens], dim=1)  # (batch, 22, embed_dim)
        return tokens


class PoolTransformer(nn.Module):
    """Transformer encoder for pool table state."""

    def __init__(self, embed_dim=128, num_heads=4, num_layers=3, ff_dim=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens, mask=None):
        """
        tokens: (batch, num_tokens, embed_dim)
        mask: (batch, num_tokens) - True for tokens to IGNORE (pocketed balls)
        Returns: (batch, num_tokens, embed_dim)
        """
        return self.encoder(tokens, src_key_padding_mask=mask)


class TransformerActorCritic(nn.Module):
    """
    Transformer-based Actor-Critic for pool.

    Architecture:
    1. BallEncoder: embed each ball and pocket as a token
    2. PoolTransformer: self-attention learns entity relationships
    3. Global pooling + game state -> shared representation
    4. Actor head: outputs shot parameters (aim, force, english, elevation)
    5. Critic head: outputs state value estimate
    """

    def __init__(self, embed_dim=128, num_heads=4, num_layers=3, act_dim=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.act_dim = act_dim

        # Entity encoder
        self.encoder = BallEncoder(embed_dim)

        # Transformer
        self.transformer = PoolTransformer(embed_dim, num_heads, num_layers)

        # Game state encoder (scores, fouls, etc.)
        self.game_state_encoder = nn.Linear(6, embed_dim)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, act_dim),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Learnable log std for action distribution
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        # Running observation normalization (for the raw obs)
        self.obs_count = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

    def _parse_observation(self, obs):
        """
        Parse flat observation (38,) into structured inputs.
        obs layout: [cue_x, cue_y, ball1_x, ball1_y, ..., ball15_x, ball15_y,
                     my_score, opp_score, balls_remaining, consec_fouls, rerack_flag, current_player]
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Ball features: (batch, 16, 4) = [x, y, on_table, type_id]
        ball_features = torch.zeros(batch_size, 16, 4, device=device)

        # Cue ball (id=0)
        ball_features[:, 0, 0] = obs[:, 0]  # x (normalized)
        ball_features[:, 0, 1] = obs[:, 1]  # y (normalized)
        ball_features[:, 0, 2] = (obs[:, 0] >= 0).float()  # on_table (x >= 0 means not pocketed)
        ball_features[:, 0, 3] = 0  # type: cue ball

        # Object balls (id 1-15)
        for b in range(1, 16):
            idx = 2 + (b - 1) * 2
            ball_features[:, b, 0] = obs[:, idx]      # x
            ball_features[:, b, 1] = obs[:, idx + 1]  # y
            ball_features[:, b, 2] = (obs[:, idx] >= 0).float()  # on_table
            # Type: 1=solid (1-7), 2=stripe (9-15), 3=eight (8)
            if b <= 7:
                ball_features[:, b, 3] = 1
            elif b == 8:
                ball_features[:, b, 3] = 3
            else:
                ball_features[:, b, 3] = 2

        # Pocket features: (batch, 6, 2) - fixed positions, normalized
        pocket_features = torch.zeros(batch_size, 6, 2, device=device)
        pocket_positions = [
            (0, 0), (0.5, 0), (1, 0),
            (0, 1), (0.5, 1), (1, 1),
        ]
        for p, (px, py) in enumerate(pocket_positions):
            pocket_features[:, p, 0] = px
            pocket_features[:, p, 1] = py

        # Attention mask: mask out pocketed balls (True = ignore)
        mask = torch.zeros(batch_size, 22, dtype=torch.bool, device=device)
        for b in range(16):
            if b == 0:
                mask[:, b] = (obs[:, 0] < 0)  # cue ball pocketed
            else:
                idx = 2 + (b - 1) * 2
                mask[:, b] = (obs[:, idx] < 0)  # object ball pocketed
        # Pockets are never masked (indices 16-21)

        # Game state: [my_score, opp_score, balls_remaining, consec_fouls, rerack, player]
        game_state = obs[:, 32:38]

        return ball_features, pocket_features, mask, game_state

    def forward(self, obs):
        """Forward pass: obs -> (action_mean, value)"""
        ball_feat, pocket_feat, mask, game_state = self._parse_observation(obs)

        # Encode entities into tokens
        tokens = self.encoder(ball_feat, pocket_feat)

        # Transformer self-attention
        encoded = self.transformer(tokens, mask=mask)

        # Global pooling: mean of non-masked tokens
        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()
        masked_encoded = encoded * (1 - mask_expanded)
        num_active = (1 - mask_expanded).sum(dim=1).clamp(min=1)
        global_repr = masked_encoded.sum(dim=1) / num_active

        # Combine with game state
        game_repr = self.game_state_encoder(game_state)
        combined = torch.cat([global_repr, game_repr], dim=-1)

        # Actor and critic heads
        action_mean = self.actor(combined)
        value = self.critic(combined)

        return action_mean, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs, actions):
        """Evaluate log probability and entropy of given actions."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """Storage for rollout data."""

    def __init__(self, num_envs, num_steps, obs_dim, act_dim):
        self.obs = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.step = 0
        self.num_steps = num_steps
        self.num_envs = num_envs

    def add(self, obs, actions, rewards, dones, log_probs, values):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        self.step += 1

    def compute_returns(self, last_values, gamma=0.995, lam=0.95):
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        total = self.num_steps * self.num_envs
        indices = np.random.permutation(total)
        obs_flat = self.obs.reshape(total, -1)
        actions_flat = self.actions.reshape(total, -1)
        log_probs_flat = self.log_probs.reshape(total)
        returns_flat = self.returns.reshape(total)
        advantages_flat = self.advantages.reshape(total)
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std() + 1e-8
        advantages_flat = (advantages_flat - adv_mean) / adv_std

        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield (
                torch.FloatTensor(obs_flat[idx]),
                torch.FloatTensor(actions_flat[idx]),
                torch.FloatTensor(log_probs_flat[idx]),
                torch.FloatTensor(returns_flat[idx]),
                torch.FloatTensor(advantages_flat[idx]),
            )

    def reset(self):
        self.step = 0
