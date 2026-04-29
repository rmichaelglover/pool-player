"""Training configuration."""

class Config:
    # Environment (geometric = instant, so we can use many envs and steps)
    num_envs = 128
    target_score = 10
    curriculum_balls = 15   # full rack -- more balls = more chances to hit
    use_geometric = True    # Phase 1: geometric only. Set False for physics.

    # PPO
    num_steps_per_env = 64  # more steps per update since env is fast
    num_epochs = 5          # PPO epochs per update
    batch_size = 512        # minibatch size (larger with more data)
    learning_rate = 3e-4
    gamma = 0.995           # discount factor (long horizon)
    gae_lambda = 0.95
    clip_param = 0.2
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 1.0

    # Network (Transformer)
    obs_dim = 38
    act_dim = 5
    embed_dim = 128       # token embedding dimension
    num_heads = 4         # attention heads
    num_layers = 3        # transformer encoder layers
    ff_dim = 256          # feedforward dim inside transformer

    # Training
    max_iterations = 50000
    log_interval = 10
    save_interval = 200
    curriculum_schedule = {
        # Start with full rack -- more balls = more chances to hit/pocket
        # Increase target score as the agent improves
        0:     {'curriculum_balls': 15, 'target_score': 5},
        3000:  {'curriculum_balls': 15, 'target_score': 10},
        8000:  {'curriculum_balls': 15, 'target_score': 20},
        15000: {'curriculum_balls': 15, 'target_score': 35},
        25000: {'curriculum_balls': 15, 'target_score': 50},
    }
