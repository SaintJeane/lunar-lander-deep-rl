# config.py
"""
Configuration management for LunarLander project.
Centralizes all hyperparameters and settings.
"""

from dataclasses import dataclass
import os
import torch
import numpy as np


@dataclass
class DQNConfig:
    """DQN Agent hyperparameters."""

    # Network architecture
    hidden_dim: int = 128

    # Training hyperparameters
    learning_rate: float = 1e-4  # try 5e-4
    gamma: float = 0.99
    batch_size: int = 64

    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Memory
    buffer_capacity: int = 100000  # 100_000

    # Target network
    target_update_freq: int = 10

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Other Features
    use_double_dqn: bool = True
    use_soft_update: bool = True
    tau: float = 0.01  # try 0.05

    # Device
    device: str = (
        "cuda" if torch.cuda.is_available() and os.environ.get("USE_GPU", "1") != "0" else "cpu"
    )


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    env_name: str = "LunarLander-v2"
    seed: int = 42
    num_episodes: int = 2000
    max_steps: int = 1000
    log_interval: int = 10
    save_interval: int = 500

    # Directories
    save_dir: str = "./models"
    plot_dir: str = "./plots"
    log_dir: str = "./logs"
    video_dir: str = "./videos"

    # Early stopping
    early_stopping: bool = False
    patience: int = 200
    target_reward: float = 250.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    num_episodes: int = 100
    render: bool = False
    record_video: bool = False
    model_path: str = "./models/best_model.pth"


def get_config(config_type="training"):
    """
    Factory function to get configuration objects.

    Args:
        config_type: One of 'training', 'evaluation', 'dqn'

    Returns:
        Configuration dataclass instance
    """
    configs = {
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(),
        "dqn": DQNConfig(),
    }

    if config_type not in configs:
        raise ValueError(f"Invalid config type: {config_type}")
    return configs.get(config_type, TrainingConfig())


# Environment variable overrides
def load_config_from_env():
    """Load configuration from environment variables."""
    dqn_config = DQNConfig()
    train_config = TrainingConfig()

    # DQN config overrides
    if "LEARNING_RATE" in os.environ:
        dqn_config.learning_rate = float(os.environ["LEARNING_RATE"])
    if "GAMMA" in os.environ:
        dqn_config.gamma = float(os.environ["GAMMA"])
    if "BATCH_SIZE" in os.environ:
        dqn_config.batch_size = int(os.environ["BATCH_SIZE"])

    # Training config overrides
    if "NUM_EPISODES" in os.environ:
        train_config.num_episodes = int(os.environ["NUM_EPISODES"])
    if "MAX_STEPS" in os.environ:
        train_config.max_steps = int(os.environ["MAX_STEPS"])

    return dqn_config, train_config
