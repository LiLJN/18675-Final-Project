from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvironmentConfig:
    noise_std = 0.0
    max_steps = 200

@dataclass
class TrainConfig:
    num_episodes = 500
    warmup_episodes = 10
    batch_size = 256
    explore_noise = 0.1
    seq_len = 16

@dataclass
class TD3Config:
    lr_actor = 3e-4
    lr_critic = 3e-4
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_delay = 2
    hidden_dim = 256

@dataclass
class EKFConfig:
    lr_ekf = 3e-4

@dataclass
class EvalConfig:
    eval_every = 25
    num_eval_episodes = 10

@dataclass
class LogConfig:
    save_dir: str


@dataclass
class AgentConfig:
    """
    Top level config for training.
    """
    environment_config: EnvironmentConfig
    train_config: TrainConfig
    policy_config: TD3Config
    ekf_config: EKFConfig
