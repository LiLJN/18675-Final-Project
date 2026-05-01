# rl.py
# actor and critic for td3
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random



class Actor(nn.Module):
    """Maps EKF output [estimate + uncertainty] → action (torque)."""

    def __init__(self, input_dim=4, hidden_dim=256, max_action=2.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Twin Q-networks for TD3."""

    def __init__(self, input_dim=4, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# Replay Buffer
class ReplayBuffer:
    """
    Stores complete episodes and samples fixed-length subsequences.
    """

    def __init__(self, capacity=1000, device=None):
        self.episodes = deque(maxlen=capacity)  # completed episodes
        self.current_episode = [] # episode in progress
        self.device = device

    def push(self, obs, action, reward, done, true_state=None):
        """Store a single transition. Automatically segments into episodes."""
        self.current_episode.append((
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.float32).flatten(),
            float(reward),
            float(done),
            np.array(true_state, dtype=np.float32) if true_state is not None else np.zeros(2, dtype=np.float32)
        ))
        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def sample(self, batch_size, seq_len=16):
        """
        Sample batch_size subsequences of length seq_len.

        Returns:
            obs_seq:  (B, T, obs_dim)   raw masked observations
            act_seq:  (B, T, act_dim)   actions taken
            rew_seq:  (B, T)            rewards
            done_seq: (B, T)            done flags
        """
        batch_obs, batch_act, batch_rew, batch_done, batch_true = [], [], [], [], []

        for _ in range(batch_size):
            ep = random.choice(self.episodes)

            # Random start index (ensure at least seq_len steps)
            max_start = max(0, len(ep) - seq_len)
            start = random.randint(0, max_start)
            subseq = ep[start : start + seq_len]

            # Pad short subsequences by repeating last transition pineapple
            # while len(subseq) < seq_len:
            #     last = subseq[-1]
            #     subseq.append((last[0], last[1], 0.0, 1.0))

            obs_s, act_s, rew_s, done_s, true_s = zip(*subseq)
            batch_obs.append(torch.tensor(np.stack(obs_s)))
            batch_act.append(torch.tensor(np.stack(act_s)))
            batch_rew.append(torch.tensor(rew_s))
            batch_done.append(torch.tensor(done_s))
            batch_true.append(torch.tensor(np.stack(true_s)))

        return (
            torch.stack(batch_obs).to(self.device),   # (B, T, obs_dim)
            torch.stack(batch_act).to(self.device),   # (B, T, act_dim)
            torch.stack(batch_rew).to(self.device),   # (B, T)
            torch.stack(batch_done).to(self.device),  # (B, T)
            torch.stack(batch_true).to(self.device),   # (B, T, 2)  [theta, theta_dot]
        )

    @property
    def num_episodes(self):
        return len(self.episodes)

    def ready(self, min_episodes=5):
        return len(self.episodes) >= min_episodes


class FlatReplayBuffer:
    """Standard flat (s, a, r, s', done) buffer for vanilla TD3."""

    def __init__(self, capacity=500_000, device=None):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.float32).flatten(),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.stack(obs)).to(self.device),
            torch.tensor(np.stack(act)).to(self.device),
            torch.tensor(np.stack(rew)).unsqueeze(-1).float().to(self.device),
            torch.tensor(np.stack(next_obs)).to(self.device),
            torch.tensor(np.stack(done)).unsqueeze(-1).float().to(self.device),
        )

    def ready(self, min_samples=1000):
        return len(self.buffer) >= min_samples


class FrameStack:
    def __init__(self, n_frames=3):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, obs):
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, obs):
        self.frames.append(obs)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(list(self.frames))
