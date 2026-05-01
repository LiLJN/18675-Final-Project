from src.rl import Actor, Critic, FlatReplayBuffer, FrameStack

import torch
import torch.nn.functional as F
import copy
import numpy as np


class TD3_Agent:
    """
    Standard TD3 operating on raw observations (no EKF).
    Serves as a lower bound baseline — same actor/critic architecture,
    but receives [cos θ, sin θ, θ̇] directly instead of filtered state.
    """

    def __init__(
        self,
        obs_dim=3,
        hidden_dim=256,
        max_action=2.0,
        n_frames=3,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        lr_actor=3e-4,
        lr_critic=3e-4,
        buffer_capacity=500_000,
        device=None,
    ):
        self.device = device

        self.n_frames = n_frames
        self.frame_stack = FrameStack(n_frames)
        stacked_dim = obs_dim * n_frames

        self.actor = Actor(stacked_dim, hidden_dim, max_action).to(self.device)
        self.critic = Critic(stacked_dim, hidden_dim).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        self.replay_buffer = FlatReplayBuffer(
            capacity=buffer_capacity, device=self.device
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_updates = 0

    # DATA COLLECTION
    # def select_action(self, obs, explore_noise=0.1):
    #     with torch.no_grad():
    #         state = torch.tensor(obs, dtype=torch.float32, device=self.device)
    #         action = self.actor(state).cpu().numpy().flatten()
    #     if explore_noise > 0:
    #         action += np.random.normal(0, explore_noise, size=action.shape)
    #         action = np.clip(action, -self.max_action, self.max_action)
    #     return action

    def select_action(self, stacked_obs, explore_noise=0.1):
        with torch.no_grad():
            state = torch.tensor(stacked_obs, dtype=torch.float32, device=self.device)
            action = self.actor(state).cpu().numpy().flatten()
        if explore_noise > 0:
            action += np.random.normal(0, explore_noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    # ── Training ─────────────────────────────────────────────────
    def train_step(self, batch_size=256):
        if not self.replay_buffer.ready(batch_size):
            return {}

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            tq1, tq2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.discount * torch.min(tq1, tq2)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        info = {"critic_loss": critic_loss.item()}

        # Delayed actor update
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic.Q1(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.actor.parameters(),
                             self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            info["actor_loss"] = actor_loss.item()

        return info

    def reset(self, obs):
        "Reset FrameStack at beginning of each episode."
        return self.frame_stack.reset(obs)
