from src.pendulum_ekf import DifferentiableEKF
from src.rl import Actor, Critic, ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class Separate_TD3_EKF_Agent:
    """
    TD3 with separately optimized EKF front-end.

    EKF noise parameters Q and R are optimized via state estimation MSE loss
    (train_ekf_step), independent of the TD3 policy update (train_step).
    Gradients from the actor never flow back through the EKF — the filter
    is frozen during all TD3 updates.
    """

    def __init__(
        self,
        ekf_input_dim=4,
        hidden_dim=256,
        max_action=2.0,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        seq_len=16,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_ekf=3e-4,
        device=None
    ):
        self.device = device
        self.ekf = DifferentiableEKF().to(self.device)
        self.actor = Actor(ekf_input_dim, hidden_dim, max_action).to(self.device)
        self.critic = Critic(ekf_input_dim, hidden_dim).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # 3 optimizers
        self.ekf_optimizer = torch.optim.Adam(
            self.ekf.parameters(), lr=lr_ekf
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor  # no ekf.parameters() here
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        self.replay_buffer = ReplayBuffer(device=self.device)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.seq_len = seq_len
        self.total_updates = 0

        # Running EKF state for data collection
        self.x_est = None
        self.P_est = None

    # Data collection (single-env, no gradients)
    def reset_ekf(self, obs):
        z = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.x_est, self.P_est = self.ekf.init_state(z)

    def select_action(self, obs, explore_noise=0.1):
        with torch.no_grad():
            policy_input = self.ekf.get_policy_input(self.x_est, self.P_est)
            action = self.actor(policy_input).cpu().numpy().flatten()
        if explore_noise > 0:
            action += np.random.normal(0, explore_noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def ekf_step(self, obs, action):
        with torch.no_grad():
            z = torch.tensor(obs, dtype=torch.float32, device=self.device)
            u = torch.tensor(action, dtype=torch.float32, device=self.device).squeeze()
            self.x_est, self.P_est = self.ekf(z, u, self.x_est, self.P_est)

    def store_transition(self, obs, action, reward, done, true_state):
        """Store raw observation — NOT detached EKF state."""
        self.replay_buffer.push(obs, action, reward, done, true_state)

    # EKF unrolling over sequences
    def _unroll_ekf(self, obs_seq, act_seq, with_grad=False):
        """
        Re-run the EKF over batched sequences.

        Args:
            obs_seq:   (B, T, obs_dim)  raw observations
            act_seq:   (B, T, act_dim)  actions
            with_grad: if True, keeps computation graph for backprop to Q, R

        Returns:
            ekf_states: (B, T, 4)  policy inputs at each timestep
        """
        B, T, _ = obs_seq.shape
        ctx = torch.enable_grad if with_grad else torch.no_grad

        with ctx():
            x, P = self.ekf.init_state_batched(obs_seq[:, 0])
            all_states = [self.ekf.get_policy_input_batched(x, P)]

            for t in range(1, T):
                # Predict with action from t-1, update with observation at t
                u = act_seq[:, t - 1].squeeze(-1)           # (B,)
                x, P = self.ekf.forward_batched(obs_seq[:, t], u, x, P)
                all_states.append(self.ekf.get_policy_input_batched(x, P))

            return torch.stack(all_states, dim=1)            # (B, T, 4)

    # Training
    def train_ekf_step(self, batch_size=32):
        if not self.replay_buffer.ready(5): # changed from batch_size to 5 (episodes)
            return {}

        obs_seq, act_seq, _, _, true_seq = self.replay_buffer.sample(
            batch_size, self.seq_len
        )
        # true_seq: (B, T, 2) — [theta, theta_dot]

        ekf_states = self._unroll_ekf(obs_seq, act_seq, with_grad=True)
        x_est = ekf_states[:, :, :2]                        # (B, T, 2)

        estimation_loss = F.mse_loss(x_est, true_seq)

        self.ekf_optimizer.zero_grad()
        estimation_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ekf.parameters(), max_norm=1.0)
        self.ekf_optimizer.step()

        return {
            "estimation_loss": estimation_loss.item(),
            "Q_learned": self.ekf.Q.detach().cpu().numpy().tolist(),
            "R_learned": self.ekf.R.detach().cpu().numpy().tolist(),
        }

    def train_step(self, batch_size=32):
        if not self.replay_buffer.ready(batch_size):
            return {}

        obs_seq, act_seq, rew_seq, done_seq, _ = self.replay_buffer.sample(
            batch_size, self.seq_len
        )

        # EKF always frozen during TD3 updates — no with_grad=True here
        ekf_states = self._unroll_ekf(obs_seq, act_seq, with_grad=False)

        states = ekf_states[:, :-1].reshape(-1, 4)
        next_states = ekf_states[:, 1:].reshape(-1, 4)
        actions = act_seq[:, :-1].reshape(-1, 1)
        rewards = rew_seq[:, :-1].reshape(-1, 1)
        dones = done_seq[:, :-1].reshape(-1, 1)

        #  Critic update — IDENTICAL
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

        # ── Actor update — no EKF gradient, actor only ───────────
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:

            actor_actions = self.actor(states)
            actor_loss = -self.critic.Q1(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            info["actor_loss"] = actor_loss.item()

        return info
