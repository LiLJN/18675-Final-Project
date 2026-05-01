# ekf.py
# Differentiable Extended Kalman Filter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableEKF(nn.Module):
    """
    Extended Kalman Filter with learnable Q and R.

    Two sets of methods:
      - Single-env methods (dynamics, predict, update, forward)
        → used during data collection with torch.no_grad()
      - Batched methods (*_batched)
        → used during training to re-run EKF over sequences with gradients
    """

    def __init__(self):
        super().__init__()

        # Pendulum constants
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.05
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.state_dim = 2
        self.obs_dim = 2

        # Learnable noise covariances (log-Cholesky parameterization)
        self.q_log_diag = nn.Parameter(torch.tensor([-2.0, -2.0]))
        self.q_off_diag = nn.Parameter(torch.tensor([0.0]))
        self.r_log_diag = nn.Parameter(torch.tensor([-1.0, -1.0]))
        self.r_off_diag = nn.Parameter(torch.tensor([0.0]))

    # Covariance construction
    def _build_covariance(self, log_diag, off_diag):
        L = torch.zeros(2, 2, device=log_diag.device)
        L[0, 0] = torch.exp(log_diag[0])
        L[1, 0] = off_diag[0]
        L[1, 1] = torch.exp(log_diag[1])
        return L @ L.t() + 1e-6 * torch.eye(2, device=log_diag.device)

    @property
    def Q(self):
        return self._build_covariance(self.q_log_diag, self.q_off_diag)

    @property
    def R(self):
        return self._build_covariance(self.r_log_diag, self.r_off_diag)

    # Single-env methods (for data collection)
    def dynamics(self, x, u):
        theta, theta_dot = x[0], x[1]
        u_clipped = torch.clamp(u, -self.max_torque, self.max_torque)
        new_theta_dot = theta_dot + (
            3.0 * self.g / (2.0 * self.l) * torch.sin(theta)
            + 3.0 / (self.m * self.l ** 2) * u_clipped
        ) * self.dt
        new_theta_dot = torch.clamp(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt
        new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))
        return torch.stack([new_theta, new_theta_dot])

    def dynamics_jacobian(self, x, u):
        cos_th = torch.cos(x[0])
        d = 3.0 * self.g / (2.0 * self.l) * cos_th * self.dt
        F = torch.stack([
            torch.stack([1.0 + d * self.dt, torch.tensor(self.dt, device=x.device)]),
            torch.stack([d, torch.tensor(1.0, device=x.device)])
        ])
        return F

    def observation_model(self, x):
        return torch.stack([torch.cos(x[0]), torch.sin(x[0])])

    def observation_jacobian(self, x):
        H = torch.zeros(2, 2, device=x.device)
        H[0, 0] = -torch.sin(x[0])
        H[1, 0] = torch.cos(x[0])
        return H

    def init_state(self, z0):
        theta = torch.atan2(z0[1], z0[0])
        x0 = torch.stack([theta, torch.tensor(0.0, device=z0.device)])
        P0 = torch.eye(2, device=z0.device) * 1.0
        return x0, P0

    def predict(self, x, P, u):
        x_pred = self.dynamics(x, u)
        F = self.dynamics_jacobian(x, u)
        P_pred = F @ P @ F.t() + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        H = self.observation_jacobian(x_pred)
        y = z - self.observation_model(x_pred)
        S = H @ P_pred @ H.t() + self.R
        K = torch.linalg.solve(S, H @ P_pred).t()
        x_upd = x_pred + K @ y
        I = torch.eye(2, device=x_pred.device)
        IKH = I - K @ H
        P_upd = IKH @ P_pred @ IKH.t() + K @ self.R @ K.t()
        return x_upd, P_upd

    def forward(self, z, u, x_prev, P_prev):
        x_pred, P_pred = self.predict(x_prev, P_prev, u)
        return self.update(x_pred, P_pred, z)

    def get_policy_input(self, x_est, P_est):
        return torch.cat([x_est, torch.diagonal(P_est)])

    # ── Batched methods (for training with gradients) ────────────

    def dynamics_batched(self, x, u):
        """x: (B, 2), u: (B,) → (B, 2)"""
        theta = x[:, 0]
        theta_dot = x[:, 1]
        u_clipped = torch.clamp(u, -self.max_torque, self.max_torque)
        new_theta_dot = theta_dot + (
            3.0 * self.g / (2.0 * self.l) * torch.sin(theta)
            + 3.0 / (self.m * self.l ** 2) * u_clipped
        ) * self.dt
        new_theta_dot = torch.clamp(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt
        new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))
        return torch.stack([new_theta, new_theta_dot], dim=-1)

    def dynamics_jacobian_batched(self, x, u):
        """x: (B, 2) → F: (B, 2, 2)"""
        B = x.shape[0]
        cos_th = torch.cos(x[:, 0])
        d = 3.0 * self.g / (2.0 * self.l) * cos_th * self.dt
        F = torch.zeros(B, 2, 2, device=x.device)
        F[:, 0, 0] = 1.0 + d * self.dt
        F[:, 0, 1] = self.dt
        F[:, 1, 0] = d
        F[:, 1, 1] = 1.0
        return F

    def observation_model_batched(self, x):
        """x: (B, 2) → z: (B, 2)"""
        return torch.stack([torch.cos(x[:, 0]), torch.sin(x[:, 0])], dim=-1)

    def observation_jacobian_batched(self, x):
        """x: (B, 2) → H: (B, 2, 2)"""
        B = x.shape[0]
        H = torch.zeros(B, 2, 2, device=x.device)
        H[:, 0, 0] = -torch.sin(x[:, 0])
        H[:, 1, 0] = torch.cos(x[:, 0])
        return H

    def init_state_batched(self, z0):
        """z0: (B, 2) → x0: (B, 2), P0: (B, 2, 2)"""
        B = z0.shape[0]
        theta = torch.atan2(z0[:, 1], z0[:, 0])
        x0 = torch.stack([theta, torch.zeros(B, device=z0.device)], dim=-1)
        P0 = torch.eye(2, device=z0.device).unsqueeze(0).expand(B, -1, -1).clone()
        return x0, P0

    def predict_batched(self, x, P, u):
        """x: (B,2), P: (B,2,2), u: (B,) → x_pred, P_pred"""
        x_pred = self.dynamics_batched(x, u)
        F = self.dynamics_jacobian_batched(x, u)          # (B, 2, 2)
        Q = self.Q.unsqueeze(0)                            # (1, 2, 2)
        P_pred = torch.bmm(torch.bmm(F, P), F.transpose(-1, -2)) + Q
        return x_pred, P_pred

    def update_batched(self, x_pred, P_pred, z):
        """x_pred: (B,2), P_pred: (B,2,2), z: (B,2) → x_upd, P_upd"""
        B = x_pred.shape[0]
        H = self.observation_jacobian_batched(x_pred)     # (B, 2, 2)
        z_pred = self.observation_model_batched(x_pred)    # (B, 2)

        y = (z - z_pred).unsqueeze(-1)                     # (B, 2, 1)

        R = self.R.unsqueeze(0)                            # (1, 2, 2)
        S = torch.bmm(torch.bmm(H, P_pred), H.transpose(-1, -2)) + R

        # Kalman gain via solve: S @ K^T = H @ P_pred → K^T, then transpose
        HP = torch.bmm(H, P_pred)                         # (B, 2, 2)
        K = torch.linalg.solve(S, HP).transpose(-1, -2)   # (B, 2, 2)

        x_upd = x_pred + torch.bmm(K, y).squeeze(-1)      # (B, 2)

        I = torch.eye(2, device=x_pred.device).unsqueeze(0)
        IKH = I - torch.bmm(K, H)
        P_upd = (
            torch.bmm(torch.bmm(IKH, P_pred), IKH.transpose(-1, -2))
            + torch.bmm(torch.bmm(K, R.expand(B, -1, -1)), K.transpose(-1, -2))
        )
        return x_upd, P_upd

    def forward_batched(self, z, u, x_prev, P_prev):
        """Single batched EKF step: predict then update."""
        x_pred, P_pred = self.predict_batched(x_prev, P_prev, u)
        return self.update_batched(x_pred, P_pred, z)

    def get_policy_input_batched(self, x_est, P_est):
        """x_est: (B,2), P_est: (B,2,2) → (B, 4)"""
        p_diag = torch.diagonal(P_est, dim1=-2, dim2=-1)  # (B, 2)
        return torch.cat([x_est, p_diag], dim=-1)
