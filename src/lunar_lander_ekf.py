"""
Differentiable Extended Kalman Filter — Lunar Lander (Continuous Actions)
=========================================================================

Mirrors the structure of the Pendulum DifferentiableEKF exactly:
  - Single-env methods  (dynamics, predict, update, forward)
      → used during data collection with torch.no_grad()
  - Batched methods      (*_batched)
      → used during training to re-run EKF over sequences with gradients

State-space model
-----------------
  State   x ∈ R^6 : [px, py, vx, vy, theta, omega]
  Obs     z ∈ R^3 : [px, py, theta]        (linear selection, H is constant)
  Action  u ∈ R^2 : [main_throttle, lateral_throttle]
                     main    ∈ [-1, 1], clipped to [0, 1] (only fires when > 0)
                     lateral ∈ [-1, 1]

  Dynamics use an *approximate* rigid-body model:
    - Main engine applies thrust along the body axis, scaled by clamp(u[0], 0, 1)
    - Lateral engine applies torque proportional to u[1]
    - Gravity acts downward
  The learned Q matrix absorbs model mismatch vs. the true Box2D physics.

  Observation model is linear:  h(x) = H x,  H = [[1,0,0,0,0,0],
                                                    [0,1,0,0,0,0],
                                                    [0,0,0,0,1,0]]

Leg contacts (binary) are NOT part of the EKF — they are passed through
by the agent and appended to the policy input after filtering.

Learnable parameters
--------------------
  Q (6x6) and R (3x3) via log-Cholesky parameterization, identical to
  the pendulum EKF but generalized to arbitrary dimensions.
"""

import torch
import torch.nn as nn


# ======================================================================
#  Helpers
# ======================================================================

def build_lower_triangular(log_diag, off_diag, dim, device=None):
    """Construct a lower-triangular matrix L from log-diagonal and
    strict-lower-triangular entries.  Returns L (dim x dim)."""
    L = torch.zeros(dim, dim, device=device or log_diag.device,
                    dtype=log_diag.dtype)
    L[range(dim), range(dim)] = torch.exp(log_diag)
    if off_diag.numel() > 0:
        rows, cols = torch.tril_indices(dim, dim, offset=-1)
        L[rows, cols] = off_diag
    return L


def build_psd_matrix(log_diag, off_diag, dim):
    """L @ L^T + eI  ->  guaranteed positive-definite (dim x dim)."""
    L = build_lower_triangular(log_diag, off_diag, dim)
    return L @ L.t() + 1e-6 * torch.eye(dim, device=log_diag.device)


# ======================================================================
#  EKF Module
# ======================================================================

class LunarLanderEKF(nn.Module):
    """
    Extended Kalman Filter with learnable Q and R for the Lunar Lander
    (continuous action variant).

    Two sets of methods:
      - Single-env methods (dynamics, predict, update, forward)
        -> used during data collection with torch.no_grad()
      - Batched methods (*_batched)
        -> used during training to re-run EKF over sequences with gradients
    """

    def __init__(
        self,
        gravity: float = 10.0,
        main_thrust: float = 6.0,
        side_torque: float = 1.0,
        dt: float = 0.02,           # 1 / FPS;  LunarLander FPS = 50
    ):
        super().__init__()

        # Dimensions
        self.state_dim = 6   # [px, py, vx, vy, theta, omega]
        self.obs_dim = 3     # [px, py, theta]
        self.action_dim = 2  # [main_throttle, lateral_throttle]

        # Approximate physics (Q absorbs mismatch with true Box2D)
        self.gravity = gravity
        self.main_thrust = main_thrust
        self.side_torque = side_torque
        self.dt = dt

        # -- Constant observation matrix --
        # H selects [px, py, theta] from the 6D state
        self.register_buffer(
            "_H", torch.tensor([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ], dtype=torch.float32)
        )

        # -- Learnable Q (6x6): log-Cholesky --
        #   6 diagonal + 15 off-diagonal = 21 parameters
        #   Initial diag: smaller for positions/angle, larger for velocities
        self.q_log_diag = nn.Parameter(torch.tensor(
            [-3.0, -3.0, -2.0, -2.0, -3.0, -2.0]
        ))
        self.q_off_diag = nn.Parameter(torch.zeros(15))

        # -- Learnable R (3x3): log-Cholesky --
        #   3 diagonal + 3 off-diagonal = 6 parameters
        self.r_log_diag = nn.Parameter(torch.tensor([-2.0, -2.0, -2.0]))
        self.r_off_diag = nn.Parameter(torch.zeros(3))

    # -- Covariance properties --

    @property
    def Q(self):
        return build_psd_matrix(self.q_log_diag, self.q_off_diag, 6)

    @property
    def R(self):
        return build_psd_matrix(self.r_log_diag, self.r_off_diag, 3)

    # ==================================================================
    #  Single-env methods  (data collection, no grad)
    # ==================================================================

    def dynamics(self, x, u):
        """
        Approximate rigid-body dynamics (continuous actions).
        x: (6,)  state vector  [px, py, vx, vy, theta, omega]
        u: (2,)  action vector [main_throttle, lateral_throttle]
        Returns: (6,) next state
        """
        px, py, vx, vy, theta, omega = (
            x[0], x[1], x[2], x[3], x[4], x[5]
        )

        # Main engine: only fires when throttle > 0
        main_power = torch.clamp(u[0], 0.0, 1.0)
        lateral = u[1]

        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)

        # Accelerations
        ax = -self.main_thrust * sin_th * main_power
        ay = self.main_thrust * cos_th * main_power - self.gravity
        alpha = self.side_torque * lateral

        # Semi-implicit Euler (update velocity first, then position)
        new_vx = vx + ax * self.dt
        new_vy = vy + ay * self.dt
        new_omega = omega + alpha * self.dt
        new_px = px + new_vx * self.dt
        new_py = py + new_vy * self.dt
        new_theta = theta + new_omega * self.dt

        return torch.stack([new_px, new_py, new_vx, new_vy,
                            new_theta, new_omega])

    def dynamics_jacobian(self, x, u):
        """
        Analytical df/dx  (6x6).  Non-trivial entries appear in the
        theta-column scaled by main_power (continuous, not binary).
        """
        theta = x[4]
        main_power = torch.clamp(u[0], 0.0, 1.0)
        dt = self.dt
        T = self.main_thrust

        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)

        Tm = T * main_power  # effective thrust magnitude

        F = torch.eye(6, device=x.device)
        # Kinematic coupling (always present)
        F[0, 2] = dt      # dpx'/dvx
        F[1, 3] = dt      # dpy'/dvy
        F[4, 5] = dt      # dtheta'/domega

        # Main-engine coupling via theta (scaled by continuous main_power)
        F[2, 4] = -Tm * cos_th * dt            # dvx'/dtheta
        F[3, 4] = -Tm * sin_th * dt            # dvy'/dtheta
        F[0, 4] = -Tm * cos_th * (dt * dt)     # dpx'/dtheta
        F[1, 4] = -Tm * sin_th * (dt * dt)     # dpy'/dtheta

        return F

    def observation_model(self, x):
        """h(x) = H x  ->  (3,)  selects [px, py, theta]."""
        return self._H @ x

    def observation_jacobian(self, x):
        """H is constant  ->  (3, 6)."""
        return self._H

    def init_state(self, z0):
        """
        Initialize from first observation z0 = [px, py, theta].
        Velocities start at 0 with high uncertainty.
        z0: (3,) -> x0: (6,), P0: (6, 6)
        """
        x0 = torch.zeros(6, device=z0.device)
        x0[0] = z0[0]   # px
        x0[1] = z0[1]   # py
        x0[4] = z0[2]   # theta
        # Initial covariance: low for observed, high for hidden
        P0 = torch.diag(torch.tensor(
            [0.1, 0.1, 1.0, 1.0, 0.1, 1.0], device=z0.device
        ))
        return x0, P0

    def predict(self, x, P, u):
        x_pred = self.dynamics(x, u)
        F = self.dynamics_jacobian(x, u)
        P_pred = F @ P @ F.t() + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        H = self.observation_jacobian(x_pred)            # (3, 6)
        y = z - self.observation_model(x_pred)            # (3,)
        S = H @ P_pred @ H.t() + self.R                   # (3, 3)
        K = torch.linalg.solve(S, H @ P_pred).t()         # (6, 3)

        x_upd = x_pred + K @ y
        IKH = torch.eye(6, device=x_pred.device) - K @ H
        P_upd = IKH @ P_pred @ IKH.t() + K @ self.R @ K.t()
        return x_upd, P_upd

    def forward(self, z, u, x_prev, P_prev):
        """Single EKF step: predict -> update."""
        x_pred, P_pred = self.predict(x_prev, P_prev, u)
        return self.update(x_pred, P_pred, z)

    def get_policy_input(self, x_est, P_est):
        """
        Pack EKF output for the policy network.
        Returns: (12,) = [x_est (6), diag(P) (6)]
        NOTE: Leg contacts (2D) should be appended by the agent -> (14,).
        """
        return torch.cat([x_est, torch.diagonal(P_est)])

    # ==================================================================
    #  Batched methods  (training with gradients)
    # ==================================================================

    def dynamics_batched(self, x, u):
        """
        x: (B, 6)   state batch
        u: (B, 2)   continuous action batch [main_throttle, lateral_throttle]
        Returns: (B, 6)
        """
        px    = x[:, 0]
        py    = x[:, 1]
        vx    = x[:, 2]
        vy    = x[:, 3]
        theta = x[:, 4]
        omega = x[:, 5]

        main_power = torch.clamp(u[:, 0], 0.0, 1.0)   # (B,)
        lateral    = u[:, 1]                            # (B,)

        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)

        ax    = -self.main_thrust * sin_th * main_power
        ay    = self.main_thrust * cos_th * main_power - self.gravity
        alpha = self.side_torque * lateral

        new_vx    = vx + ax * self.dt
        new_vy    = vy + ay * self.dt
        new_omega = omega + alpha * self.dt
        new_px    = px + new_vx * self.dt
        new_py    = py + new_vy * self.dt
        new_theta = theta + new_omega * self.dt

        return torch.stack([new_px, new_py, new_vx, new_vy,
                            new_theta, new_omega], dim=-1)

    def dynamics_jacobian_batched(self, x, u):
        """
        Batched df/dx.
        x: (B, 6), u: (B, 2) -> F: (B, 6, 6)

        The theta-column entries scale continuously with main_power.
        """
        B = x.shape[0]
        dt = self.dt
        T = self.main_thrust

        theta = x[:, 4]
        main_power = torch.clamp(u[:, 0], 0.0, 1.0)

        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)
        Tm = T * main_power          # (B,) effective thrust

        F = torch.zeros(B, 6, 6, device=x.device)
        for i in range(6):
            F[:, i, i] = 1.0

        # Kinematic coupling (always present)
        F[:, 0, 2] = dt       # dpx'/dvx
        F[:, 1, 3] = dt       # dpy'/dvy
        F[:, 4, 5] = dt       # dtheta'/domega

        # Main-engine coupling via theta (scaled by continuous main_power)
        F[:, 2, 4] = -Tm * cos_th * dt            # dvx'/dtheta
        F[:, 3, 4] = -Tm * sin_th * dt            # dvy'/dtheta
        F[:, 0, 4] = -Tm * cos_th * (dt * dt)     # dpx'/dtheta
        F[:, 1, 4] = -Tm * sin_th * (dt * dt)     # dpy'/dtheta

        return F

    def observation_model_batched(self, x):
        """x: (B, 6) -> z_pred: (B, 3).  Linear: H @ x."""
        return (self._H @ x.unsqueeze(-1)).squeeze(-1)

    def observation_jacobian_batched(self, x):
        """Returns (B, 3, 6).  Constant H expanded over batch."""
        B = x.shape[0]
        return self._H.unsqueeze(0).expand(B, -1, -1)

    def init_state_batched(self, z0):
        """z0: (B, 3) -> x0: (B, 6), P0: (B, 6, 6)"""
        B = z0.shape[0]
        x0 = torch.zeros(B, 6, device=z0.device)
        x0[:, 0] = z0[:, 0]
        x0[:, 1] = z0[:, 1]
        x0[:, 4] = z0[:, 2]

        p0_diag = torch.tensor(
            [0.1, 0.1, 1.0, 1.0, 0.1, 1.0], device=z0.device
        )
        P0 = torch.diag(p0_diag).unsqueeze(0).expand(B, -1, -1).clone()
        return x0, P0

    def predict_batched(self, x, P, u):
        """x: (B,6), P: (B,6,6), u: (B,2) -> x_pred, P_pred"""
        x_pred = self.dynamics_batched(x, u)
        F = self.dynamics_jacobian_batched(x, u)
        Q = self.Q.unsqueeze(0)
        P_pred = torch.bmm(torch.bmm(F, P),
                           F.transpose(-1, -2)) + Q
        return x_pred, P_pred

    def update_batched(self, x_pred, P_pred, z):
        """x_pred: (B,6), P_pred: (B,6,6), z: (B,3) -> x_upd, P_upd"""
        B = x_pred.shape[0]
        H = self.observation_jacobian_batched(x_pred)
        z_pred = self.observation_model_batched(x_pred)

        y = (z - z_pred).unsqueeze(-1)

        R = self.R.unsqueeze(0)
        HP = torch.bmm(H, P_pred)
        S = torch.bmm(HP, H.transpose(-1, -2)) + R

        K = torch.linalg.solve(S, HP).transpose(-1, -2)

        x_upd = x_pred + torch.bmm(K, y).squeeze(-1)

        I = torch.eye(6, device=x_pred.device).unsqueeze(0)
        IKH = I - torch.bmm(K, H)
        P_upd = (
            torch.bmm(torch.bmm(IKH, P_pred),
                      IKH.transpose(-1, -2))
            + torch.bmm(torch.bmm(K, R.expand(B, -1, -1)),
                        K.transpose(-1, -2))
        )
        return x_upd, P_upd

    def forward_batched(self, z, u, x_prev, P_prev):
        """Single batched EKF step: predict -> update."""
        x_pred, P_pred = self.predict_batched(x_prev, P_prev, u)
        return self.update_batched(x_pred, P_pred, z)

    def get_policy_input_batched(self, x_est, P_est):
        """
        x_est: (B, 6), P_est: (B, 6, 6) -> (B, 12)
        NOTE: Leg contacts (B, 2) should be appended by the agent
              to produce the final (B, 14) policy input.
        """
        p_diag = torch.diagonal(P_est, dim1=-2, dim2=-1)
        return torch.cat([x_est, p_diag], dim=-1)


# ======================================================================
#  Smoke tests
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")
    ekf = LunarLanderEKF().to(device)

    print("=" * 60)
    print("LunarLanderEKF (Continuous Actions) — smoke tests")
    print("=" * 60)

    # -- 1. Parameter counts --
    n_params = sum(p.numel() for p in ekf.parameters())
    print(f"\nLearnable parameters: {n_params}")
    print(f"  Q: 6 diag + 15 off-diag = 21")
    print(f"  R: 3 diag + 3 off-diag  = 6")
    print(f"  Total: {n_params}")

    # -- 2. Covariance shapes & PD check --
    Q = ekf.Q
    R = ekf.R
    print(f"\nQ shape: {Q.shape},  min eigenvalue: {torch.linalg.eigvalsh(Q).min():.6f}")
    print(f"R shape: {R.shape},  min eigenvalue: {torch.linalg.eigvalsh(R).min():.6f}")
    assert torch.linalg.eigvalsh(Q).min() > 0, "Q not PD!"
    assert torch.linalg.eigvalsh(R).min() > 0, "R not PD!"

    # -- 3. Single-env forward pass with continuous actions --
    z0 = torch.tensor([0.0, 1.4, -0.01])
    x_est, P_est = ekf.init_state(z0)
    print(f"\nInit state: x={x_est.tolist()}")

    test_actions = [
        ("no thrust",        torch.tensor([0.0, 0.0])),
        ("full main",        torch.tensor([1.0, 0.0])),
        ("half main + left", torch.tensor([0.5, -0.8])),
        ("lateral only",     torch.tensor([0.0, 1.0])),
        ("negative main",    torch.tensor([-0.5, 0.0])),  # should clamp to 0
    ]
    z = torch.tensor([0.01, 1.38, -0.02])
    for label, action in test_actions:
        x_new, P_new = ekf.forward(z, action, x_est, P_est)
        pi_input = ekf.get_policy_input(x_new, P_new)
        print(f"  {label:20s}: vy_est={x_new[3].item():.4f}, "
              f"policy_input dim={pi_input.shape[0]}")

    # -- 4. Batched forward pass --
    B = 8
    z0_batch = torch.randn(B, 3) * 0.1
    x_b, P_b = ekf.init_state_batched(z0_batch)
    assert x_b.shape == (B, 6)
    assert P_b.shape == (B, 6, 6)

    u_batch = torch.randn(B, 2).clamp(-1, 1)   # continuous actions
    z_batch = torch.randn(B, 3) * 0.1
    x_upd, P_upd = ekf.forward_batched(z_batch, u_batch, x_b, P_b)
    pi_batch = ekf.get_policy_input_batched(x_upd, P_upd)
    assert x_upd.shape == (B, 6)
    assert P_upd.shape == (B, 6, 6)
    assert pi_batch.shape == (B, 12)
    print(f"\nBatched: x_upd {x_upd.shape}, P_upd {P_upd.shape}, "
          f"policy_input {pi_batch.shape}")

    # -- 5. Gradient flow check --
    ekf.zero_grad()
    z0_g = torch.randn(4, 3) * 0.1
    x_g, P_g = ekf.init_state_batched(z0_g)

    for t in range(5):
        z_t = torch.randn(4, 3) * 0.1
        u_t = torch.randn(4, 2).clamp(-1, 1)
        x_g, P_g = ekf.forward_batched(z_t, u_t, x_g, P_g)

    pi_g = ekf.get_policy_input_batched(x_g, P_g)
    loss = pi_g.sum()
    loss.backward()

    q_grad = ekf.q_log_diag.grad
    r_grad = ekf.r_log_diag.grad
    print(f"\nGradient flow:")
    print(f"  Q log-diag grad: {q_grad}")
    print(f"  R log-diag grad: {r_grad}")
    assert q_grad is not None and q_grad.abs().sum() > 0, "No grad to Q!"
    assert r_grad is not None and r_grad.abs().sum() > 0, "No grad to R!"

    # -- 6. Clamp behavior: negative main throttle = no thrust --
    x_test = torch.zeros(6)
    x_test[1] = 1.0
    u_neg = torch.tensor([-1.0, 0.0])
    u_zero = torch.tensor([0.0, 0.0])
    x_neg = ekf.dynamics(x_test, u_neg)
    x_zero = ekf.dynamics(x_test, u_zero)
    assert torch.allclose(x_neg, x_zero), \
        f"Negative throttle should equal zero throttle!\n  neg={x_neg}\n  zero={x_zero}"
    print("\nClamp check: negative throttle == zero throttle ✓")

    # -- 7. Multi-step sequence --
    seq_len, B2 = 20, 16
    z_seq = torch.randn(seq_len, B2, 3) * 0.1
    u_seq = torch.randn(seq_len, B2, 2).clamp(-1, 1)

    x_s, P_s = ekf.init_state_batched(z_seq[0])
    for t in range(1, seq_len):
        x_s, P_s = ekf.forward_batched(z_seq[t], u_seq[t], x_s, P_s)

    final_pi = ekf.get_policy_input_batched(x_s, P_s)
    print(f"\n20-step sequence: final policy input {final_pi.shape}")

    print("\n✓ All smoke tests passed.")
