"""
Kalman Filter for recovering masked observation dimensions in Gymnasium environments.

This version is compatible with runner.py.

Core idea
---------
For each masked dimension, the filter keeps a 2D latent state.  When the masked
quantity is a velocity-like variable that has a known visible partner
(e.g. x -> v_x, y -> v_y, theta -> omega), the state is

    [paired_visible_quantity, masked_velocity]

and the visible quantity is used as the measurement every step so the filter can
infer the hidden velocity from the measured position / angle trajectory.

When no known visible partner exists, the filter falls back to the original
prediction-only model for that masked dimension:

    [hidden_value, hidden_rate]

The predicted velocities are written back into the observation vector at their
original indices so the agent receives a complete (reconstructed) observation.

Public API (kept compatible with runner.py):
    - KalmanFilter
    - ENV_OBS_PRESETS
    - list_presets
    - resolve_mask
    - MaskedObsWrapper
    - make_masked_env
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym


# ═══════════════════════════════════════════════════════════════
# Kalman Filter
# ═══════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    Kalman Filter that estimates masked (hidden) observation dimensions.

    For masked dimensions that correspond to known velocity-like variables,
    the per-dimension state is

        x = [paired_visible_position_or_angle, masked_velocity]

    with transition

        [p_{k+1}]   [1 dt] [p_k]
        [v_{k+1}] = [0  1] [v_k]  + process noise

    and measurement

        z_k = [1 0] x_k + measurement noise

    where z_k is the current visible paired position / angle.

    For masked dimensions without a known visible pair, the filter falls back
    to prediction-only mode.

    Args:
        obs_dim:               full observation dimensionality
        masked_indices:        which dims are zeroed out / hidden
        env_id:                Gymnasium env id — used to look up
                               position->velocity pairings
        dt:                    assumed time-step between observations
        process_noise_std:     std of process noise
        measurement_noise_std: std of measurement noise
    """

    # ── Known visible->masked-velocity pairings per environment ──
    # Each tuple is (visible_position_or_angle_index, masked_velocity_index).
    ENV_STATE_PAIRS: dict[str, list[tuple[int, int]]] = {
        # LunarLanderContinuous-v3  (obs_dim = 8)
        #   0: x   1: y   2: v_x   3: v_y   4: theta   5: omega   6: left_leg   7: right_leg
        "LunarLanderContinuous-v3": [(0, 2), (1, 3), (4, 5)],

        # HalfCheetah-v4  (obs_dim = 17)
        #   0-7: positions/angles   8: v_x   9: v_z   10: omega_tip   11-16: joint omega
        "HalfCheetah-v4": [
            (0, 9), (1, 10),
            (2, 11), (3, 12), (4, 13), (5, 14), (6, 15), (7, 16),
        ],

        # Hopper-v4  (obs_dim = 11)
        #   0-4: positions/angles   5: v_x   6: v_z   7: omega_torso   8-10: joint omega
        "Hopper-v4": [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10)],

        # Walker2d-v4  (obs_dim = 17)
        "Walker2d-v4": [
            (0, 10), (1, 11),
            (2, 12), (3, 13), (4, 14), (5, 15), (6, 16),
        ],

        # Pendulum-v1  (obs_dim = 3)
        #   cos(theta), sin(theta) -> omega is nonlinear, no linear pair.
        "Pendulum-v1": [],
    }

    def __init__(
        self,
        obs_dim: int,
        masked_indices: list | np.ndarray,
        env_id: str = "",
        dt: float = 1.0,
        process_noise_std: float = 0.1,
        measurement_noise_std: float = 0.01,
    ):
        self.obs_dim = int(obs_dim)
        self.masked_indices = np.asarray(masked_indices, dtype=int)
        self.unmasked_indices = np.array(
            [i for i in range(self.obs_dim) if i not in self.masked_indices],
            dtype=int,
        )
        self.n_masked = len(self.masked_indices)
        self.dt = float(dt)
        self.env_id = env_id

        # Constant-velocity state-transition.
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]], dtype=float)
        # Measurement model: observe the first state component (position/angle).
        self.H = np.array([[1.0, 0.0]], dtype=float)

        q = float(process_noise_std) ** 2
        self.Q = q * np.array(
            [[self.dt ** 3 / 3.0, self.dt ** 2 / 2.0],
             [self.dt ** 2 / 2.0, self.dt]],
            dtype=float,
        )
        self.R = np.array([[float(measurement_noise_std) ** 2]], dtype=float)

        # Per-masked-dimension latent state: each row is [component_0, component_1].
        self.x = np.zeros((self.n_masked, 2), dtype=float)
        self.P = np.stack([np.eye(2, dtype=float)] * self.n_masked)

        # For each masked dim: either the visible partner index, or None.
        #   paired  -> output estimate is self.x[i, 1]  (velocity)
        #   None    -> output estimate is self.x[i, 0]  (fallback)
        self.paired_visible_idx: list[int | None] = self._build_pairs()

        # ── Logging: full observation vectors ──
        self._predicted_obs_log: list[np.ndarray] = []
        self._true_obs_log: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Pair lookup
    # ------------------------------------------------------------------

    def _build_pairs(self) -> list[int | None]:
        """
        Look up visible->velocity pairs for this env_id and return a list
        (one entry per masked dim) of either the visible partner index or None.
        """
        masked_set = set(int(i) for i in self.masked_indices.tolist())

        # Try exact match first, then strip version suffix for robustness.
        pairs = self.ENV_STATE_PAIRS.get(self.env_id, None)
        if pairs is None:
            base = self.env_id.rsplit("-", 1)[0] if "-" in self.env_id else ""
            for key, val in self.ENV_STATE_PAIRS.items():
                if key.rsplit("-", 1)[0] == base:
                    pairs = val
                    break
        if pairs is None:
            pairs = []

        pair_map: dict[int, int] = {}
        for visible_idx, masked_idx in pairs:
            if masked_idx in masked_set:
                pair_map[masked_idx] = visible_idx

        result = [pair_map.get(int(idx)) for idx in self.masked_indices]

        # Print summary once at construction.
        n_paired = sum(1 for v in result if v is not None)
        n_fallback = self.n_masked - n_paired
        print(f"KalmanFilter: {n_paired} paired dims, {n_fallback} fallback dims "
              f"(env_id={self.env_id!r})")
        for i, (midx, vidx) in enumerate(zip(self.masked_indices, result)):
            tag = (f"visible[{vidx}] -> masked[{midx}]"
                   if vidx is not None
                   else f"masked[{midx}] (prediction-only)")
            print(f"  dim {i}: {tag}")

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, full_obs: np.ndarray | None = None):
        """
        Reset filter state.

        For paired dims: initialise position component from the visible partner,
        velocity component to 0.
        For fallback dims: initialise to 0.
        """
        self.x[:] = 0.0

        if full_obs is not None:
            full_obs = np.asarray(full_obs, dtype=float)
            for i, visible_idx in enumerate(self.paired_visible_idx):
                if visible_idx is not None:
                    self.x[i, 0] = float(full_obs[visible_idx])

        self.P = np.stack([np.eye(2, dtype=float)] * self.n_masked)

    def predict(self):
        """Time-update (predict) step for every masked dimension."""
        for i in range(self.n_masked):
            self.x[i] = self.F @ self.x[i]
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q

    def update(self, z: float, dim_index: int):
        """
        Measurement-update for one masked dimension.

        For paired dims, z should be the current visible partner value.
        For fallback dims, z is a direct measurement of the hidden quantity.
        """
        z_arr = np.array([float(z)], dtype=float)
        y = z_arr - self.H @ self.x[dim_index]
        S = self.H @ self.P[dim_index] @ self.H.T + self.R
        K = self.P[dim_index] @ self.H.T @ np.linalg.inv(S)
        self.x[dim_index] = self.x[dim_index] + (K @ y).flatten()
        I2 = np.eye(2, dtype=float)
        self.P[dim_index] = (I2 - K @ self.H) @ self.P[dim_index]

    def step(self, masked_obs: np.ndarray, full_obs_gt: np.ndarray | None = None):
        """
        Full predict -> update cycle.

        1. Predict all masked dimensions forward.
        2. For paired dims: run measurement update using the visible partner
           value from ``masked_obs`` (positions/angles are NOT masked).
        3. Build the reconstructed observation vector:
           - Start from ``masked_obs`` (has zeros at masked indices).
           - Plug KF-estimated velocities into their correct indices.

        For LunarLanderContinuous-v3 with mask="velocities" this means:
            reconstructed[2] = predicted v_x
            reconstructed[3] = predicted v_y
            reconstructed[5] = predicted omega

        Args:
            masked_obs:  observation the agent sees (masked dims = 0)
            full_obs_gt: ground-truth full observation -- used ONLY for logging,
                         never fed to the filter or agent.

        Returns:
            reconstructed_obs: masked_obs with masked dims filled by KF estimates.
        """
        masked_obs = np.asarray(masked_obs, dtype=float)

        # 1. Predict
        self.predict()

        # 2. Measurement update for paired dims
        for i, visible_idx in enumerate(self.paired_visible_idx):
            if visible_idx is None:
                continue
            z = float(masked_obs[visible_idx])
            self.update(z, i)

        # 3. Reconstruct: plug predicted values into the observation vector
        reconstructed = masked_obs.copy()
        for i, masked_idx in enumerate(self.masked_indices):
            if self.paired_visible_idx[i] is not None:
                # Paired dim: the velocity is the second state component
                reconstructed[int(masked_idx)] = self.x[i, 1]
            else:
                # Fallback dim: the hidden value is the first state component
                reconstructed[int(masked_idx)] = self.x[i, 0]

        # 4. Log full predicted obs vs full ground truth
        if full_obs_gt is not None:
            full_obs_gt = np.asarray(full_obs_gt, dtype=float)
            self._predicted_obs_log.append(reconstructed.copy())
            self._true_obs_log.append(full_obs_gt.copy())

        return reconstructed

    # ------------------------------------------------------------------
    # Accessors for logging
    # ------------------------------------------------------------------

    def get_current_estimates(self) -> np.ndarray:
        """Current estimates for all masked dims. Shape (n_masked,)."""
        out = np.zeros(self.n_masked, dtype=float)
        for i in range(self.n_masked):
            out[i] = (self.x[i, 1] if self.paired_visible_idx[i] is not None
                       else self.x[i, 0])
        return out

    def get_estimation_error(self) -> float | None:
        """MAE over masked dims across logged steps."""
        if not self._predicted_obs_log:
            return None
        pred = np.array(self._predicted_obs_log)[:, self.masked_indices]
        true = np.array(self._true_obs_log)[:, self.masked_indices]
        return float(np.mean(np.abs(pred - true)))

    def get_logs(self):
        """
        Return (predicted_obs, true_obs) arrays of shape (T, obs_dim).

        These are the FULL observation vectors (all dims), not just masked dims,
        so the caller can slice whichever dimensions it wants.
        """
        if not self._predicted_obs_log:
            return None, None
        return np.array(self._predicted_obs_log), np.array(self._true_obs_log)

    def get_masked_dim_logs(self):
        """
        Convenience: return (pred, true) sliced to masked dims only.
        Shape (T, n_masked).
        """
        pred_full, true_full = self.get_logs()
        if pred_full is None:
            return None, None
        return pred_full[:, self.masked_indices], true_full[:, self.masked_indices]

    def clear_logs(self):
        self._predicted_obs_log.clear()
        self._true_obs_log.clear()


# ═══════════════════════════════════════════════════════════════
# Environment masking presets & wrapper
# ═══════════════════════════════════════════════════════════════

ENV_OBS_PRESETS: dict[str, dict[str, list[int]]] = {
    # --- HalfCheetah-v4  (obs_dim = 17) ---
    "HalfCheetah-v4": {
        "positions":  [0, 1, 2, 3, 4, 5, 6, 7],
        "velocities": [8, 9, 10, 11, 12, 13, 14, 15, 16],
        "front_body": [0, 1, 8, 9, 10],
        "joints":     [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16],
    },

    # --- Hopper-v4  (obs_dim = 11) ---
    "Hopper-v4": {
        "positions":  [0, 1, 2, 3, 4],
        "velocities": [5, 6, 7, 8, 9, 10],
        "torso":      [0, 1, 5, 6, 7],
        "joints":     [2, 3, 4, 8, 9, 10],
    },

    # --- Walker2d-v4  (obs_dim = 17) ---
    "Walker2d-v4": {
        "positions":  [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "velocities": [9, 10, 11, 12, 13, 14, 15, 16],
    },

    # --- Ant-v4  (obs_dim = 27) ---
    "Ant-v4": {
        "positions":  list(range(0, 13)),
        "velocities": list(range(13, 27)),
    },

    # --- LunarLanderContinuous-v3  (obs_dim = 8) ---
    #   0: x   1: y   2: v_x   3: v_y   4: theta   5: omega   6: left_leg   7: right_leg
    "LunarLanderContinuous-v3": {
        "positions":    [0, 1],
        "velocities":   [2, 3, 5],
        "orientation":  [4],
        "contacts":     [6, 7],
    },

    # --- Pendulum-v1  (obs_dim = 3) ---
    "Pendulum-v1": {
        "angle":      [0, 1],
        "velocities": [2],
    },
}


def list_presets(env_id: str):
    """Print the available masking presets for an environment."""
    presets = ENV_OBS_PRESETS.get(env_id, {})
    if not presets:
        print(f"No presets defined for '{env_id}'. Use raw indices instead.")
        return
    print(f"\nMasking presets for {env_id}:")
    print("-" * 50)
    for name, indices in presets.items():
        print(f"  {name:15s} -> dims {indices}")
    print()


def resolve_mask(env_id: str, mask_spec: str, obs_dim: int) -> list[int]:
    """
    Turn a user-provided mask specification into a validated list of indices.

    mask_spec can be:
      - A preset name        e.g.  "velocities"
      - Comma-sep indices    e.g.  "2,5,7"
      - preset+indices mix   e.g.  "velocities,0,1"
    """
    if not mask_spec or mask_spec.strip() == "":
        raise ValueError(
            "mask_spec must not be empty -- you must choose which observation "
            "dimensions to mask. Use --list_presets to see named groups, or "
            "pass comma-separated indices like '2,5,7'."
        )

    presets = ENV_OBS_PRESETS.get(env_id, {})
    indices: list[int] = []

    for token in mask_spec.split(","):
        token = token.strip()
        if token in presets:
            indices.extend(presets[token])
        else:
            try:
                indices.append(int(token))
            except ValueError as exc:
                available = list(presets.keys()) if presets else ["(none)"]
                raise ValueError(
                    f"'{token}' is neither a valid preset for {env_id} "
                    f"nor an integer index. Available presets: {available}"
                ) from exc

    indices = sorted(set(indices))
    for idx in indices:
        if idx < 0 or idx >= obs_dim:
            raise ValueError(
                f"Masked index {idx} is out of range for obs_dim={obs_dim}"
            )

    if len(indices) == 0:
        raise ValueError("Resolved mask is empty -- nothing would be masked.")

    if len(indices) >= obs_dim:
        raise ValueError(
            f"Cannot mask ALL {obs_dim} observation dimensions -- the agent "
            f"needs at least one visible dimension."
        )

    return indices


class MaskedObsWrapper(gym.Wrapper):
    """
    Wrap a Gymnasium environment so selected observation indices are zeroed out
    before being returned to the agent.  ``last_full_obs`` always holds the
    ground-truth unmasked observation.
    """

    def __init__(self, env, masked_indices: list[int]):
        assert len(masked_indices) > 0, "masked_indices must be non-empty"
        super().__init__(env)
        self.masked_indices = np.asarray(masked_indices, dtype=int)
        self.last_full_obs: np.ndarray | None = None

    def _mask(self, obs: np.ndarray) -> np.ndarray:
        masked = obs.copy()
        masked[self.masked_indices] = 0.0
        return masked

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_full_obs = obs.copy()
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_full_obs = obs.copy()
        return self._mask(obs), reward, terminated, truncated, info


def make_masked_env(env_id: str, mask_spec: str, render: bool = False,
                    seed: int | None = None):
    """
    Create a masked Gymnasium environment and return (env, masked_indices).
    """
    render_mode = "rgb_array" if render else None
    raw_env = gym.make(env_id, render_mode=render_mode)
    obs_dim = int(np.prod(raw_env.observation_space.shape))

    masked_indices = resolve_mask(env_id, mask_spec, obs_dim)

    env = MaskedObsWrapper(raw_env, masked_indices)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    return env, masked_indices