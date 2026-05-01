"""
Partially Observable Lunar Lander Environment (Continuous Actions)
==================================================================

Wraps LunarLander-v3 (continuous=True) to hide velocity components,
mirroring the PartiallyObservablePendulum pattern.

Action space (continuous, 2D):
    u[0] : main engine throttle   ∈ [-1, 1]  (only fires when > 0)
    u[1] : lateral engine throttle ∈ [-1, 1]

Full state (8D):
    [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]

Masked observation (5D, default):
    [x, y, angle, left_leg_contact, right_leg_contact]

Hidden (3D):  vx, vy, angular_vel  — these are what the EKF must estimate.

Design notes for EKF integration:
    - The continuous state for the EKF is 6D: [x, y, vx, vy, angle, angular_vel]
    - The observation model maps this 6D state to 3D: [x, y, angle]
    - Leg contacts are binary/discrete and passed through unchanged —
      they should be appended to the EKF estimate before feeding into the policy.
    - `info["full_state"]`      → full 8D vector (for logging / ground truth)
    - `info["continuous_state"]` → 6D continuous-only (for EKF supervision if needed)
    - `info["leg_contacts"]`     → 2D binary (for pass-through)
"""

import gymnasium as gym
import numpy as np


# Indices into the full 8D LunarLander observation
IDX_X         = 0
IDX_Y         = 1
IDX_VX        = 2
IDX_VY        = 3
IDX_ANGLE     = 4
IDX_ANG_VEL   = 5
IDX_LEFT_LEG  = 6
IDX_RIGHT_LEG = 7

# Default: hide all three velocity components
DEFAULT_HIDDEN = [IDX_VX, IDX_VY, IDX_ANG_VEL]

# Continuous vs discrete index sets
CONTINUOUS_IDX  = [IDX_X, IDX_Y, IDX_VX, IDX_VY, IDX_ANGLE, IDX_ANG_VEL]
LEG_CONTACT_IDX = [IDX_LEFT_LEG, IDX_RIGHT_LEG]


class PartiallyObservableLunarLander(gym.Wrapper):
    """
    Wraps LunarLander-v3 (continuous) to produce partial observations
    by dropping specified state components (default: velocities).

    Parameters
    ----------
    hidden_indices : list[int]
        Indices into the 8D state vector to hide. Must be a subset of
        the continuous indices [0..5]; leg contacts [6,7] are always
        included in the observation (they're discrete pass-through).
    noise_std : float
        Gaussian noise std added to the *continuous* observed components.
        Leg contacts are never noised.
    """

    def __init__(self, hidden_indices=None, noise_std=0.0):
        super().__init__(gym.make("LunarLander-v3", continuous=True))

        self.hidden_indices = hidden_indices or DEFAULT_HIDDEN
        self.noise_std = noise_std

        # Validate: only continuous indices can be hidden
        for idx in self.hidden_indices:
            assert idx in CONTINUOUS_IDX, (
                f"Can only hide continuous state indices {CONTINUOUS_IDX}, got {idx}"
            )

        # Observed continuous indices = continuous minus hidden
        self.observed_continuous_idx = [
            i for i in CONTINUOUS_IDX if i not in self.hidden_indices
        ]
        # Full observed = observed continuous + leg contacts
        self.observed_idx = self.observed_continuous_idx + LEG_CONTACT_IDX

        obs_dim = len(self.observed_idx)
        # Build observation bounds from the original space
        orig_low = self.env.observation_space.low
        orig_high = self.env.observation_space.high
        self.observation_space = gym.spaces.Box(
            low=orig_low[self.observed_idx],
            high=orig_high[self.observed_idx],
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _mask(self, obs):
        """Extract observed components and optionally add noise."""
        masked = obs[self.observed_idx].copy()

        if self.noise_std > 0:
            # Noise only the continuous part, not leg contacts
            n_cont = len(self.observed_continuous_idx)
            masked[:n_cont] += np.random.normal(
                0, self.noise_std, size=n_cont
            )

        return masked.astype(np.float32)

    def _enrich_info(self, obs, info):
        """Attach full state, continuous state, and leg contacts to info."""
        info["full_state"] = obs                         # 8D
        info["continuous_state"] = obs[CONTINUOUS_IDX]    # 6D — for EKF ground truth
        info["leg_contacts"] = obs[LEG_CONTACT_IDX]       # 2D — binary pass-through
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = self._enrich_info(obs, info)
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._enrich_info(obs, info)
        return self._mask(obs), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_clean_lunar_lander():
    """Position + angle only, no noise, continuous actions."""
    return PartiallyObservableLunarLander(noise_std=0.0)


def make_noisy_lunar_lander(noise_std=0.05):
    """Position + angle only, with Gaussian sensor noise, continuous actions."""
    return PartiallyObservableLunarLander(noise_std=noise_std)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Partially Observable Lunar Lander (Continuous) — smoke test")
    print("=" * 60)

    # --- Clean variant ---
    env = make_clean_lunar_lander()

    print(f"\nAction space: {env.action_space}")
    print(f"Obs space   : {env.observation_space}")

    obs, info = env.reset(seed=42)
    print(f"\n[Clean]")
    print(f"  Masked obs shape : {obs.shape}")               # (5,)
    print(f"  Masked obs       : {obs}")
    print(f"  Full state shape : {info['full_state'].shape}") # (8,)
    print(f"  Continuous state : {info['continuous_state'].shape}")  # (6,)
    print(f"  Leg contacts     : {info['leg_contacts']}")

    # Take steps with continuous actions
    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()   # (2,) continuous
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  After 50 steps   : obs={obs}, reward_sum={total_reward:.1f}")
    print(f"  Sample action    : {action}  (shape {action.shape})")
    env.close()

    # --- Noisy variant ---
    env_noisy = make_noisy_lunar_lander(noise_std=0.1)
    obs, info = env_noisy.reset(seed=42)
    print(f"\n[Noisy, std=0.1]")
    print(f"  Masked obs shape : {obs.shape}")
    print(f"  Masked obs       : {obs}")
    env_noisy.close()

    # --- Custom: hide only vx, vy (keep angular velocity) ---
    env_custom = PartiallyObservableLunarLander(
        hidden_indices=[IDX_VX, IDX_VY], noise_std=0.0
    )
    obs, info = env_custom.reset(seed=42)
    print(f"\n[Custom: hide vx,vy only]")
    print(f"  Masked obs shape : {obs.shape}")               # (6,)
    print(f"  Observed indices : {env_custom.observed_idx}")
    env_custom.close()

    print("\nAll checks passed.")
