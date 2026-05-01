# environment.py

import gymnasium as gym
import numpy as np

class PartiallyObservablePendulum(gym.Wrapper):
    def __init__(self, noise_std=0.0):
        super().__init__(gym.make("Pendulum-v1"))
        # Only observe [cos(θ), sin(θ)] — hide θ_dot
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.noise_std = noise_std

    def _mask(self, obs, mask_idx=2):
        masked = obs[:mask_idx]  # drop angular velocity
        if self.noise_std > 0:
            masked = masked + np.random.normal(0, self.noise_std, size=masked.shape)
        return masked.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["full_state"] = obs  # keep for evaluation
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["full_state"] = obs
        return self._mask(obs), reward, terminated, truncated, info

if __name__ == "__main__":

    env = PartiallyObservablePendulum(noise_std=0.0)   # clean
    env_noisy = PartiallyObservablePendulum(noise_std=0.1)  # noisy variant

    obs, info = env.reset()
    print(f"Masked obs shape: {obs.shape}")       # (2,)
    print(f"Full state shape: {info['full_state'].shape}")  # (3,)
