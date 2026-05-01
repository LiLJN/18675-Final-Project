# 18675-Final-Project

Final project for **18-675**. The project investigates using a **Kalman filter**
to reconstruct missing (masked) dimensions of an environment's observation
space, and trains modern continuous-control RL agents (**PPO**, **SAC**, **TD3**)
on the reconstructed observations to maximize episodic reward.

In all experiments the **velocity-like** dimensions of the observation are
masked out. A per-dimension Kalman filter then estimates each hidden velocity
from its visible position/angle partner (e.g. `x → v_x`, `θ → ω`), and the
filled-in observation is fed to the policy. Training metrics, KF prediction
plots, and evaluation videos are all logged to **Weights & Biases**.

## Project layout

| File | Purpose |
| --- | --- |
| `runner.py` | Unified training script (PPO / SAC / TD3 + Kalman filter + wandb logging) |
| `kalman_filter.py` | `KalmanFilter`, `MaskedObsWrapper`, `make_masked_env`, env presets |
| `ppo_agent.py` | PPO agent |
| `sac_agent.py` | SAC agent |
| `td3_agent.py` | TD3 agent |
| `policies.py` | Actor / critic network definitions |
| `buffer.py` | Replay / rollout buffers |
| `utils.py` | Seeding, env creation, plotting helpers |

## Dependencies

- Python 3.10+
- [PyTorch](https://pytorch.org/) (`torch`)
- [Gymnasium](https://gymnasium.farama.org/) (`gymnasium`)
  - `gymnasium[box2d]` — required for `LunarLanderContinuous-v3`
  - `gymnasium[mujoco]` — required for `HalfCheetah-v4`, `Hopper-v4`, etc.
  - `gymnasium[other]` — pulls in `moviepy` for video recording
- `numpy`
- `matplotlib`
- `wandb` (Weights & Biases)

Quick install:

```bash
pip install torch numpy matplotlib wandb
pip install "gymnasium[box2d,mujoco,other]"
```

## Running `runner.py`

The default configuration trains a PPO agent on `LunarLanderContinuous-v3`
with all velocity dimensions masked and reconstructed by the Kalman filter:

```bash
python runner.py
```

### Common options

```bash
python runner.py \
    --agent td3 \
    --env_id LunarLanderContinuous-v3 \
    --total_steps 1000000 \
    --seed 42000 \
    --wandb_project 18675-Final-Project \
    --wandb_run_name my-run
```

### Selected CLI flags

| Flag | Default | Description |
| --- | --- | --- |
| `--agent {ppo,sac,td3}` | `ppo` | RL algorithm |
| `--env_id` | `LunarLanderContinuous-v3` | Gymnasium env (also supports `HalfCheetah-v4`, `Hopper-v4`, …) |
| `--total_steps` | `1000000` | Total environment steps |
| `--cpu` | off | Force CPU even if CUDA is available |
| `--seed` | `42000` | Random seed |
| `--lr` | `3e-4` | Optimizer learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--kf_dt` | `0.05` | Kalman filter timestep |
| `--kf_process_noise` | `0.1` | KF process-noise std |
| `--kf_measurement_noise` | `0.01` | KF measurement-noise std |
| `--kf_log_every` | `1000` | Log KF MAE every N steps |
| `--kf_plot_every` | `50000` | Log KF comparison plots every N steps (`0` to disable) |
| `--eval_every` | `50000` | Run greedy evaluation every N steps |
| `--eval_episodes` | `10` | Episodes per evaluation |
| `--video_every` | `100000` | Record + upload eval video every N steps (`0` to disable) |
| `--wandb_project` | `18675-Final-Project` | wandb project name |
| `--wandb_run_name` | (auto-generated) | wandb run name |
| `--wandb_api_key` | (env / netrc) | API key (or set `WANDB_API_KEY`) |

PPO-, SAC- and TD3-specific hyperparameters (`--rollout_steps`, `--clip_coef`,
`--tau`, `--alpha`, `--policy_noise`, `--delay`, `--buffer_size`,
`--warmup_steps`, etc.) are also exposed — see `runner.py` for the full list.

### Weights & Biases

Authenticate once with either:

```bash
export WANDB_API_KEY=<your-key>
# or
wandb login
```

Then every run will stream training curves, KF prediction plots, evaluation
returns, and rendered evaluation videos to your wandb project.

### Example commands

Train SAC on HalfCheetah with masked velocities:

```bash
python runner.py --agent sac --env_id HalfCheetah-v4 --total_steps 2000000
```

Train TD3 on Hopper, force CPU, disable video logging:

```bash
python runner.py --agent td3 --env_id Hopper-v4 --cpu --video_every 0
```
