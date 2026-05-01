# runner.py
"""
Unified RL training runner with:
  - Optional observation masking + Kalman-filter reconstruction
  - All plots logged to Weights & Biases (no local file saves)
"""

import argparse, os, tempfile
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
import wandb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppo_agent import PPOAgent
from sac_agent import SACAgent
from td3_agent import TD3Agent
from kalman_filter import (
    KalmanFilter,
    make_masked_env,
    MaskedObsWrapper,
    list_presets,
)
from utils import set_seed, make_env, greedy_action, _to_env_action, plot_curves


# ────────────────────────── helpers ──────────────────────────


def create_agent(agent_type, env_info, args, device):
    """Factory function to create the appropriate agent."""
    if agent_type.lower() == "ppo":
        return PPOAgent(
            env_info=env_info, lr=args.lr, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_coef=args.clip_coef,
            vf_coef=args.vf_coef, ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm, update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size, rollout_steps=args.rollout_steps,
            device=device,
        )
    elif agent_type.lower() == "sac":
        return SACAgent(
            env_info=env_info, lr=args.lr, gamma=args.gamma,
            tau=args.tau, alpha=args.alpha, batch_size=args.batch_size,
            update_every=args.update_every, buffer_size=args.buffer_size,
            warmup_steps=args.warmup_steps, utd_ratio=args.utd_ratio,
            device=device,
        )
    elif agent_type.lower() == "td3":
        return TD3Agent(
            env_info=env_info, lr=args.lr, gamma=args.gamma,
            tau=args.tau, delay=args.delay, batch_size=args.batch_size,
            update_every=args.update_every, buffer_size=args.buffer_size,
            warmup_steps=args.warmup_steps, policy_noise=args.policy_noise,
            noise_clip=args.noise_clip, exploration_noise=args.exploration_noise,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_transition(agent_type, obs, action, reward, next_obs,
                      terminated, truncated, action_info):
    if agent_type.lower() in ["sac", "td3"]:
        done_flag = bool(terminated or truncated)
    else:
        done_flag = bool(terminated)

    base = {
        "obs": obs.copy(), "action": action.copy(),
        "reward": float(reward), "next_obs": next_obs.copy(),
        "done": done_flag, "truncated": bool(truncated),
    }
    if agent_type.lower() == "ppo":
        base["log_prob"] = action_info.get("log_prob", 0.0)
        base["value"] = action_info.get("value", 0.0)
    return base


def evaluate_policy_with_kf(agent, env_id, masked_indices, kf_params,
                            episodes=10, seed=42):
    """Evaluate using the same mask + KF pipeline as training."""
    raw_env = make_env(env_id, render=False, seed=seed)
    env = MaskedObsWrapper(raw_env, masked_indices)
    kf = KalmanFilter(
        obs_dim=int(np.prod(raw_env.observation_space.shape)),
        masked_indices=masked_indices, env_id=env_id, **kf_params,
    )

    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        kf.reset(env.last_full_obs)
        obs = kf.step(obs, env.last_full_obs)
        done = truncated = False
        ep_r = 0.0
        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            policy = agent.actor if hasattr(agent, "actor") else agent.policy
            a = greedy_action(policy, obs_t)
            a_env = np.asarray(a.squeeze(0).cpu().numpy(), dtype=np.float32)
            a_env = np.clip(a_env, env.action_space.low, env.action_space.high)
            obs, r, done, truncated, _ = env.step(a_env)
            obs = kf.step(obs, env.last_full_obs)
            ep_r += r
        scores.append(ep_r)
    env.close()
    return float(np.mean(scores)), float(np.std(scores))


def record_eval_video_with_kf(agent, env_id, masked_indices, kf_params,
                               step: int, video_prefix="eval",
                               episodes=1, seed=None):
    """
    Record an evaluation video using the same mask + KF pipeline as training,
    then upload the .mp4 to wandb.

    The video is rendered into a temporary directory, uploaded, and cleaned up
    so nothing is saved locally.
    """
    from gymnasium.wrappers import RecordVideo
    import glob, shutil

    tmp_dir = tempfile.mkdtemp(prefix="rl_video_")
    try:
        # Create env with render_mode="rgb_array" for video capture
        raw_env = make_env(env_id, render=True, seed=seed)
        env = MaskedObsWrapper(raw_env, masked_indices)

        obs_dim = int(np.prod(raw_env.observation_space.shape))
        kf = KalmanFilter(obs_dim=obs_dim, masked_indices=masked_indices,
                          env_id=env_id, **kf_params)

        # Wrap with Gymnasium's RecordVideo
        video_name = f"{video_prefix}_step{step}"
        rec_env = RecordVideo(env, video_folder=tmp_dir,
                              name_prefix=video_name,
                              episode_trigger=lambda _: True)

        policy = agent.actor if hasattr(agent, "actor") else agent.policy

        for ep in range(episodes):
            obs, _ = rec_env.reset(seed=(seed + ep) if seed else None)
            # Read last_full_obs from env (MaskedObsWrapper) directly — RecordVideo
            # does not forward custom attributes reliably through the wrapper chain.
            kf.reset(env.last_full_obs)
            obs = kf.step(obs, env.last_full_obs)
            done = truncated = False
            while not (done or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                a = greedy_action(policy, obs_t)
                a_env = np.asarray(a.squeeze(0).cpu().numpy(), dtype=np.float32)
                a_env = np.clip(a_env, rec_env.action_space.low,
                                rec_env.action_space.high)
                obs, _, done, truncated, _ = rec_env.step(a_env)
                obs = kf.step(obs, env.last_full_obs)
        rec_env.close()

        # Find the mp4(s) that were written
        mp4s = sorted(glob.glob(os.path.join(tmp_dir, "*.mp4")))
        if mp4s:
            # Upload each video to wandb
            for mp4_path in mp4s:
                wandb.log({
                    "video/eval": wandb.Video(mp4_path, fps=30,
                                              format="mp4"),
                }, step=step)
            print(f"  Video uploaded to wandb (step {step})")
        else:
            print(f"  Warning: no .mp4 files found in {tmp_dir}")

    except Exception as e:
        print(f"  Video recording failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp directory — nothing saved locally
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ────────────────── logging / plotting helpers ───────────────────


def log_training_stats(agent_type, stats, ep_return_log, total_steps, updates_performed):
    """Print formatted training statistics to stdout."""
    if not stats:
        return
    avg_return = np.mean(ep_return_log[-10:]) if ep_return_log else 0.0

    if agent_type.lower() == "ppo":
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Loss: {stats.get('loss', 0):>7.4f} | "
            f"π: {stats.get('policy_loss', 0):>7.4f} | "
            f"V: {stats.get('value_loss', 0):>7.4f} | "
            f"H: {stats.get('entropy', 0):>6.3f} | "
            f"KL: {stats.get('kl', 0):>8.5f} | "
            f"Clip%: {stats.get('clipfrac', 0):>5.1%} | "
            f"Ret: {avg_return:>7.1f}"
        )
    elif agent_type.lower() == "sac":
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Actor: {stats.get('actor_loss', 0):>7.4f} | "
            f"C1: {stats.get('critic1_loss', 0):>7.4f} | "
            f"C2: {stats.get('critic2_loss', 0):>7.4f} | "
            f"Q1: {stats.get('q1', 0):>7.2f} | "
            f"Q2: {stats.get('q2', 0):>7.2f} | "
            f"H: {stats.get('entropy', 0):>6.3f} | "
            f"Ret: {avg_return:>7.1f}"
        )
    elif agent_type.lower() == "td3":
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Actor: {stats.get('actor_loss', 0):>7.4f} | "
            f"C1: {stats.get('critic1_loss', 0):>7.4f} | "
            f"C2: {stats.get('critic2_loss', 0):>7.4f} | "
            f"Q1: {stats.get('q1', 0):>7.2f} | "
            f"Q2: {stats.get('q2', 0):>7.2f} | "
            f"Ret: {avg_return:>7.1f}"
        )
    else:
        summary = "  ".join(f"{k}={v:.4f}" for k, v in stats.items())
        print(f"[Step {total_steps:>8d}] {summary}  Ret: {avg_return:>7.1f}")


def _log_curves_to_wandb(ep_steps_log, ep_return_log, loss_log, agent_type, step: int):
    """Build a log dict, call utils.plot_curves, and upload the PNG to wandb."""
    plot_log = {
        "steps": ep_steps_log,
        "episodic_return": ep_return_log,
        **loss_log,
    }
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        plot_curves(plot_log, out_path=tmp_path)
        wandb.log({"plots/training_curves": wandb.Image(tmp_path)}, step=step)
    except Exception as e:
        print(f"  Plotting failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def log_kf_comparison(kf: KalmanFilter, step: int):
    """
    Per-masked-dim: actual observation (ground truth) vs KF-predicted
    observation, overlaid on the same axes.
    Logged as  wandb «plots/kf_obs_comparison».
    """
    est, gt = kf.get_masked_dim_logs()
    if est is None or len(est) < 2:
        return

    n_dims = est.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3.2 * n_dims), squeeze=False)
    for d in range(n_dims):
        ax = axes[d, 0]
        t = np.arange(len(gt))
        ax.plot(t, gt[:, d], color="tab:green", alpha=0.8, linewidth=1.2,
                label="Actual (ground truth)")
        ax.plot(t, est[:, d], color="tab:red", alpha=0.8, linewidth=1.2,
                linestyle="--", label="KF Predicted")
        ax.fill_between(t, gt[:, d], est[:, d], color="salmon", alpha=0.15)
        ax.set_ylabel(f"Obs dim {kf.masked_indices[d]}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Timestep within logging window")
    fig.suptitle(f"Actual vs Predicted Observation  (step {step})", fontsize=13)
    fig.tight_layout()
    wandb.log({"plots/kf_obs_comparison": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # Also log per-dim MAE and RMSE as scalars
    error = est - gt
    per_dim_mae = np.mean(np.abs(error), axis=0)
    per_dim_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    metrics = {}
    for d in range(n_dims):
        dim_idx = int(kf.masked_indices[d])
        metrics[f"kf/mae_dim{dim_idx}"] = float(per_dim_mae[d])
        metrics[f"kf/rmse_dim{dim_idx}"] = float(per_dim_rmse[d])
    wandb.log(metrics, step=step)


def log_kf_abs_error_plot(kf: KalmanFilter, step: int):
    """
    Per-masked-dim: absolute error |KF predicted - ground truth| over time,
    with a moving-average trend line.
    Logged as wandb «plots/kf_abs_error».
    """
    est, gt = kf.get_masked_dim_logs()
    if est is None or len(est) < 2:
        return

    abs_err = np.abs(est - gt)          # shape (T, n_dims)
    t = np.arange(len(abs_err))
    n_dims = abs_err.shape[1]

    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3.2 * n_dims), squeeze=False)
    for d in range(n_dims):
        ax = axes[d, 0]
        err_d = abs_err[:, d]
        ax.plot(t, err_d, color="tab:orange", alpha=0.35, linewidth=0.8,
                label="|error|")
        window = max(1, min(50, len(err_d) // 10))
        if len(err_d) >= window:
            kernel = np.ones(window) / window
            ma = np.concatenate([np.full(window - 1, np.nan),
                                  np.convolve(err_d, kernel, mode="valid")])
            ax.plot(t, ma, color="tab:orange", linewidth=2.0,
                    label=f"MA({window})")
        ax.axhline(float(np.mean(err_d)), color="tab:red", linewidth=1.0,
                   linestyle="--", alpha=0.7, label=f"mean={np.mean(err_d):.4f}")
        ax.set_ylabel(f"Abs error dim {kf.masked_indices[d]}")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Timestep within logging window")
    fig.suptitle(f"KF Absolute Prediction Error  (step {step})", fontsize=13)
    fig.tight_layout()
    wandb.log({"plots/kf_abs_error": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ────────────────────────── main loop ────────────────────────


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    # ── Environment setup — velocities always masked ──
    env, masked_indices = make_masked_env(
        args.env_id, "velocities", render=False, seed=None
    )
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    print(f"Environment : {args.env_id}  (obs={obs_dim}, act={act_dim})")
    print(f"Masked dims : {masked_indices}  ({len(masked_indices)}/{obs_dim} hidden)")

    kf_params = dict(
        dt=args.kf_dt,
        process_noise_std=args.kf_process_noise,
        measurement_noise_std=args.kf_measurement_noise,
    )

    kf = KalmanFilter(obs_dim=obs_dim, masked_indices=masked_indices,
                       env_id=args.env_id, **kf_params)

    # The agent always sees full obs_dim (KF fills in masked dims)
    env_info = {
        "obs_dim": obs_dim, "act_dim": act_dim,
        "act_low": env.action_space.low, "act_high": env.action_space.high,
    }

    set_seed(args.seed)
    agent = create_agent(args.agent, env_info, args, device)
    print(f"Created {args.agent.upper()} agent  (obs={obs_dim}, act={act_dim})")

    # ── wandb init ──
    wandb_config = vars(args).copy()
    wandb_config["masked_indices"] = masked_indices
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.agent}_{args.env_id}_mask{len(masked_indices)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=wandb_config,
    )

    # ── Training loop ──
    obs, _ = env.reset(seed=args.seed)
    kf.reset(env.last_full_obs)
    obs = kf.step(obs, env.last_full_obs)

    ep_ret = 0.0
    total_steps = 0
    best_eval = -1e9
    episode_idx = 0
    updates_performed = 0
    last_log_step = 0

    # Accumulators for summary plots
    ep_steps_log: list[int] = []        # step at which each episode ended
    ep_return_log: list[float] = []     # return of each episode
    loss_log: dict[str, list[float]] = {}  # keyed by stat name

    print(f"Starting training for {args.total_steps} steps …")

    while total_steps < args.total_steps:
        # ── act ──
        action_info = agent.act(obs)
        action = action_info["action"]
        a_env = np.asarray(action, dtype=np.float32).reshape(-1)
        a_env = np.clip(a_env, env.action_space.low, env.action_space.high)

        # ── env step ──
        next_obs_raw, r, terminated, truncated, _ = env.step(a_env)
        full_next = env.last_full_obs

        # Reconstruct next_obs through KF
        next_obs = kf.step(next_obs_raw, full_next)

        stop = bool(terminated or truncated)

        transition = create_transition(
            args.agent, obs, action, r, next_obs, terminated, truncated, action_info
        )
        stats = agent.step(transition)

        ep_ret += r
        obs = next_obs
        total_steps += 1

        # ── episode boundary ──
        if stop:
            ep_steps_log.append(total_steps)
            ep_return_log.append(ep_ret)
            wandb.log({"train/episodic_return": ep_ret,
                       "train/episode": episode_idx}, step=total_steps)
            episode_idx += 1
            obs, _ = env.reset(seed=args.seed + 100 * episode_idx)
            kf.reset(env.last_full_obs)
            obs = kf.step(obs, env.last_full_obs)
            ep_ret = 0.0

        # ── log optimisation stats ──
        if stats:
            updates_performed += 1
            wandb_stats = {f"train/{k}": v for k, v in stats.items()}
            wandb.log(wandb_stats, step=total_steps)

            # Accumulate for loss summary plot
            for k, v in stats.items():
                loss_log.setdefault(k, []).append(v)

            if total_steps - last_log_step >= args.log_every:
                log_training_stats(args.agent, stats, ep_return_log, total_steps, updates_performed)
                last_log_step = total_steps

        # ── KF scalar MAE ──
        if total_steps % args.kf_log_every == 0:
            mae = kf.get_estimation_error()
            if mae is not None:
                wandb.log({"kf/mae": mae}, step=total_steps)

        # ── periodic summary plots ──
        if args.kf_plot_every > 0 and total_steps % args.kf_plot_every == 0:
            log_kf_comparison(kf, total_steps)
            log_kf_abs_error_plot(kf, total_steps)
            kf.clear_logs()

            _log_curves_to_wandb(ep_steps_log, ep_return_log, loss_log, args.agent, total_steps)

        # ── evaluation ──
        if args.eval_every > 0 and total_steps % args.eval_every == 0:
            with torch.no_grad():
                mean_r, std_r = evaluate_policy_with_kf(
                    agent, args.env_id, masked_indices,
                    kf_params,
                    episodes=args.eval_episodes, seed=1000,
                )
            wandb.log({
                "eval/mean_return": mean_r, "eval/std_return": std_r,
            }, step=total_steps)
            print(f"  [Eval @ {total_steps}]  {mean_r:.1f} ± {std_r:.1f}")

            if mean_r > best_eval:
                best_eval = mean_r

        # ── periodic video recording ──
        if args.video_every > 0 and total_steps % args.video_every == 0:
            print(f"Recording video at step {total_steps}...")
            with torch.no_grad():
                record_eval_video_with_kf(
                    agent, args.env_id, masked_indices, kf_params,
                    step=total_steps,
                    video_prefix=args.video_prefix,
                    episodes=1, seed=1234,
                )

    # ── final eval ──
    if args.eval_episodes > 0:
        with torch.no_grad():
            mean_r, std_r = evaluate_policy_with_kf(
                agent, args.env_id, masked_indices,
                kf_params,
                episodes=args.eval_episodes * 2, seed=4200,
            )
        wandb.log({"eval/final_mean": mean_r, "eval/final_std": std_r}, step=total_steps)
        print(f"Final eval: {mean_r:.1f} ± {std_r:.1f}")

    # ── final summary plots ──
    _log_curves_to_wandb(ep_steps_log, ep_return_log, loss_log, args.agent, total_steps)
    log_kf_comparison(kf, total_steps)
    log_kf_abs_error_plot(kf, total_steps)

    # ── final video ──
    if args.video_every > 0:
        print("Recording final video...")
        with torch.no_grad():
            record_eval_video_with_kf(
                agent, args.env_id, masked_indices, kf_params,
                step=total_steps,
                video_prefix=f"{args.video_prefix}_final",
                episodes=1, seed=9999,
            )

    env.close()
    wandb.finish()
    print("Training complete!")


# ────────────────────── argument parser ──────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RL Training with optional obs masking + Kalman filter")

    # Algorithm
    p.add_argument("--agent", type=str, choices=["ppo", "sac", "td3"], default="td3")
    p.add_argument("--env_id", type=str, default="LunarLanderContinuous-v3")

    # Training
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42000)

    # Common hypers
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)

    # PPO
    p.add_argument("--rollout_steps", type=int, default=4096)
    p.add_argument("--update_epochs", type=int, default=10)
    p.add_argument("--minibatch_size", type=int, default=128)
    p.add_argument("--gae_lambda", type=float, default=0.98)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # SAC / TD3
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--utd_ratio", type=int, default=1)

    # TD3-specific
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--exploration_noise", type=float, default=0.1)

    # Kalman filter — velocities are always masked
    p.add_argument("--kf_dt", type=float, default=0.05,
                   help="Kalman filter time-step (matches MuJoCo/LunarLander default)")
    p.add_argument("--kf_process_noise", type=float, default=0.1,
                   help="Kalman filter process noise std")
    p.add_argument("--kf_measurement_noise", type=float, default=0.01,
                   help="Kalman filter measurement noise std")
    p.add_argument("--kf_log_every", type=int, default=1000,
                   help="Log KF MAE every N steps")
    p.add_argument("--kf_plot_every", type=int, default=50_000,
                   help="Log KF comparison plot every N steps (0=disable)")

    # Logging
    p.add_argument("--log_every", type=int, default=10_000)
    p.add_argument("--eval_every", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=10)

    # Video recording
    p.add_argument("--video_every", type=int, default=100_000,
                   help="Record and upload eval video every N steps (0 to disable)")
    p.add_argument("--video_prefix", type=str, default="eval",
                   help="Filename prefix for recorded videos")

    # wandb
    p.add_argument("--wandb_api_key", type=str, default="",
                   help="Weights & Biases API key.  Also reads WANDB_API_KEY env var.")
    p.add_argument("--wandb_project", type=str, default="18675-Final-Project")
    p.add_argument("--wandb_run_name", type=str, default="")

    args = p.parse_args()

    # ── wandb authentication ──
    api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        print("Logged in to Weights & Biases via API key.")
    else:
        # wandb will fall back to ~/.netrc or interactive login
        print("No --wandb_api_key provided and WANDB_API_KEY not set. "
              "wandb will attempt interactive login or use cached credentials.")

    start = datetime.now()
    run(args)
    print(f"Total time: {datetime.now() - start}")