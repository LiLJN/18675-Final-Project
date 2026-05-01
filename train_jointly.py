from src.train import train_joint

def main():
    agent, rewards, eval_rewards = train_joint(
        num_episodes=500,
        noise_std=0.0,       # set to 0.1 for noisy variant
    )

    # Print final learned EKF parameters
    print("\n" + "=" * 50)
    print("Final learned EKF parameters:")
    print(f"Q (process noise):\n{agent.ekf.Q.detach().cpu().numpy()}")
    print(f"R (measurement noise):\n{agent.ekf.R.detach().cpu().numpy()}")

if __name__ == "__main__":
    main()
