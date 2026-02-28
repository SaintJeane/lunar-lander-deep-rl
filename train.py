# train.py
import random
import argparse
import os
import json
from datetime import datetime

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import DQNConfig, TrainingConfig
from agent import DQNAgent

# ---------------------------------------------
# Utilities
# ---------------------------------------------


# Seed control (for reproducibility of results)
def set_seed(seed, env):
    """Seed control"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


# argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train agent on LunarLander")

    # Core Hyperparameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--batch_size", type=int)

    # Double DQN toggle
    parser.add_argument("--double_dqn", "--double-dqn", action="store_true")

    # Soft update toggle
    parser.add_argument("--soft_update", action="store_true")
    parser.add_argument("--tau", type=float)

    # Training
    parser.add_argument("--episodes", type=int)

    return parser.parse_args()


# ----------------------------------------------------
# Training
# ----------------------------------------------------


def train_dqn(args):
    """
    Train agent on LunarLander environment.

    Args:
        args: parsed arguments used in training the agent
        max_steps: Maximum steps per episode
        save_dir: Directory to save model checkpoints
        plot_dir: Directory to save training plots
        log_interval: Episodes between progress logs
    """
    # Load configs
    train_config = TrainingConfig()
    dqn_config = DQNConfig()

    # Override config with CLI args if provided
    if args.lr:
        dqn_config.learning_rate = args.lr
    if args.gamma:
        dqn_config.gamma = args.gamma
    if args.batch_size:
        dqn_config.batch_size = args.batch_size
    if args.episodes:
        train_config.num_episodes = args.episodes

    # Determine algorithm name
    algorithm_name = "double_dqn" if args.double_dqn else "dqn"

    # Create a unique timestamp (YearMonthDay_HourMinute)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{algorithm_name}_{timestamp}"

    # Directories
    # os.makedirs(train_config.save_dir, exist_ok=True)
    # os.makedirs(train_config.plot_dir, exist_ok=True)
    save_dir = os.path.join(train_config.save_dir, run_name)
    plot_dir = os.path.join(train_config.plot_dir, run_name)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Create environment
    env = gym.make(train_config.env_name)
    set_seed(train_config.seed, env=env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent using config injection
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=dqn_config,
    )

    # Optional toggles
    agent.use_double_dqn = args.double_dqn
    agent.use_soft_update = args.soft_update
    if args.tau:
        agent.tau = args.tau

    # Training metrics
    episode_rewards = []
    episode_losses = []
    moving_avg_rewards = []
    best_avg_reward = -float("inf")

    print("=" * 60)
    print(f"Starting DQN Training on {train_config.env_name}")
    print("=" * 60)
    print(f"Episodes: {train_config.num_episodes}")
    print(f"Max Steps per Episode: {train_config.max_steps}")
    print(f"Learning Rate: {dqn_config.learning_rate}")
    print(f"Initial Epsilon: {agent.epsilon}")
    print(f"Batch Size: {agent.batch_size}")
    print(f"Gamma: {agent.gamma}")
    print("=" * 60)

    for episode in range(1, train_config.num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        # episode_reward = 0
        episode_loss = []

        for step in range(train_config.max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)

            # episode_reward += reward
            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(total_reward)

        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # Calculate moving average (last 100 episodes)
        window_size = min(100, episode)
        moving_avg = np.mean(episode_rewards[-window_size:])
        moving_avg_rewards.append(moving_avg)

        # Log progress
        if episode % train_config.log_interval == 0:
            print(
                f"Episode {episode} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg Reward (100ep): {moving_avg:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )

        # Save best model (preventing early noise saves)
        if moving_avg > best_avg_reward and episode > 100:
            best_avg_reward = moving_avg
            # agent.save(os.path.join(train_config.save_dir, "best_model.pth"))
            agent.save(os.path.join(save_dir, "best_model.pth"))
            print(f"New best average reward: {best_avg_reward:.2f} - Model saved!")

        # Save checkpoint every 500 episodes
        if episode % 500 == 0:
            # agent.save(os.path.join(train_config.save_dir, f"checkpoint_ep{episode}.pth"))
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pth"))

    # Save final model
    # agent.save(os.path.join(train_config.save_dir, "final_model.pth"))
    agent.save(os.path.join(save_dir, "final_model.pth"))

    # Save training history
    history = {
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "moving_avg_rewards": moving_avg_rewards,
        "best_avg_reward": best_avg_reward,
        "training_date": datetime.now().isoformat(),
        "algorithm": algorithm_name,
    }
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot results
    plot_training_results(
        episode_rewards,
        moving_avg_rewards,
        episode_losses,
        plot_dir,
        algorithm_name,
    )

    env.close()
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Models saved to: {train_config.save_dir}")
    print(f"Plots saved to: {train_config.plot_dir}")
    print("=" * 60)

    return agent, history


def plot_training_results(
    episode_rewards, moving_avg_rewards, episode_losses, plot_dir, algorithm_name
):
    """Generate and save training visualization plots."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot rewards
    axes[0].plot(episode_rewards, alpha=0.3, label="Episode Reward", color="blue")
    axes[0].plot(
        moving_avg_rewards,
        label="Moving Average (100 episodes)",
        color="red",
        linewidth=2,
    )
    axes[0].axhline(
        y=200,
        color="green",
        linestyle="--",
        label="Solved Threshold (200)",
        linewidth=2,
    )
    axes[0].set_xlabel("Episode", fontsize=12)
    axes[0].set_ylabel("Total Reward", fontsize=12)
    axes[0].set_title(
        f"{algorithm_name} Training Progress - {TrainingConfig().env_name}",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot loss
    axes[1].plot(episode_losses, alpha=0.7, color="orange")
    axes[1].set_xlabel("Episode", fontsize=12)
    axes[1].set_ylabel("Average Loss", fontsize=12)
    axes[1].set_title("Training Loss Over Time", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{algorithm_name}_results.png"
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches="tight")
    print(
        f"\nTraining plot saved to: {os.path.join(plot_dir, algorithm_name.replace(' ', '_').lower() + '_training_results.png')}"
    )

    # Create separate reward plot for README
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label="Episode Reward", color="blue")
    plt.plot(
        moving_avg_rewards,
        label="Moving Average (100 episodes)",
        color="red",
        linewidth=2,
    )
    plt.axhline(
        y=200,
        color="green",
        linestyle="--",
        label="Solved Threshold (200)",
        linewidth=2,
    )
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(f"{algorithm_name} Learning Progress", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "reward_curve.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    args = parse_args()
    train_dqn(args)


# python train.py --lr 5e-4 --gamma 0.95 --double_dqn --episodes 1500
# python train.py --lr 5e-4 --gamma 0.95 --double_dqn --episodes 1000
# python train.py --double_dqn/ --double-dqn
