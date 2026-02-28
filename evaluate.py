# evaluate.py
import os
import argparse
import time
import random
import glob
import numpy as np
import torch
import gymnasium as gym

from config import DQNConfig, EvaluationConfig, TrainingConfig
from agent import DQNAgent

# -------------------------------------
# Utilities
# --------------------------------------

def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    
# ----------------------------------------
# Evaluation
# ----------------------------------------

eval_config = EvaluationConfig()
train_config = TrainingConfig()
dqn_config = DQNConfig()

def evaluate_agent(
    model_path=eval_config.model_path, 
    num_episodes=eval_config.num_episodes, 
    render=eval_config.render, 
    record_video=eval_config.record_video,
    video_dir=None,
    results_dir=None,
    ):
    """
    Evaluate a trained DQN agent.

    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        record_video: Whether to record evaluation videos
        video_dir: the directory path for the downloaded evaluation videos
        results_dir: directory for results summary saver
    """    
    if record_video:
        # Use specific video_dir provided, or fall back to default
        final_video_path = video_dir if video_dir else train_config.video_dir
        os.makedirs(final_video_path, exist_ok=True)
       
        # Create environment 
        env = gym.make(train_config.env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, 
            final_video_path, 
            episode_trigger=lambda x: True
        )
    elif render:
        env = gym.make(train_config.env_name, render_mode="human")
    else:
        env = gym.make(train_config.env_name)

    # Initialize and load agent
    set_seed(train_config.seed, env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent using same DQNConfig as training
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=dqn_config)
    agent.load(model_path)
    
    # Disable exploration completely
    agent.epsilon = 0.0
    agent.policy_net.eval()
    
    print(f"\nLoaded model from: {model_path}")
    print(f"Evaluation epsilon: {agent.epsilon}")
    print("=" * 60)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0  # Episodes with reward > 200

    print("\n" + "=" * 60)
    print(f"Evaluating agent for {num_episodes} episodes")
    print("=" * 60)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            state = next_state

            if render and not record_video:
                time.sleep(0.01)  # Slow down for visualization

            if done:
                landed = next_state[6] == 1 and next_state[7] == 1
                status = "SUCCESSFUL LANDING" if landed else "FAILED/TIMED OUT"
                print(f"Terminal Status: {status}")
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if total_reward >= 200:
            success_count += 1

        print(
            f"Episode {episode}/{num_episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps}"
        )

    env.close()

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = (success_count / num_episodes) * 100

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Mean Episode Length: {mean_length:.2f} steps")
    print(f"Success Rate (≥200): {success_rate:.1f}% ({success_count}/{num_episodes})")
    print("=" * 60)

    results =  {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "episode_rewards": episode_rewards,
    }
    
    # Save summary text file
    if results_dir:
        summary_path = os.path.join(results_dir, "evaluation_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Evaluation of model: {model_path}\n")
            f.write(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            f.write(f"Date: {time.ctime()}\n")
        print(f"📊 Summary saved to: {summary_path}")
    
    return results

# Helper function to find the most recent folder
def get_latest_model_path(base_dir="./models", algorithm="dqn"):
    """
    Finds the most recent timestamped folder for a given algorithm
    and returns the path to the best_model.pth inside it.
    """
    # Search for folders starting with the algorithm name (e.g., dqn_*)
    search_pattern = os.path.join(base_dir, f"{algorithm}_*")
    folders = glob.glob(search_pattern)
    if not folders:
        return None
    
    # Sort by creation time (latest first)
    latest_folder = max(folders, key=os.path.getctime)
    return os.path.join(latest_folder, "best_model.pth")

def record_first_success(model_path, video_dir=None, max_attempts: int=50):
    """
    Run episodes until first success (reward >= 200),
    then re-run that episode with video recording enabled.
    """
    print("\nSearching for a successful landing...")
    
    env = gym.make(train_config.env_name)
    set_seed(train_config.seed, env=env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=dqn_config)
    agent.load(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    
    successful_seed = None
    
    for attempt in range(1, max_attempts + 1):
        # Add attempt to seed so each loop is unique
        current_seed = train_config.seed + attempt
        set_seed(current_seed, env)
        
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            
            if done:
                break
            
        print(f"Attempt {attempt}: Reward = {total_reward:.2f}")
        
        if total_reward >= 200:
            successful_seed = current_seed # Save the actual seed used
            print("✅ Success found! Recording clean episode...")
            break
        
    env.close()
    
    if successful_seed is None:
        print("❌ No successful episode found.")
        return
    
    final_video_path = video_dir if video_dir else train_config.video_dir
    os.makedirs(final_video_path, exist_ok=True)
    
    # Record clean version
    env = gym.make(train_config.env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=final_video_path,
        episode_trigger=lambda x: True,
    )
    
    set_seed(train_config.seed + successful_seed, env)
    
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        state = next_state
        
        if done:
            break
        
    env.close()
    
    print(f"🎥 Video saved in {train_config.video_dir}")
    print(f"Final recorded reward: {total_reward:.2f}")

def test_single_episode(model_path=eval_config.model_path, render=True):
    """Run a single episode with visualization."""
    print("Running single test episode...")
    return evaluate_agent(model_path, num_episodes=1, render=render)

# --------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate DQN/ DDQN agent on LunarLander")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "double_dqn", "double-dqn"])
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--record", action="store_true", help="Record evaluation videos")
    parser.add_argument("--record-success", action="store_true", 
                        help="Record first successful landing (reward >= 200)")
    parser.add_argument("--test", action="store_true", help="Run single test episode with rendering")

    args = parser.parse_args()

    # Determine which model path to use
    model_to_use = args.model
    
    if model_to_use is None:
        # Ensure dashes are converted to underscores to match the folder naming convention 
        algo_name = args.algo.replace("-", "_")
        # Try to find the latest automatically based on algorithm type
        model_to_use = get_latest_model_path(algorithm=algo_name)
        
    if model_to_use is None or not os.path.exists(model_to_use):
        print(f"❌ Error: Could not find model at {model_to_use}")
        print("Please specify path with --model path/to/model.pth")
    else:
        # Create a dynamic video path
        # Get the timestamped folder name
        model_run_folder = os.path.basename(os.path.dirname(model_to_use))
        
        # Point to the existing model folder to save the .txt summary there
        results_dir = os.path.dirname(model_to_use)
        
        # Point to a specific video subfolder
        specific_video_dir = os.path.join(train_config.video_dir, model_run_folder)
        
        # Proceed with evaluation using model_to_use (pass specific_video_dir to functions)
        if args.record_success:
            record_first_success(model_to_use, video_dir=specific_video_dir)
        elif args.test:
            test_single_episode(model_to_use, render=True)
        else:
            evaluate_agent(model_to_use, args.episodes, args.render, args.record, video_dir=specific_video_dir, results_dir=results_dir)

# CLI commands to run
# Testing : python evaluate.py --test
# Evaluation: python evaluate.py --episodes 100 / python evaluate.py --model models/best_model.pth --episodes 100
# Record video: python evaluate.py --episodes 3 --record
# Record successful landing video: python evaluate.py --record-success
# Test most recent Double DQN: python evaluate.py --algo double_dqn --test
# To test specific old run from last week: python evaluate.py --model models/dqn_20260220_1000/best_model.pth