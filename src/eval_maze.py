# src/eval_maze.py
import os
import csv
import time
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.miniworld.maze_env import MiniWorldPersonaEnv

ALGO_MAP = {"ppo": PPO, "a2c": A2C}

def evaluate_model(model, env, n_episodes=10, max_steps=300):
    """Evaluate a trained model for n episodes and return summary statistics."""
    rewards, steps, successes = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done, step = 0.0, False, 0

        while not done and step < max_steps:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        # success = reached goal (curr_dist < 0.5 at done)
        agent_pos = np.array(env.unwrapped.agent.pos)

        # Handle missing or None goal position
        goal_pos = getattr(env, "goal_pos", None)
        if goal_pos is None:
            # Fallback to center or safe position if no goal defined
            goal_pos = np.array([0.0, 0.0, 0.0])
        else:
            goal_pos = np.array(goal_pos)

        # Compute 2D distance in XZ-plane
        dist = np.linalg.norm(agent_pos[[0, 2]] - goal_pos[[0, 2]])
        success = dist < 0.5

        rewards.append(total_reward)
        steps.append(step)
        successes.append(int(success))

        print(f"Episode {ep+1:02d} | Steps: {step:3d} | "
              f"Total Reward: {total_reward:7.2f} | Success: {success}")

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps),
        "success_rate": np.mean(successes),
    }


def save_csv(results, csv_path):
    """Append or create CSV with evaluation results."""
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c"], required=True)
    parser.add_argument("--persona", choices=["explorer", "hunter"], required=True)
    parser.add_argument("--model_dir", type=str, default="models/miniworld")
    parser.add_argument("--env_id", default="MiniWorld-Maze-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--csv_out", default="logs/miniworld/eval_maze_results.csv")
    args = parser.parse_args()

    # Construct model path automatically
    model_filename = f"{args.algo}_miniworld_{args.persona}_seed7.zip"
    model_path = os.path.join(args.model_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print(f"🧠 Evaluating {args.algo.upper()} ({args.persona}) model from {model_path}")
    env = MiniWorldPersonaEnv(env_id=args.env_id, persona=args.persona, max_steps=args.max_steps)
    model = ALGO_MAP[args.algo].load(model_path)

    stats = evaluate_model(model, env, args.episodes, args.max_steps)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algo": args.algo,
        "persona": args.persona,
        "episodes": args.episodes,
        "mean_reward": round(stats["mean_reward"], 3),
        "std_reward": round(stats["std_reward"], 3),
        "mean_steps": round(stats["mean_steps"], 2),
        "success_rate": round(stats["success_rate"], 2),
        "model_path": model_path,
    }

    save_csv(results, args.csv_out)
    print(f"✅ Saved results to {args.csv_out}")
    print(f"📊 mean_reward={results['mean_reward']}, "
          f"success_rate={results['success_rate']*100:.1f}%")

if __name__ == "__main__":
    main()
