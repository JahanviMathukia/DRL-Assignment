import os
import csv
import time
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.miniworld.maze_env import MiniWorldPersonaEnv

ALGO_MAP = {"ppo": PPO, "a2c": A2C}

def evaluate_model(model, env, n_episodes=10, max_steps=300, algo="ppo", persona="explorer", per_episode_csv=None):
    """Evaluate a trained model and optionally save per-episode results."""
    rewards, steps, successes = [], [], []

    if per_episode_csv:
        os.makedirs(os.path.dirname(per_episode_csv), exist_ok=True)
        with open(per_episode_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "algo", "persona", "reward", "score", "steps", "mean_reward_per_step"])

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done, step = 0.0, False, 0

        while not done and step < max_steps:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        # --- Calculate metrics ---
        agent_pos = np.array(env.unwrapped.agent.pos)
        goal_pos = getattr(env, "goal_pos", None)
        if goal_pos is None:
            goal_pos = np.array([0.0, 0.0, 0.0])
        else:
            goal_pos = np.array(goal_pos)

        dist = np.linalg.norm(agent_pos[[0, 2]] - goal_pos[[0, 2]])
        success = dist < 0.5
        mean_rps = total_reward / max(1, step)

        rewards.append(total_reward)
        steps.append(step)
        successes.append(int(success))

        print(f"Episode {ep+1:02d} | Steps: {step:3d} | "
              f"Total Reward: {total_reward:7.2f} | Success: {success}")

        # --- Append per-episode row ---
        if per_episode_csv:
            with open(per_episode_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep + 1,
                    algo,
                    persona,
                    round(total_reward, 3),
                    int(success),
                    step,
                    round(mean_rps, 4),
                ])
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps),
        "success_rate": np.mean(successes),
    }

def save_summary_csv(results, csv_path):
    """Append or create the summary CSV (aggregated results)."""
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
    parser.add_argument("--csv_out", default="logs/miniworld/eval_maze_summary.csv")
    parser.add_argument("--episode_csv_dir", default="logs/miniworld/eval")
    args = parser.parse_args()

    # Build model path
    model_filename = f"{args.algo}_miniworld_{args.persona}_seed7.zip"
    model_path = os.path.join(args.model_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Evaluating {args.algo.upper()} ({args.persona}) model...")
    env = MiniWorldPersonaEnv(env_id=args.env_id, persona=args.persona, max_steps=args.max_steps)
    model = ALGO_MAP[args.algo].load(model_path)

    # Per-episode CSV filename
    per_episode_csv = os.path.join(
        args.episode_csv_dir, f"{args.algo}_miniworld_{args.persona}_eval.csv"
    )

    # Evaluate model
    stats = evaluate_model(
        model,
        env,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        algo=args.algo,
        persona=args.persona,
        per_episode_csv=per_episode_csv,
    )

    # Save summary
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

    save_summary_csv(results, args.csv_out)
    print(f"Summary saved to {args.csv_out}")
    print(f"Per-episode results saved to {per_episode_csv}")
    print(f"mean_reward={results['mean_reward']}, "
          f"success_rate={results['success_rate']*100:.1f}%")

if __name__ == "__main__":
    main()