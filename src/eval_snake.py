# src/eval_snake.py
import os
import csv
import time
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.snake.snake_env import SnakeEnv

"""
Evaluate a trained Snake RL agent over multiple episodes
and save the performance metrics to a CSV file.

Metrics recorded:
-----------------
- episode number
- total reward
- final score (# of apples eaten)
- total steps survived
- mean reward per step
"""

ALGO_MAP = {"ppo": PPO, "a2c": A2C}

def evaluate_model(algo, persona, grid_size, episodes, seed, model_dir, csv_out):
    """Run multiple episodes and save evaluation metrics."""
    model_path = os.path.join(model_dir, f"{algo}_snake_{persona}_seed{seed}.zip")
    print(f"Loading model from: {model_path}")

    # Load model and create environment
    env = SnakeEnv(grid_size=grid_size, reward_mode=persona, render_mode=None, seed=seed)
    ModelClass = ALGO_MAP[algo]
    model = ModelClass.load(model_path, env=env)

    all_rewards, all_scores, all_steps = [], [], []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(int(action))
            total_reward += reward
            steps += 1

        all_rewards.append(total_reward)
        all_scores.append(info["score"])
        all_steps.append(steps)

        print(f"Episode {ep:>3}: Reward={total_reward:>7.2f} | "
              f"Score={info['score']:>2} | Steps={steps:>4}")

    # Aggregate results
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_score = np.mean(all_scores)
    mean_steps = np.mean(all_steps)

    print("\nEvaluation Summary:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Score : {mean_score:.2f}")
    print(f"Mean Steps : {mean_steps:.2f}")

    # Save to CSV
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "score", "steps", "mean_reward_per_step"
        ])
        for i in range(episodes):
            mean_per_step = all_rewards[i] / all_steps[i] if all_steps[i] > 0 else 0
            writer.writerow([
                i + 1,
                algo,
                persona,
                round(all_rewards[i], 3),
                all_scores[i],
                all_steps[i],
                round(mean_per_step, 4),
            ])

    print(f"\nSaved evaluation results to: {csv_out}")
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c"], required=True, help="Algorithm to evaluate")
    parser.add_argument("--persona", choices=["survivor", "hunter"], required=True, help="Persona/reward mode")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes to run")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model_dir", type=str, default="models/snake")
    parser.add_argument("--csv_out", type=str, default="logs/snake/eval/snake_eval.csv")
    args = parser.parse_args()

    evaluate_model(
        algo=args.algo,
        persona=args.persona,
        grid_size=args.grid_size,
        episodes=args.episodes,
        seed=args.seed,
        model_dir=args.model_dir,
        csv_out=args.csv_out,
    )

if __name__ == "__main__":
    main()
