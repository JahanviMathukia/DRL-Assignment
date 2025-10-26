"""
eval_ll.py
==========
Evaluate trained LunarLander persona models (PPO or DQN)
and export episode-level metrics to CSV under logs/lunarlander/eval/.

Metrics recorded:
-----------------
episode, total_reward, steps, land_success, crash, avg_speed, avg_angle
"""

import os
import csv
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, DQN
from envs.lunarlander.lunarlander_env import LunarLanderPersonaEnv


def evaluate_model(model_path, algo, persona, episodes=10, max_steps=1000, seed=42):
    """Runs evaluation for given model and returns list of episode metrics."""
    algo_cls = PPO if algo.lower() == "ppo" else DQN
    print(f"📂 Loading {algo.upper()} ({persona}) model from {model_path}")
    model = algo_cls.load(model_path)

    env = LunarLanderPersonaEnv(reward_mode=persona, max_steps=max_steps)
    results = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        total_reward, steps = 0.0, 0
        avg_speed, avg_angle = [], []
        success, crash = 0, 0

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Extract physical state values
            x, y, x_dot, y_dot, theta, theta_dot, left_contact, right_contact = obs
            total_reward += reward
            avg_speed.append(np.sqrt(x_dot ** 2 + y_dot ** 2))
            avg_angle.append(abs(theta))
            steps += 1

            if terminated or truncated:
                if left_contact or right_contact:
                    success = 1  # successful landing
                elif y > 0.1:
                    crash = 1  # crash in mid-air
                break

        results.append({
            "episode": ep,
            "total_reward": round(total_reward, 2),
            "steps": steps,
            "land_success": success,
            "crash": crash,
            "avg_speed": round(np.mean(avg_speed), 4),
            "avg_angle": round(np.mean(avg_angle), 4),
        })
        print(f"🏁 Ep {ep} | Reward={total_reward:.2f} | Steps={steps} | Success={success}")

    env.close()
    return results


def save_results_csv(results, algo, persona, seed):
    """Save evaluation results to a timestamped CSV file."""
    os.makedirs("logs/lunarlander/eval", exist_ok=True)
    csv_path = f"logs/lunarlander/eval/{algo}_{persona}_seed{seed}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LunarLander models.")
    parser.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--persona", choices=["cautious", "aggressive"], required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--modeldir", default="models/lunarlander")
    args = parser.parse_args()

    # Models now stored directly under models/lunarlander/
    model_path = os.path.join(args.modeldir, f"{args.algo}_{args.persona}_seed{args.seed}.zip")

    results = evaluate_model(
        model_path=model_path,
        algo=args.algo,
        persona=args.persona,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    save_results_csv(results, args.algo, args.persona, args.seed)
