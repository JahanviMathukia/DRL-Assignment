"""
Evaluation script for CartPole RL agents.
Runs trained PPO/A2C models for multiple episodes and exports metrics to CSV.
"""

import os
import csv
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.cartpole.cartpole_env import CartPolePersonaEnv

def evaluate_cartpole(model_path, algo, persona, episodes=50, max_steps=500, csv_out=None):
    Algo = PPO if algo.lower() == "ppo" else A2C
    model = Algo.load(model_path)
    print(f"✅ Loaded {algo.upper()} model from {model_path}")

    env = CartPolePersonaEnv(reward_mode=persona, max_steps=max_steps, difficulty="progressive")

    results = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward, steps, unique_states = 0, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            unique_states = info["unique_states"]
        results.append({
            "episode": ep,
            "steps": steps,
            "reward": total_reward,
            "unique_states": unique_states,
            "final_angle": info["angle"]
        })
        print(f"Episode {ep:03d} | Steps={steps:3d} | Reward={total_reward:8.3f} | Unique={unique_states}")

    # write CSV
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # summary stats
    rewards = [r["reward"] for r in results]
    steps_list = [r["steps"] for r in results]
    print("\n----- Summary -----")
    print(f"Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg Steps:  {np.mean(steps_list):.1f}")
    print(f"Saved results → {csv_out}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CartPole models.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--persona", choices=["solver", "fuzzer"], default="solver")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--csv_out", type=str, default=None)
    args = parser.parse_args()

    if not args.csv_out:
        run_name = f"{args.algo}_cartpole_{args.persona}_seed7.csv"
        args.csv_out = os.path.join("logs", "cartpole", "eval", run_name)

    evaluate_cartpole(
        model_path=args.model_path,
        algo=args.algo,
        persona=args.persona,
        episodes=args.episodes,
        max_steps=args.max_steps,
        csv_out=args.csv_out,
    )
