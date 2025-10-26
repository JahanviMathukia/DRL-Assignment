"""
visualize_ll.py
===============
Visualize a trained LunarLander persona agent playing the game
and show live reward progression in a Matplotlib plot.
"""

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO, DQN


def run_visualization(model_path, algo="ppo", persona="cautious", episodes=3, max_steps=1000):
    algo_cls = PPO if algo.lower() == "ppo" else DQN
    print(f"🚀 Loading {algo.upper()} ({persona}) model from {model_path}")
    model = algo_cls.load(model_path)

    env = gym.make("LunarLander-v2", render_mode="human")
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(f"{algo.upper()} ({persona}) — Live Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward, rewards = 0.0, []

        print(f"\n▶️ Episode {ep}")
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(total_reward)

            if step % 5 == 0:  # refresh plot every few frames
                ax.clear()
                ax.plot(rewards, color="blue")
                ax.set_title(f"{algo.upper()} ({persona}) — Ep {ep}, Reward={total_reward:.1f}")
                ax.set_xlabel("Step")
                ax.set_ylabel("Cumulative Reward")
                plt.pause(0.001)

            time.sleep(0.02)
            if terminated or truncated:
                print(f"🏁 Finished | Steps={step+1}, Total Reward={total_reward:.2f}")
                break

        time.sleep(0.5)

    env.close()
    plt.ioff()
    plt.show()
    print("\n✅ Visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained PPO/DQN LunarLander agent.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--persona", choices=["cautious", "aggressive"], default="cautious")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    run_visualization(args.model_path, args.algo, args.persona, args.episodes, args.max_steps)


# import argparse
# import time
# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# from stable_baselines3 import PPO, DQN


# def run_lunarlander_visualization(model_path, algo="dqn", episodes=5, max_steps=1000):
#     """
#     Visualize a trained PPO/DQN LunarLander agent
#     and plot rewards for all episodes.
#     """
#     algo = algo.lower()
#     if algo not in ["ppo", "dqn"]:
#         raise ValueError("Algorithm must be either 'ppo' or 'dqn'")

#     # Load model
#     algo_cls = PPO if algo == "ppo" else DQN
#     print(f"🚀 Loading {algo.upper()} model from: {model_path}")
#     model = algo_cls.load(model_path)
#     print("✅ Model loaded successfully.\n")

#     # Create environment
#     env = gym.make("LunarLander-v2", render_mode="human")
#     episode_rewards = []

#     for ep in range(1, episodes + 1):
#         obs, _ = env.reset()
#         total_reward = 0.0
#         steps = 0

#         print(f"▶️ Starting Episode {ep}")
#         while True:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)

#             total_reward += reward
#             steps += 1
#             time.sleep(0.02)  # control render speed

#             if terminated or truncated or steps >= max_steps:
#                 episode_rewards.append(total_reward)
#                 print(f"🏁 Episode {ep} finished | Steps={steps} | Total Reward={total_reward:.2f}")
#                 break

#     env.close()

#     # --- Plot rewards per episode ---
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linewidth=2, color="#0080FF")
#     plt.title(f"LunarLander Performance — {algo.upper()}")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.grid(True, linestyle="--", alpha=0.7)
#     plt.show()

#     avg = np.mean(episode_rewards)
#     print(f"\n📊 Average Reward over {episodes} episodes: {avg:.2f}")
#     print("✅ Visualization complete — window closed.\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize trained PPO/DQN LunarLander agent.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.zip)")
#     parser.add_argument("--algo", choices=["ppo", "dqn"], default="dqn", help="Algorithm type")
#     parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
#     parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
#     args = parser.parse_args()

#     run_lunarlander_visualization(
#         model_path=args.model_path,
#         algo=args.algo,
#         episodes=args.episodes,
#         max_steps=args.max_steps
#     )