"""
Visualize CartPole agents live and optionally record episodes to MP4/GIF.
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from envs.cartpole.cartpole_env import CartPolePersonaEnv
from gymnasium.wrappers import RecordVideo


def visualize_cartpole(model_path, algo="ppo", persona="solver",
                       episodes=3, difficulty="progressive",
                       max_steps=500, record=False, outdir="media"):
    Algo = PPO if algo.lower() == "ppo" else A2C
    model = Algo.load(model_path)
    print(f"🎮 Loaded {algo.upper()} model from {model_path}")

    # optionally record videos
    env = CartPolePersonaEnv(reward_mode=persona, render_mode="human", max_steps=max_steps, difficulty=difficulty)
    if record:
        os.makedirs(outdir, exist_ok=True)
        env = RecordVideo(env, video_folder=outdir,
                          episode_trigger=lambda e: True,
                          name_prefix=f"{algo}_{persona}_cartpole")

    # setup live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    line, = ax.plot([], [], lw=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{algo.upper()} ({persona}) — {difficulty}")
    ax.grid(True)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        rewards, total = [], 0
        print(f"▶️ Episode {ep}")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            rewards.append(total)
            # live plot
            line.set_xdata(np.arange(len(rewards)))
            line.set_ydata(rewards)
            ax.relim(); ax.autoscale_view()
            plt.pause(0.001)
            time.sleep(0.02)
            if term or trunc:
                print(f"🏁 Episode {ep} done | Steps={len(rewards)} | Reward={total:.2f}")
                break
        time.sleep(0.4)

    env.close()
    plt.ioff()
    plt.show()
    print("✅ Visualization complete.")
    if record:
        print(f"🎞️ Videos saved to {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--persona", choices=["solver", "fuzzer"], default="solver")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", choices=["none", "progressive", "randomize_on_reset"], default="progressive")
    parser.add_argument("--record", action="store_true", help="Record videos to media/")
    args = parser.parse_args()

    visualize_cartpole(
        model_path=args.model_path,
        algo=args.algo,
        persona=args.persona,
        episodes=args.episodes,
        difficulty=args.difficulty,
        record=args.record,
    )