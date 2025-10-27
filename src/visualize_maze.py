# src/visualize_maze_cv2.py
import os
import cv2
import time
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.miniworld.maze_env import MiniWorldPersonaEnv

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MINIWORLD_NO_TEXT_LABEL"] = "1"

ALGO = {"ppo": PPO, "a2c": A2C}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--persona", choices=["explorer", "hunter"], default="explorer")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--env_id", default="MiniWorld-Maze-v0")
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    env = MiniWorldPersonaEnv(
        env_id=args.env_id,
        persona=args.persona,
        max_steps=args.max_steps,
        render_mode="rgb_array",
    )

    model = ALGO[args.algo].load(args.model_path)
    obs, _ = env.reset()

    print(f"🎮 Playing {args.env_id} as persona='{args.persona}' using {args.algo.upper()}")

    done = False
    total_r = 0
    step = 0
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        step += 1
        done = terminated or truncated

        # Render frame (as RGB array)
        frame = env.render()
        if frame is not None and frame.size > 0:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("MiniWorld Maze (cv2)", frame_bgr)

        # Handle keyboard + GUI events every frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or done:
            break

        time.sleep(0.01) 

    print(f"✅ Finished episode — Steps: {step}, Total Reward: {total_r:.2f}")
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
