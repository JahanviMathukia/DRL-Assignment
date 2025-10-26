# src/train_maze.py
import os
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from envs.miniworld.maze_env import MiniWorldPersonaEnv

ALGO_MAP = {"ppo": PPO, "a2c": A2C}

# ---------- Wrapper to fix dtype issue ----------
class ImageToFloat32(gym.ObservationWrapper):
    """Convert uint8 image observations to float32 [0,1]."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

# ---------- Environment factory ----------
def make_env(env_id, persona, max_steps, log_dir):
    def _init():
        env = MiniWorldPersonaEnv(env_id=env_id, persona=persona, max_steps=max_steps)
        env = ImageToFloat32(env)
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, os.path.join(log_dir, "monitor.csv"))
        return env
    return _init

# ---------- Main Training ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--persona", choices=["explorer", "hunter"], default="explorer")
    parser.add_argument("--env_id", default="MiniWorld-Maze-v0")
    parser.add_argument("--timesteps", type=float, default=3e5)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tb", action="store_true")
    args = parser.parse_args()

    # Paths
    model_dir = f"models/miniworld"
    tb_dir = f"logs/miniworld/tensorboard"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Environment setup
    env = DummyVecEnv([make_env(args.env_id, args.persona, args.max_steps, tb_dir)])
    env = VecTransposeImage(env)

    Algo = ALGO_MAP[args.algo]

    # Algorithm-specific settings
    if args.algo == "ppo":
        model = Algo(
            "CnnPolicy",
            env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=(tb_dir if args.tb else None),
            learning_rate=3e-4,
            ent_coef=0.01,
            n_steps=1024,
            batch_size=256,
            clip_range=0.2,
        )
    else:  # A2C
        model = Algo(
            "CnnPolicy",
            env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=(tb_dir if args.tb else None),
            learning_rate=7e-4,
            ent_coef=0.01,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=False,
        )

    print(f"🚀 Training {args.algo.upper()} for persona={args.persona}")
    model.learn(total_timesteps=int(args.timesteps))
    model.save(os.path.join(model_dir, f"{args.algo}_miniworld_{args.persona}_seed{args.seed}.zip"))
    print(f"✅ Model saved to {model_dir}")

if __name__ == "__main__":
    main()
