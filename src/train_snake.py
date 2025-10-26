# src/train_snake.py
import os
import json
import argparse
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from envs.snake.snake_env import SnakeEnv

"""
Train PPO or A2C agents on the custom Snake environment
for two distinct personas: 'survivor' and 'hunter'.

Features:
---------
- Models saved under: models/snake/<algo>_<persona>_seed<seed>.zip
- TensorBoard logs under: logs/snake/tensorboard/<algo>_<persona>_seed<seed>/
- Reproducible seeds
- Multiple parallel environments for stable training
"""

ALGO_MAP = {"ppo": PPO, "a2c": A2C}

def make_env(persona, grid_size, seed):
    """Factory for creating a new Snake environment instance."""
    def _init():
        env = SnakeEnv(grid_size=grid_size, reward_mode=persona, render_mode=None, seed=seed)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--persona", type=str, choices=["survivor", "hunter"], default="survivor")
    parser.add_argument("--timesteps", type=float, default=5e5)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--base_logdir", type=str, default="logs/snake/tensorboard")
    parser.add_argument("--base_modeldir", type=str, default="models/snake")
    args = parser.parse_args()

    # Construct unique run name and directories
    run_name = f"{args.algo}_snake_{args.persona}_seed{args.seed}"
    logdir = os.path.join(args.base_logdir, run_name)
    model_path = os.path.join(args.base_modeldir, f"{run_name}.zip")
    cfg_path = os.path.join(args.base_modeldir, run_name + "_cfg.json")

    # Create directories
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Create parallel environments
    env_fns = [make_env(args.persona, args.grid_size, args.seed + i) for i in range(args.num_envs)]
    env = DummyVecEnv(env_fns)

    Algo = ALGO_MAP[args.algo]

    # Initialize algorithm
    common_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=logdir,
        seed=args.seed,
    )

    if args.algo == "ppo":
        model = PPO(
            **common_kwargs,
            n_steps=1024,
            batch_size=256,
        )
    else:  # A2C
        model = A2C(
            **common_kwargs,
            n_steps=5 * args.num_envs,
        )

    print(f"Training {run_name} for {int(args.timesteps):,} timesteps...")
    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    # Save model
    model.save(model_path)
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved model to: {model_path}")

    env.close()

if __name__ == "__main__":
    main()
