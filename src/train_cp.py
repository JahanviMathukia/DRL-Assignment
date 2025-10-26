"""
Training script for CartPole persona RL agents using Stable Baselines3.
Algorithms: PPO, A2C
Personas: solver (stability-focused), fuzzer (exploration-focused)

Artifacts created:
  • models/cartpole/{algo}_cartpole_{persona}_seed{seed}.zip
  • models/cartpole/{algo}_cartpole_{persona}_seed{seed}_cfg.json
  • logs/cartpole/tensorboard/{algo}_cartpole_{persona}_seed{seed}/events.out.tfevents...
"""

import os
import json
import argparse
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import VecTransposeImage
from gymnasium import spaces as gym_spaces
from envs.cartpole.cartpole_env import CartPolePersonaEnv

ALGO_MAP = {"ppo": PPO, "a2c": A2C}

# Environment factory (returns a function that builds one monitored env)
def make_cartpole_env_fn(persona: str, max_steps: int, seed: int):
    def _thunk():
        env = CartPolePersonaEnv(reward_mode=persona, max_steps=max_steps)

        # Ensure gym → gymnasium space compatibility (for SB3)
        def _to_gymnasium_space(space):
            if hasattr(space, "n"):  # Discrete
                return gym_spaces.Discrete(space.n)
            elif hasattr(space, "nvec"):  # MultiDiscrete
                return gym_spaces.MultiDiscrete(space.nvec)
            elif hasattr(space, "low"):  # Box
                return gym_spaces.Box(
                    low=space.low,
                    high=space.high,
                    shape=space.shape,
                    dtype=space.dtype,
                )
            return space

        env.action_space = _to_gymnasium_space(env.action_space)
        env.observation_space = _to_gymnasium_space(env.observation_space)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk

# Main training entrypoint
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGO_MAP.keys(), default="ppo")
    parser.add_argument("--persona", choices=["solver", "fuzzer"], default="solver")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=float, default=1e5)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=500)
    # common hyperparams
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    # PPO-specific
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    # A2C-specific
    parser.add_argument("--a2c_n_steps", type=int, default=5)
    parser.add_argument("--a2c_ent_coef", type=float, default=0.01)
    parser.add_argument("--a2c_vf_coef", type=float, default=0.5)

    args = parser.parse_args()

    # ---------------- Directory setup ----------------
    base_model_dir = os.path.join("models", "cartpole")
    base_log_dir = os.path.join("logs", "cartpole", "tensorboard")
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)

    run_name = f"{args.algo}_cartpole_{args.persona}_seed{args.seed}"
    model_path = os.path.join(base_model_dir, run_name + ".zip")
    cfg_path = os.path.join(base_model_dir, run_name + "_cfg.json")
    tb_dir = os.path.join(base_log_dir, run_name)

    # ---------------- Reproducibility ----------------
    set_random_seed(args.seed)

    # ---------------- Vectorized environments ----------------
    env_fns = [make_cartpole_env_fn(args.persona, args.max_steps, args.seed + i)
               for i in range(args.num_envs)]
    vec_env = make_vec_env(env_fns[0], n_envs=args.num_envs, seed=args.seed)

    # Use CNN policy only if observation space is image-like
    if is_image_space(vec_env.observation_space):
        vec_env = VecTransposeImage(vec_env)
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"

    # ---------------- Algorithm setup ----------------
    Algo = ALGO_MAP[args.algo]
    if args.algo == "ppo":
        model = Algo(
            policy,
            vec_env,
            verbose=1,
            tensorboard_log=base_log_dir,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
        )
    else:  # A2C
        model = Algo(
            policy,
            vec_env,
            verbose=1,
            tensorboard_log=base_log_dir,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=args.a2c_n_steps,
            ent_coef=args.a2c_ent_coef,
            vf_coef=args.a2c_vf_coef,
        )

    # ---------------- Training ----------------
    total_timesteps = int(args.timesteps)
    print(f"\n[INFO] Training {run_name} for {total_timesteps} timesteps ...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)

    # ---------------- Save model + config ----------------
    model.save(model_path)
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[DONE] Saved model → {model_path}")
    print(f"[DONE] TensorBoard logs → {tb_dir}")
    print(f"[DONE] Config JSON → {cfg_path}")

if __name__ == "__main__":
    main()