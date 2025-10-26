"""
train_ll.py
===========
Training script for persona-based LunarLander-v2 agents using PPO or DQN.

Features:
----------
- Supports two personas: 'cautious' and 'aggressive'.
- Automatically saves models and configs under:
    models/lunarlander/seed<seed>/
- Logs training progress to:
    logs/lunarlander/tensorboard/
"""

import os
import json
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from envs.lunarlander.lunarlander_env import LunarLanderPersonaEnv

ALGO_MAP = {"dqn": DQN, "ppo": PPO}
PERSONAS = ["cautious", "aggressive"]

def make_env_fn(persona, max_steps, seed):
    """Single-env factory."""
    def _thunk():
        env = LunarLanderPersonaEnv(reward_mode=persona, max_steps=max_steps)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk

def main():
    parser = argparse.ArgumentParser(description="Train PPO/DQN on LunarLander with persona shaping.")
    parser.add_argument("--algo", choices=ALGO_MAP.keys(), default="ppo")
    parser.add_argument("--persona", choices=PERSONAS, default="cautious")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--timesteps", type=float, default=8e5)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--logdir", default="logs/lunarlander")
    parser.add_argument("--modeldir", default="models/lunarlander")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    set_random_seed(args.seed)

    # === Environment setup ===
    if args.algo == "ppo":
        env = make_vec_env(make_env_fn(args.persona, args.max_steps, args.seed), n_envs=args.num_envs)
    else:
        env = make_env_fn(args.persona, args.max_steps, args.seed)()

    # === Hyperparameter ===
    if args.algo == "dqn":
        lr = 1e-3 if args.persona == "aggressive" else 7e-4
        gamma = 0.99 if args.persona == "cautious" else 0.985
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            buffer_size=120000,
            learning_starts=5000,
            batch_size=64,
            gamma=gamma,
            train_freq=4,
            target_update_interval=800,
            exploration_fraction=0.25,
            exploration_final_eps=0.03,
            tensorboard_log=os.path.join(args.logdir, "tensorboard"),
        )
    else: # PPO
        lr = 3e-4 if args.persona == "cautious" else 5e-4
        gamma = 0.99 if args.persona == "cautious" else 0.985
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr,
            gamma=gamma,
            n_steps=2048,
            batch_size=64,
            clip_range=0.2,
            ent_coef=0.005 if args.persona == "cautious" else 0.02,
            tensorboard_log=os.path.join(args.logdir, "tensorboard"),
        )

    run_name = f"{args.algo}_{args.persona}_seed{args.seed}"
    print(f"\n🚀 Training {args.algo.upper()} ({args.persona}) for {args.timesteps} steps…\n")
    model.learn(total_timesteps=int(args.timesteps), tb_log_name=run_name)

    save_path = os.path.join(args.modeldir, run_name)
    model.save(save_path)
    with open(save_path + "_cfg.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n✅ Saved model to {save_path}\n")

if __name__ == "__main__":
    main()