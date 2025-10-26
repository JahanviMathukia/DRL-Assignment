set -e

# PPO / A2C × solver / fuzzer
python -m src.train_cp --algo ppo --persona solver --seed 7  --timesteps 1e5 --num_envs 4 --max_steps 500 --logdir logs/tb --modeldir models
python -m src.train_cp --algo a2c --persona solver --seed 7  --timesteps 1e5 --num_envs 4 --max_steps 500 --logdir logs/tb --modeldir models
python -m src.train_cp --algo ppo --persona fuzzer --seed 7  --timesteps 1e5 --num_envs 4 --max_steps 500 --logdir logs/tb --modeldir models
python -m src.train_cp --algo a2c --persona fuzzer --seed 7  --timesteps 1e5 --num_envs 4 --max_steps 500 --logdir logs/tb --modeldir models
