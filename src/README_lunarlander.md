## Lunar Lander
## Training 
```bash
python -m src.train_ll --algo ppo --persona cautious --seed 10 --timesteps 8e5
python -m src.train_ll --algo dqn --persona cautious --seed 10 --timesteps 8e5
python -m src.train_ll --algo ppo --persona aggressive --seed 10 --timesteps 8e5
python -m src.train_ll --algo dqn --persona aggressive --seed 10 --timesteps 8e5
```

## Evaluate
```bash
# PPO Cautious
python -m src.eval_ll --algo ppo --persona cautious --seed 10 --episodes 100

# PPO Aggressive
python -m src.eval_ll --algo ppo --persona aggressive --seed 10 --episodes 100

# DQN Cautious
python -m src.eval_ll --algo dqn --persona cautious --seed 10 --episodes 100

# DQN Aggressive
python -m src.eval_ll --algo dqn --persona aggressive --seed 10 --episodes 100
```


## Visualize
```bash
# PPO Cautious
python -m src.visualize_ll --algo ppo --persona cautious --model_path models/lunarlander/ppo_cautious_seed10

# PPO Aggressive
python -m src.visualize_ll --algo ppo --persona aggressive --model_path models/lunarlander/ppo_aggressive_seed10

# DQN Cautious
python -m src.visualize_ll --algo dqn --persona cautious --model_path models/lunarlander/dqn_cautious_seed10.zip

# DQN Aggressive
python -m src.visualize_ll --algo dqn --persona aggressive --model_path models/lunarlander/dqn_aggressive_seed10.zip
```

