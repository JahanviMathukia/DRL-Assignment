# 🧠 DRL for Automated Testing — CartPole 

**Author:** Jahanvi Mathukiya  

---

## 🎯 Overview
This repository implements a **Deep Reinforcement Learning (DRL)** framework for **automated application/game testing**, following the assignment’s goals:

1. **Automate testing** of two non-trivial environments using trained DRL agents (not scripted bots).  
2. **Compare algorithms (PPO vs. A2C)** across distinct **reward-based personas** (“solver” vs. “fuzzer”).  
3. **Collect domain-specific metrics** such as reward, steps survived, unique states visited, and crash rates.  
4. Provide full **reproducibility** via scripts, saved artifacts, fixed seeds, and pinned dependencies.  

Environments  
- 🧩 **CartPole-v1** (custom Gymnasium wrapper with persona rewards)  
- 🎮 **Sokoban** (wrapper available under `envs/sokoban`, optional second app)

---

## 🧱 Project Structure
```bash
.
├── envs/
│ └── cartpole/
│ └── cartpole_env.py # Persona environment (solver / fuzzer)
├── src/
│ ├── train_cp.py # Train PPO / A2C on CartPole
│ ├── eval_cp.py # Evaluate trained agents → CSV
│ └── visualize_cp.py # Live visualization + reward plot
├── models/
│ └── cartpole/ # Saved models (.zip + _cfg.json)
├── logs/
│ └── cartpole/
│ ├── tensorboard/ # TB logs for each run
│ └── eval/ # Per-episode metrics CSV
├── notebooks/
│ └── cartpole_analysis.ipynb # Analysis + comparison plots
├── plots/
│ └── cartpole/ # Auto-generated figures + summary CSV
└── README.md
```

# CARTPOLE
```bash
python -m src.train_cp --algo ppo --persona solver --seed 7 --timesteps 1e5 --num_envs 4 --max_steps 500


python -m src.train_cp --algo ppo --persona fuzzer --seed 7 --timesteps 1e5 --num_envs 4 --max_steps 500

python -m src.train_cp --algo a2c --persona solver --seed 7 --timesteps 1e5 --num_envs 4 --max_steps 500

python -m src.train_cp --algo a2c --persona fuzzer --seed 7 --timesteps 1e5 --num_envs 4 --max_steps 500
```
# Evaluate
```bash
python -m src.eval_cp --model_path models/cartpole/ppo_cartpole_solver_seed7.zip --algo ppo --persona solver --episodes 100 --csv_out logs/cartpole/eval/ppo_cartpole_solver_seed7.csv

python -m src.eval_cp --model_path models/cartpole/ppo_cartpole_fuzzer_seed7.zip --algo ppo --persona fuzzer --episodes 100 --csv_out logs/cartpole/eval/ppo_cartpole_fuzzer_seed7.csv

python -m src.eval_cp --model_path models/cartpole/a2c_cartpole_solver_seed7.zip --algo a2c --persona solver --episodes 100 --csv_out logs/cartpole/eval/a2c_cartpole_solver_seed7.csv

python -m src.eval_cp --model_path models/cartpole/a2c_cartpole_fuzzer_seed7.zip --algo a2c --persona fuzzer --episodes 100 --csv_out logs/cartpole/eval/a2c_cartpole_fuzzer_seed7.csv
```
# Visualize
```bash
python -m src.visualize_cp --model_path models/cartpole/ppo_cartpole_solver_seed7.zip --algo ppo --persona solver --episodes 5 --difficulty progressive

python -m src.visualize_cp --model_path models/cartpole/ppo_cartpole_fuzzer_seed7.zip --algo ppo --persona fuzzer --episodes 5 --difficulty progressive

python -m src.visualize_cp --model_path models/cartpole/a2c_cartpole_solver_seed7.zip --algo a2c --persona solver --episodes 5 --difficulty progressive

python -m src.visualize_cp --model_path models/cartpole/a2c_cartpole_fuzzer_seed7.zip --algo a2c --persona fuzzer --episodes 5 --difficulty progressive
```