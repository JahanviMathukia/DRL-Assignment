# 🧭 MiniWorld - Maze Environment (Persona-Aware DRL)

This document describes the **MiniWorld-Maze environment** used for automated testing with **Deep Reinforcement Learning (DRL)**.  

---

## Architecture Overview

| Component | Location | Description |
|------------|-----------|--------------|
| **Environment / Game** | `envs/miniworld/maze_env.py` | Wrapper around `MiniWorld-Maze-v0` that adds persona-aware reward logic |
| **Training Script** | `src/train_maze.py` | Trains agents with PPO or A2C using Stable-Baselines3 |
| **Evaluation Script** | `src/eval_maze.py` | Evaluates trained agents, logs metrics to CSV |
| **Visualization Script** | `src/visualize_maze.py` | Runs the trained agent interactively |
| **Trained Models** | `models/miniworld/` | Saved `.zip` models for each persona and algorithm |
| **Evaluation Logs** | `logs/miniworld/eval/` | CSV results of multiple-episode evaluations |
| **Plots & Analysis** | `notebooks/miniworld_analysis.ipynb` | Jupyter notebook for plotting learning curves and metrics |

---

## Environment Description

**MiniWorld-Maze-v0** is a 3D navigation task from the MiniWorld library.  
The agent must navigate through a maze to reach a target goal location while avoiding collisions with walls.

### Wrapper Enhancements
The wrapper `MiniWorldPersonaEnv` adds:
- **Persona-specific reward shaping**
- **Exploration tracking grid**
- **Goal proximity estimation**
- **Collision detection and recovery**
- **Observation change (curiosity) rewards**

---

## Agent Training Setup

### Action Space
Continuous environment wrapped with **discrete movement controls**:
- `0` → turn left  
- `1` → go straight  
- `2` → turn right  

### Observation
The base MiniWorld observation (`RGB array`) is used as input.  
Internally, additional states are computed for rewards:
- Agent position and direction
- Distance to goal
- Collision and novelty checks

### Hyperparameters
- **Algorithms:** PPO, A2C  
- **Timesteps:** 3e5  
- **Max steps per episode:** 300  
- **Seed:** 7  
- **Rendering mode:** `rgb_array` for headless training

---

## 🧭 Persona Overview

| Persona | Goal | Behavior Style | Reward Theme |
|----------|------|----------------|---------------|
| **Explorer** | Explore new areas safely | Curious and slow | Novelty, movement, curiosity |
| **Hunter** | Reach goal efficiently | Focused and fast | Goal distance reduction, accuracy |

---

## 🧮 Reward Design

### 🧍 Explorer Persona — “Curious Wanderer”
Encourages coverage, novelty, and safe exploration.

```python
if persona == "explorer":
    reward += 0.05                # survival reward
    if new_cell: reward += 0.6    # first-time visit bonus
    if moved: reward += 0.04      # movement encouragement
    else: reward -= 0.03          # stillness penalty
    if collided: reward -= 0.2    # mild collision penalty
    else: reward += 0.02          # reward for not colliding
    reward += 0.0005 * frame_diff # curiosity (visual novelty)
    if close_to_goal: reward += 0.5
    if reached_goal: reward += 5.0
    if timed_out_far: reward -= 0.5
```


### Hunter Persona — “Focused Goal-Seeker”

Prioritizes reaching the target quickly and efficiently.
```bash
if persona == "hunter":
    reward += 0.2 * progress      # positive for approaching goal
    if facing_goal: reward += 0.05
    reward -= 0.02                # small time penalty
    if collided: reward -= 1.0
    if reached_goal: reward += 10.0
    if failed_far: reward -= 3.0

```

## Reproducibility Instructions

All experiments use fixed seed = 7 and max steps = 300 per episode.

---

### 🧩 Training Commands
```bash
# PPO - Explorer
python -m src.train_maze --algo ppo --persona explorer --timesteps 200000 --tb

# PPO - Hunter
python -m src.train_maze --algo ppo --persona hunter --timesteps 200000 --tb

# A2C - Explorer
python -m src.train_maze --algo a2c --persona explorer --timesteps 200000 --tb

# A2C - Hunter
python -m src.train_maze --algo a2c --persona hunter --timesteps 300000 --tb
```

### 🧪 Evaluation Commands

Run the trained models and record results for 100 episodes:
```bash
# PPO - Explorer
python -m src.eval_maze --algo ppo --persona explorer --episodes 10 --csv_out logs/miniworld/eval/ppo_miniworld_explorer_eval.csv

# PPO - Hunter
python -m src.eval_maze --algo ppo --persona hunter --episodes 10 --csv_out logs/miniworld/eval/ppo_miniworld_hunter_eval.csv

# A2C - Explorer
python -m src.eval_maze --algo a2c --persona explorer --episodes 10 --csv_out logs/miniworld/eval/a2c_miniworld_explorer_eval.csv

# A2C - Hunter
python -m src.eval_maze --algo a2c --persona hunter --episodes 10 --csv_out logs/miniworld/eval/a2c_miniworld_hunter_eval.csv
```

### 🎮 Visualization Commands
```bash
# PPO - Explorer
python -m src.visualize_maze --algo ppo --persona explorer --model_path models/miniworld/ppo_miniworld_explorer_seed7.zip

# PPO - Hunter
python -m src.visualize_maze --algo ppo --persona hunter --model_path models/miniworld/ppo_miniworld_hunter_seed7.zip

# A2C - Explorer
python -m src.visualize_maze --algo a2c --persona explorer --model_path models/miniworld/a2c_miniworld_explorer_seed7.zip

# A2C - Hunter
python -m src.visualize_maze --algo a2c --persona hunter --model_path models/miniworld/a2c_miniworld_hunter_seed7.zip
```

