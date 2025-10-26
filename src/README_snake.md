# 🐍 Snake Game - Deep Reinforcement Learning for Automated Testing

This document provides **complete reproducibility instructions** and explanations for the **Snake DRL Environment**.

---

## 🎯 Overview

The **Snake environment** is a custom 2D grid-based game built using the Gymnasium API.  
The goal of the agent is to control the snake to eat food and survive as long as possible without colliding with walls or itself.

Two **algorithms** and two **personas** were trained and evaluated:

| Algorithm | Persona | Goal |
|------------|----------|------|
| **PPO** | Survivor | Maximize survival time and stability |
| **PPO** | Hunter | Maximize food collection speed |
| **A2C** | Survivor | Safety and conservative navigation |
| **A2C** | Hunter | Aggressive food-seeking and exploration |

---

## Architecture Overview

| Component | Location | Description |
|------------|-----------|--------------|
| **Environment / App** | `envs/snake/snake_env.py` | Defines the *Snake* game world, rewards, actions, and rendering. |
| **Training Code** | `src/train_snake.py` | Handles model training using PPO or A2C (Stable-Baselines3).|
| **Evaluation Code** | `src/eval_snake.py` | Runs trained models for multiple episodes, logs metrics to CSV under `logs/snake/eval`. |
| **Visualization Code** | `src/visualize_snake.py` | Renders the live game and shows agent decisions visually. |
| **Trained Models** | `models/snake/` | Saved `.zip` files for each algorithm/persona. |
| **Evaluation Logs** | `logs/snake/eval/` | CSV files containing per-episode results. |
| **Plots & Analysis** | `plots/snake/` and `notebooks/snake_analysis.ipynb` | Visual comparisons of PPO vs A2C and persona performance. |
---

## Agent Training

### 1️⃣ Environment Interface
The Snake environment follows the [Gymnasium](https://gymnasium.farama.org) API.

- **Action Space:**  
  Discrete(3) → `[0 = turn left, 1 = straight, 2 = turn right]`

- **Observation Space:**  
  A 7-dimensional continuous vector (normalized to [-1, 1]):

  | Index | Feature | Description |
  |--------|----------|--------------|
  | 0 | `dx_to_food` | Horizontal distance to food |
  | 1 | `dy_to_food` | Vertical distance to food |
  | 2 | `snake_dir_x` | X-direction of current movement |
  | 3 | `snake_dir_y` | Y-direction of current movement |
  | 4 | `danger_left` | 1 if collision would occur on left turn |
  | 5 | `danger_straight` | 1 if collision straight ahead |
  | 6 | `danger_right` | 1 if collision would occur on right turn |

This vector represents a **compact world-state** that allows the policy network to reason about:
- Relative food location  
- Collision risks  
- Current heading direction

---
## 🧮 Technical Reward Shaping

| Component | Survivor Reward | Hunter Reward |
|------------|----------------|----------------|
| **Food reward** | `+10` | `+15` |
| **Alive each step** | `+0.05` | `–0.2` (time penalty) |
| **Closer to food** | `+0.1 × dist_reduction` | *None* |
| **Near walls** | `–0.2 × (1 – margin/half_grid)` | *None* |
| **Death** | `–10` | `–10` |
| **Philosophy** | Encourages patience and center play | Encourages direct and risky routes |

---

## Reward Design

The Snake environment uses **two personas** (`reward_mode`):  
`"survivor"` and `"hunter"`, each reflecting different testing objectives.

### Survivor Persona — “Play Safe, Live Long”
Goal: Maximize **survival time** while avoiding walls and self-collision.

Reward function (from code):
```python
if reward_mode == "survivor":
    reward += 0.1                     # bonus per step alive
    reward += 0.3 * dist_change       # reward for moving closer to food
    if dist_change < 0:
        reward += 0.1 * dist_change   # penalty for moving away
    reward -= 0.05 * wall_penalty     # penalize proximity to borders
    if steps % 10 == 0:
        reward += 0.2                 # milestone bonus every 10 steps
```
- +10 for eating food
- –10 for death
- Encourages long-term survival and smooth movement in open areas.
- This persona tends to learn risk-averse, stable trajectories.
---

### Hunter Persona — “Eat Fast, Risk More”
Goal: Maximize speed of food collection, even at higher risk.

Reward function (from code):

```bash
if reward_mode == "hunter":
    reward -= 0.2       # small time penalty per step (forces speed)
    if new_head == food:
        reward += 15.0  # large food bonus
```

- +15 for eating food
- –0.2 per step (time pressure)
- –10 for death
- Learns short-term greedy behavior — fast but less stable.
--- 

## 🧠 Persona Comparison Summary

| Persona | Focus | Reward Shape | Behavioral Outcome |
|----------|--------|---------------|--------------------|
| **Survivor** | Longevity & safety | `+0.1 per step alive`, `+10 food`, `-10 death` | Cautious and steady; avoids walls |
| **Hunter** | Fast food collection | `+15 food`, `-0.2 per step`, `-10 death` | Aggressive; quick toward food but risky |

---

## 🧭 Conceptual Overview

| Aspect | Survivor Persona | Hunter Persona |
|--------|------------------|----------------|
| **Goal** | Stay alive, play safe | Eat food quickly, take risks |
| **Analogy** | A careful player looping around the grid | A speedrunner dashing to the target |
| **Learning Bias** | Prefers safety and longevity | Prefers fast high-reward strategies |
| **Expected Behavior** | Long survival, smooth turns | Fast deaths, zig-zag paths |
| **Common Failure** | Dies near walls after long runs | Dies while chasing food aggressively |

---

## 🔁 Reproducibility Instructions

All experiments use a **fixed random seed = 7** and **1e6 timesteps** for training.

---

### 🧩 Training Commands

```bash
# PPO – Survivor
python -m src.train_snake --algo ppo --persona survivor --timesteps 1e6 --seed 7

# PPO – Hunter
python -m src.train_snake --algo ppo --persona hunter --timesteps 1e6 --seed 7

# A2C – Survivor
python -m src.train_snake --algo a2c --persona survivor --timesteps 1e6 --seed 7

# A2C – Hunter
python -m src.train_snake --algo a2c --persona hunter --timesteps 1e6 --seed 7
```

### Evaluation Commands

Run evaluations to generate performance CSVs for 50 episodes each:
```bash
# PPO – Survivor
python -m src.eval_snake --algo ppo --persona survivor --episodes 50  --csv_out logs/snake/eval/ppo_snake_survivor_eval.csv

# PPO – Hunter
python -m src.eval_snake --algo ppo --persona hunter --episodes 50 --csv_out logs/snake/eval/ppo_snake_hunter_eval.csv

# A2C – Survivor
python -m src.eval_snake --algo a2c --persona survivor --episodes 50  --csv_out logs/snake/eval/a2c_snake_survivor_eval.csv

# A2C – Hunter
python -m src.eval_snake --algo a2c --persona hunter --episodes 50 --csv_out logs/snake/eval/a2c_snake_hunter_eval.csv
```
### Visualization Commands

Render the trained models visually with Pygame:
```bash
# PPO – Survivor
python -m src.visualize_snake --algo ppo --persona survivor --seed 7

# PPO – Hunter
python -m src.visualize_snake --algo ppo --persona hunter --seed 7

# A2C – Survivor
python -m src.visualize_snake --algo a2c --persona survivor --seed 7

# A2C – Hunter
python -m src.visualize_snake --algo a2c --persona hunter --seed 7
```