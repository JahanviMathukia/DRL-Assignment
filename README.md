# Deep Reinforcement Learning for Automated Testing  
**Assignment 1 – CSCI 3060U / Topics in CS I**

This repository implements a modular **Deep Reinforcement Learning (DRL)** framework for **automated testing** of applications and games.  
Trained DRL agents (not scripted bots) explore environments, detect issues, and generate meaningful evaluation metrics using reproducible training setups.

---

## 🎯 Project Overview

The goal is to **automate the testing of interactive systems** (2D/3D games, apps) using DRL agents trained with reward-driven personas.  
Agents are trained and evaluated across multiple environments using **PPO** and **A2C** algorithms.

---

## 🧩 Implemented Environments

| Environment | Description | Algorithms | Personas |
|--------------|-------------|-------------|-----------|
| **Snake** | Custom grid-based snake game | PPO, A2C | Survivor, Hunter|
| **MiniWorld – Maze** | 3D maze navigation using MiniWorld | PPO, A2C | Explorer, Hunter|
| **CartPole** | Classic control benchmark for baseline testing | PPO, A2C | Solver, Fuzzer |
| **LunarLander** | Physics-based lander environment | PPO, DQN | Cautious,Aggressive |

> 🧠 Focus environments: **Snake** and **MiniWorld-Maze** (primary analysis)  
> Learning environments: **CartPole** and **LunarLander** (simpler baselines)

---

## 🏗️ Folder Structure
```bash
SOKOBAN-ASSGN/
├── envs/ # 🧠 Environment definitions & reward logic
│ ├── cartpole/
│ ├── lunarlander/
│ ├── miniworld/
│ └── snake/
│
├── logs/ # 📊 Evaluation logs & TensorBoard runs
│ ├── cartpole/
│ ├── lunarlander/
│ ├── miniworld/
│ └── snake/
│
├── media/ # 🎥 Demo clips
|
├── models/ # 💾 Trained model checkpoints
│ ├── cartpole/
│ ├── lunarlander/
│ ├── miniworld/
│ └── snake/
│
├── notebooks/ # 📒 Jupyter notebooks for analysis
│ ├── miniworld_analysis.ipynb
│ └── snake_analysis.ipynb
│
├── plots/ # 📈 Generated plots and graphs
│ ├── miniworld/
│ └── snake/
│
├── src/ # ⚙️ Source code (training, evaluation, visualization)
│ ├── train_.py # Training scripts per environment
│ ├── eval_.py # Evaluation & metric collection scripts (Generates CSV files)
│ ├── visualize_*.py # Run live visualization or gameplay
│ ├── README_cartpole.md
│ ├── README_lunarlander.md
│ ├── README_snake.md
│ └── README_miniworld.md
│
└── requirements.txt # 📦 Dependency pinning for reproducibility
```


---

## ⚙️ Setup & Installation

```bash
# 1️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # (Windows: .venv\Scripts\activate)

# 2️⃣ Install dependencies
pip install -r requirements.txt
```