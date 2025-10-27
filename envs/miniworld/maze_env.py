# envs/miniworld/maze_env.py
"""
MiniWorldPersonaEnv
====================

A persona-based reinforcement learning environment built on top of MiniWorld Maze.
Adds custom reward shaping for two agent "personas":

1. **Explorer** – Encourages exploration, movement, and curiosity.
   - Rewards novelty (visiting new areas), moving, avoiding collisions, 
     and reaching the goal slowly but safely.

2. **Hunter** – Focuses on speed, precision, and goal-seeking.
   - Rewards progress toward the goal, orientation facing the goal,
     and reaching it quickly with minimal collisions.

Both share safety, stability, and termination conditions but differ in their incentives.
"""

import gymnasium as gym
import numpy as np
import miniworld
from typing import Optional, Tuple, Dict, Any
import os
import math
import random

# # Disable text labels and enforce headless EGL rendering for compatibility
# os.environ["MINIWORLD_NO_TEXT_LABEL"] = "1"
# os.environ["PYOPENGL_PLATFORM"] = "egl"

class MiniWorldPersonaEnv(gym.Wrapper):
    """
    Adds exploration tracking, shaped rewards, and goal-based behaviors
    to accelerate DRL convergence and encourage distinct persona behaviors.
    """

    def __init__(
        self,
        env_id: str = "MiniWorld-Maze-v0",
        persona: str = "explorer",
        max_steps: int = 300,
        render_mode: Optional[str] = None,
    ):
        super().__init__(gym.make(env_id, render_mode=render_mode))
        self.persona = persona.lower()
        assert self.persona in ["explorer", "hunter"], "persona must be 'explorer' or 'hunter'"

        # Step and position trackers
        self.max_steps = max_steps
        self.step_count = 0
        self.visited_cells = set()  # track explored areas for novelty reward
        self.cell_size = 0.5        # grid resolution for exploration tracking

        self.prev_dist = None       # for goal progress
        self.prev_pos = None        # for movement tracking
        self.idle_counter = 0
        self.goal_pos = None        # fallback virtual goal (if not in env)
        self.prev_obs = None        # previous frame for curiosity reward
        self._was_colliding = False
    
    # Utility functions
    def _quantize_pos(self, pos: np.ndarray) -> Tuple[int, int]:
        """Discretize agent position into a grid cell for novelty tracking."""
        return (int(pos[0] / self.cell_size), int(pos[2] / self.cell_size))

    def _compute_distance(self, agent_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """Euclidean distance between agent and goal."""
        return float(np.linalg.norm(agent_pos[[0, 2]] - goal_pos[[0, 2]]))

    def _is_facing_goal(self, agent_dir, agent_pos, goal_pos) -> bool:
        """Check if the agent is facing roughly toward the goal (±30° tolerance)."""
        if np.isscalar(agent_dir) or np.ndim(agent_dir) == 0:
            agent_dir_vec = np.array([math.cos(agent_dir), math.sin(agent_dir)])
        else:
            agent_dir_vec = np.array(agent_dir)[[0, 2]]

        to_goal = goal_pos[[0, 2]] - agent_pos[[0, 2]]
        to_goal /= np.linalg.norm(to_goal) + 1e-8

        dot = np.dot(agent_dir_vec, to_goal)
        return dot > math.cos(math.radians(30))

    def _detect_collision(self, agent_pos: np.ndarray) -> bool:
        """Rough collision detection with room boundaries."""
        room = getattr(self.unwrapped, "room", None)
        if room is None:
            return False

        world_min = np.array([-room.width / 2, 0, -room.length / 2])
        world_max = np.array([room.width / 2, 0, room.length / 2])
        outside = np.any(agent_pos < world_min + 0.2) or np.any(agent_pos > world_max - 0.2)
        return bool(outside)

    # Core step with persona-specific reward shaping
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        done = terminated or truncated or (self.step_count >= self.max_steps)

        reward = 0.0  # reset shaped reward accumulator

        agent = self.unwrapped.agent
        agent_pos = np.array(agent.pos)
        agent_dir = np.array(agent.dir)

        # Goal detection or fallback
        if hasattr(self.unwrapped, "goal") and self.unwrapped.goal is not None:
            goal_pos = np.array(self.unwrapped.goal.pos)
        else:
            if self.goal_pos is None:
                room = getattr(self.unwrapped, "room", None)
                if room:
                    # place virtual goal in top-right region
                    gx = random.uniform(0.4 * room.width, 0.45 * room.width)
                    gz = random.uniform(0.4 * room.length, 0.45 * room.length)
                    self.goal_pos = np.array([gx, 0, gz])
                else:
                    self.goal_pos = np.array([4.0, 0, 4.0])  # default fallback
            goal_pos = self.goal_pos

        curr_dist = self._compute_distance(agent_pos, goal_pos)
        if self.prev_dist is None:
            self.prev_dist = curr_dist

        # REWARD SHAPING: EXPLORER
        if self.persona == "explorer":
            # (1) Per-step survival reward
            reward += 0.05

            # (2) Novelty reward for visiting new cells
            cell = self._quantize_pos(agent_pos)
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                reward += 0.6

            # (3) Encourage movement; discourage standing still
            if self.prev_pos is not None:
                moved = np.linalg.norm(agent_pos - self.prev_pos)
                reward += 0.04 if moved > 0.03 else -0.03

            # (4) Collision handling: penalize, and nudge agent away
            collided = self._detect_collision(agent_pos)
            if collided:
                self.unwrapped.agent.dir += np.random.uniform(-0.5, 0.5)
                self.unwrapped.agent.pos += np.random.uniform(-0.1, 0.1, size=3)
                reward -= 0.2
            else:
                reward += 0.02  # reinforce safe navigation

            # (5) Curiosity reward for visual novelty (frame difference)
            try:
                if self.prev_obs is not None:
                    obs_arr = np.asarray(obs, dtype=np.float32)
                    prev_arr = np.asarray(self.prev_obs, dtype=np.float32)
                    if obs_arr.shape == prev_arr.shape:
                        diff = float(np.mean(np.abs(obs_arr - prev_arr)))
                        reward += 0.0005 * diff
            except Exception:
                pass
            self.prev_obs = np.copy(obs)

            # (6) Reward proximity to goal; terminate if close enough
            if curr_dist < 2.0:
                reward += 0.5
            if curr_dist < 0.5:
                reward += 5.0
                done = True

            # (7) Penalize timeouts if goal not reached
            if done and curr_dist > 0.5:
                reward -= 0.5

        # REWARD SHAPING: HUNTER
        elif self.persona == "hunter":
            # (1) Reward progress toward goal
            progress = self.prev_dist - curr_dist
            reward += 0.2 * progress

            # (2) Bonus for facing goal
            if self._is_facing_goal(agent_dir, agent_pos, goal_pos):
                reward += 0.05

            # (3) Time penalty to encourage speed
            reward -= 0.02

            # (4) Collision penalty
            if self._detect_collision(agent_pos):
                reward -= 1.0

            # (5) Big reward on reaching goal; terminate episode
            if curr_dist < 0.5:
                reward += 10.0
                done = True

            # (6) Penalize termination if goal not reached
            if done and curr_dist > 0.5:
                reward -= 3.0

        # Update trackers and cleanup on episode end
        self.prev_pos = agent_pos.copy()
        self.prev_dist = curr_dist

        if done:
            self.step_count = 0
            self.visited_cells.clear()
            self.prev_dist = None
            self.idle_counter = 0
            self.prev_pos = None
            self.goal_pos = None
            self.prev_obs = None

        return obs, float(reward), done, False, info

    # Reset environment and trackers
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.visited_cells.clear()
        self.prev_dist = None
        self.prev_pos = None
        self.idle_counter = 0
        self.goal_pos = None
        self._was_colliding = False
        self.prev_obs = None
        return obs, info

    # Safe render for RGB observations
    def render(self):
        """Render RGB frame safely, even if MiniWorld fails internally."""
        try:
            base_env = self.env.unwrapped
            if hasattr(base_env, "display_text"):
                base_env.display_text = False
            return base_env.render()
        except Exception as e:
            print("Render failed:", e)
            return np.zeros((240, 320, 3), dtype=np.uint8)