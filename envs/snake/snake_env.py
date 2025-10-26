# envs/snake/snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

"""
Features:
---------
- Discrete action space: 0 = left, 1 = straight, 2 = right
- Two personas:
    1. "survivor" — maximizes life, minimizes collisions
    2. "hunter"   — prioritizes eating food quickly
- Optimized reward design for stable learning
- Compatible with Stable-Baselines3 PPO, A2C, DQN
- Optional pygame rendering for visualization

Observation:
------------
A compact numerical vector:
[dx_to_food, dy_to_food, snake_dir_x, snake_dir_y,
 danger_left, danger_straight, danger_right]

All values normalized to [-1, 1].

Reward Design:
--------------
Persona "survivor":
    +0.1 per time step alive
    +10 for eating food
    -10 for death
Persona "hunter":
    +15 for eating food
    -0.2 per step (time pressure)
    -10 for death
"""

# Direction vectors (Up, Right, Down, Left)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 12}

    def __init__(self, grid_size=20, reward_mode="survivor", render_mode=None, seed=None):
        super().__init__()
        assert reward_mode in ["survivor", "hunter"], "Invalid persona"
        self.grid_size = grid_size
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.window_size = 600
        self.block_size = self.window_size // grid_size

        # Action space: turn left, straight, right
        self.action_space = spaces.Discrete(3)

        # Observation: 7 continuous features normalized to [-1,1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Random generator
        self.rng = np.random.default_rng(seed)

        self.window = None
        self.clock = None
        self.reset()

    # Core Gym methods
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = random.choice(DIRECTIONS)
        self.spawn_food()
        self.steps = 0
        self.score = 0
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Convert relative turn to new direction
        self.direction = self._turn(self.direction, action)

        # Move snake
        new_head = (self.snake[0][0] + self.direction[0],
                    self.snake[0][1] + self.direction[1])
        self.steps += 1
        reward = 0.0

        # Collision check
        if (not 0 <= new_head[0] < self.grid_size or
                not 0 <= new_head[1] < self.grid_size or
                new_head in self.snake):
            self.done = True
            reward = -10.0
            obs = self._get_obs()
            return obs, reward, True, False, {"score": self.score}

        # Add new head
        self.snake.insert(0, new_head)

        # Distance to food before and after move
        old_dist = np.linalg.norm(
            np.array(self.snake[1]) - np.array(self.food))
        new_dist = np.linalg.norm(
            np.array(self.snake[0]) - np.array(self.food))
        dist_change = old_dist - new_dist

        # Check for food
        if new_head == self.food:
            self.score += 1
            self.spawn_food()
            reward += 10.0 if self.reward_mode == "survivor" else 15.0
        else:
            self.snake.pop()

        if self.reward_mode == "survivor":
            # Living bonus
            reward += 0.1

            # Encourage approaching food (stronger coefficient)
            reward += 0.3 * dist_change

            # Small penalty for moving away from food
            if dist_change < 0:
                reward += 0.1 * dist_change  # negative

            # Gentler wall penalty — only significant near border
            hx, hy = self.snake[0]
            margin = min(hx, hy, self.grid_size - 1 - hx, self.grid_size - 1 - hy)
            wall_penalty = max(0, 1 - margin / (self.grid_size / 2))
            reward -= 0.05 * wall_penalty

            # Extra reward every 10th step alive (encourages longer games)
            if self.steps % 10 == 0:
                reward += 0.2
        else:  # hunter
            reward -= 0.2  # time pressure

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, self.done, False, {"score": self.score}
    
    # Helpers
    def spawn_food(self):
        empty = [(x, y) for x in range(self.grid_size)
                 for y in range(self.grid_size) if (x, y) not in self.snake]
        self.food = random.choice(empty)

    def _turn(self, current_dir, action):
        idx = DIRECTIONS.index(current_dir)
        if action == 0:  # left
            idx = (idx - 1) % 4
        elif action == 2:  # right
            idx = (idx + 1) % 4
        return DIRECTIONS[idx]

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        food_dx = (self.food[0] - head_x) / self.grid_size
        food_dy = (self.food[1] - head_y) / self.grid_size

        dir_x, dir_y = self.direction

        # danger detection (left, straight, right)
        dangers = []
        for rel_turn in [-1, 0, 1]:  # left, straight, right
            dir_idx = (DIRECTIONS.index(self.direction) + rel_turn) % 4
            vec = DIRECTIONS[dir_idx]
            next_pos = (head_x + vec[0], head_y + vec[1])
            danger = (not 0 <= next_pos[0] < self.grid_size or
                      not 0 <= next_pos[1] < self.grid_size or
                      next_pos in self.snake)
            dangers.append(1.0 if danger else 0.0)

        obs = np.array([
            food_dx, food_dy, dir_x, dir_y,
            dangers[0], dangers[1], dangers[2]
        ], dtype=np.float32)
        return obs

    # Rendering
    def _render_frame(self):
        """Render the Snake game with improved visuals."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("🐍 Snake RL Agent")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.window_size, self.window_size))
        surface.fill((30, 30, 30))  # dark grey background

        # --- draw faint grid lines for depth ---
        for x in range(0, self.window_size, self.block_size):
            pygame.draw.line(surface, (50, 50, 50), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.block_size):
            pygame.draw.line(surface, (50, 50, 50), (0, y), (self.window_size, y))

        # --- draw fruit (red circle with gradient halo) ---
        fx, fy = self.food
        food_center = (
            fx * self.block_size + self.block_size // 2,
            fy * self.block_size + self.block_size // 2,
        )
        # outer glow
        pygame.draw.circle(surface, (255, 80, 80), food_center, self.block_size // 2)
        # bright center
        pygame.draw.circle(surface, (255, 0, 0), food_center, self.block_size // 3)
        # highlight
        pygame.draw.circle(surface, (255, 200, 200), (food_center[0]-3, food_center[1]-3), self.block_size // 6)

        # --- draw snake ---
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(
                x * self.block_size,
                y * self.block_size,
                self.block_size - 1,
                self.block_size - 1,
            )
            if i == 0:
                # head with bright gradient
                pygame.draw.rect(surface, (0, 255, 100), rect, border_radius=8)
                pygame.draw.circle(surface, (0, 200, 80),
                                   rect.center, self.block_size // 4)
            else:
                # softer green body
                color = (0, 180 - i * 5 if 180 - i * 5 > 80 else 80, 0)
                pygame.draw.rect(surface, color, rect, border_radius=6)

        # blit everything to window
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.render_mode == "rgb_array":
            # Return numpy image for video recording
            surface = pygame.Surface((self.window_size, self.window_size))
            surface.fill((30, 30, 30))
            for x in range(0, self.window_size, self.block_size):
                pygame.draw.line(surface, (50, 50, 50), (x, 0), (x, self.window_size))
            for y in range(0, self.window_size, self.block_size):
                pygame.draw.line(surface, (50, 50, 50), (0, y), (self.window_size, y))
            for (x, y) in self.snake:
                pygame.draw.rect(
                    surface, (0, 255, 0),
                    (x * self.block_size, y * self.block_size, self.block_size - 1, self.block_size - 1)
                )
            fx, fy = self.food
            pygame.draw.circle(surface, (255, 0, 0),
                            (fx * self.block_size + self.block_size // 2,
                                fy * self.block_size + self.block_size // 2),
                            self.block_size // 3)
            # Convert to numpy array (H, W, 3)
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), (1, 0, 2))
        elif self.render_mode == "human":
            self._render_frame()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
