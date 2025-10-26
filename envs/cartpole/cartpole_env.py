"""
CartPolePersonaEnv
------------------
Custom wrapper around Gymnasium's CartPole-v1 for persona-based RL testing.

Features:
  - Two distinct personas:
        * solver: prioritizes stability and control (human-like safe behavior)
        * fuzzer: prioritizes exploration and risky behavior for testing coverage
  - Difficulty control:
        * none: vanilla CartPole
        * progressive: ramps up environment difficulty during an episode
        * randomize_on_reset: randomizes physics parameters each episode
  - Fault injections:
        * invert_gravity: flips angular velocity (tests robustness)
        * action_noise: random action substitution probability
        * reward_delay: small sleep to simulate lag
"""

import gymnasium as gym
import numpy as np
import time

class CartPolePersonaEnv(gym.Wrapper):
    def __init__(
        self,
        reward_mode: str = "solver",
        fault_invert_gravity: bool = False,
        fault_action_noise: float = 0.0,
        fault_reward_delay: bool = False,
        max_steps: int = 500,
        difficulty: str = "none", # "none" | "progressive" | "randomize_on_reset"
        gravity_base: float = 9.8, 
        gravity_max: float = 14.0,
        force_base: float = 10.0, 
        force_max: float = 20.0,
        tau_base: float = 0.02,  
        tau_min: float = 0.01,
        angle_limit_deg_base: float = 12.0, 
        angle_limit_deg_min: float = 8.0,
        obs_noise_std: float = 0.0,
        render_mode: str | None = None,
        **_,
    ):
        super().__init__(gym.make("CartPole-v1", render_mode=render_mode))

        assert reward_mode in ("solver", "fuzzer"), "reward_mode must be solver or fuzzer"
        assert difficulty in ("none", "progressive", "randomize_on_reset")
        self.reward_mode = reward_mode
        self.difficulty = difficulty

        # Fault toggles
        self.fault_invert_gravity = fault_invert_gravity
        self.fault_action_noise = float(fault_action_noise)
        self.fault_reward_delay = bool(fault_reward_delay)

        # Episode and noise setup
        self.max_steps = int(max_steps)
        self.obs_noise_std = float(obs_noise_std)

        # Base and target physical parameters
        self.gravity_base, self.gravity_max = gravity_base, gravity_max
        self.force_base, self.force_max = force_base, force_max
        self.tau_base, self.tau_min = tau_base, tau_min
        self.angle_limit_deg_base, self.angle_limit_deg_min = angle_limit_deg_base, angle_limit_deg_min

        # Trackers for metrics
        self.step_count = 0
        self.total_reward = 0.0
        self.unique_states = set()
        self.prev_angle = 0.0

    # ---------------- Difficulty helpers ----------------
    def _lerp(self, a, b, t):
        """Linear interpolation helper"""
        return a + (b - a) * t

    def _apply_progressive_difficulty(self):
        """Gradually ramp up difficulty (gravity↑, force↑, tau↓, angle limit↓)"""
        if self.difficulty != "progressive":
            return
        prog = min(1.0, self.step_count / max(1, self.max_steps))
        u = self.env.unwrapped
        u.gravity = self._lerp(self.gravity_base, self.gravity_max, prog)
        u.force_mag = self._lerp(self.force_base, self.force_max, prog)
        u.tau = self._lerp(self.tau_base, self.tau_min, prog)
        u.theta_threshold_radians = np.deg2rad(self._lerp(self.angle_limit_deg_base, self.angle_limit_deg_min, prog))

    def _randomize_params_on_reset(self):
        """Randomize dynamics each episode (domain randomization)"""
        if self.difficulty != "randomize_on_reset":
            return
        u = self.env.unwrapped
        u.gravity = np.random.uniform(self.gravity_base, self.gravity_max)
        u.force_mag = np.random.uniform(self.force_base, self.force_max)
        u.tau = np.random.uniform(self.tau_min, self.tau_base)
        u.theta_threshold_radians = np.deg2rad(np.random.uniform(self.angle_limit_deg_min, self.angle_limit_deg_base))

    # ---------------- Gym API ----------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._randomize_params_on_reset()
        self.step_count, self.total_reward = 0, 0.0
        self.unique_states.clear()
        self.prev_angle = obs[2]
        if self.obs_noise_std > 0:
            obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        # Random action fault
        if self.fault_action_noise > 0 and np.random.rand() < self.fault_action_noise:
            action = self.env.action_space.sample()

        # Progressive difficulty scaling
        self._apply_progressive_difficulty()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Invert angular velocity fault
        if self.fault_invert_gravity:
            obs[3] = -obs[3]

        if self.obs_noise_std > 0:
            obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)

        pole_angle, cart_pos = obs[2], obs[0]
        done = terminated or truncated

        # ---------------- Persona-based reward shaping ----------------
        shaped_reward = 0.0
        if self.reward_mode == "solver":
            # Solver persona: prioritize stability
            shaped_reward += 1.0 - (abs(pole_angle) * 2.0) # upright pole
            shaped_reward += 0.5 * (1.0 - abs(cart_pos) / 2.4) # centered cart
            shaped_reward += 0.3 * np.tanh(abs(self.prev_angle) - abs(pole_angle)) # reward recovery
            self.prev_angle = pole_angle

            # penalties / bonuses for episode termination
            if done and self.step_count < self.max_steps:
                shaped_reward -= 5.0  # early failure
            if not done and self.step_count >= self.max_steps:
                shaped_reward += 5.0  # full episode success

        else:
            # Fuzzer persona: prioritize exploration and diversity
            state_hash = tuple(np.round(obs, 1))
            new_state = state_hash not in self.unique_states
            self.unique_states.add(state_hash)

            shaped_reward += 0.1 if new_state else -0.01  # novelty reward
            if abs(pole_angle) > 0.2:
                shaped_reward += 0.05  # risky angles
            if abs(cart_pos) > 2.0:
                shaped_reward += 0.1   # extreme cart positions
            if self.fault_invert_gravity:
                shaped_reward += 0.2   # bonus for chaos tolerance

        # Optional simulated lag
        if self.fault_reward_delay:
            time.sleep(0.02)

        # ---------------- Trackers & info ----------------
        self.step_count += 1
        self.total_reward += shaped_reward

        info.update({
            "steps": self.step_count,
            "total_reward": self.total_reward,
            "unique_states": len(self.unique_states),
            "angle": float(pole_angle),
        })

        if self.step_count >= self.max_steps:
            truncated = True

        return np.array(obs, dtype=np.float32), float(shaped_reward), bool(terminated), bool(truncated), info
