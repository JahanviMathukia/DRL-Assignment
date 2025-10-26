"""
LunarLanderPersonaEnv
=====================

A persona-based environment wrapper for LunarLander-v2 with shaped rewards for faster and more stable Deep Reinforcement Learning (DRL) convergence.

Personas:
----------
1. **Cautious** – Prefers smooth, slow, centered, and upright landings.
   - Rewards stability, balance, and safe touchdowns.
   - Penalizes speed, angular velocity, and off-center positions.

2. **Aggressive** – Prioritizes faster descents and fuel efficiency.
   - Encourages quicker landings with moderate tolerance for tilt or impact.

Reward Shaping Overview:
-------------------------
Each step, the total shaped reward is composed of multiple weighted terms:

    shaped = (
        w_alive                     # + small per-step survival reward
      + exp(-|x|*2.0)*w_center       # + center alignment (x≈0)
      + exp(-|y-0.3|)*w_height       # + altitude control (steady descent)
      + exp(-|θ|)*w_stability        # + upright orientation
      + exp(-|θ_dot|*0.5)*w_stability/2  # + angular velocity control
      + w_speed * (|ẋ| + |ẏ|)      # - penalize excessive linear speed
      + landing/touch bonuses        # + large reward if landed softly/upright
      + crash penalty                # - penalty for crashes
      + w_native * native_reward     # + small contribution from Gym's reward
    )

This design ensures smooth gradients and interpretable progress, letting
agents learn both safe and efficient landing strategies.

"""

import gymnasium as gym
import numpy as np

class LunarLanderPersonaEnv(gym.Wrapper):

    def __init__(self, reward_mode: str = "cautious", max_steps: int = 1000):
        assert reward_mode in ("cautious", "aggressive"), f"Invalid persona: {reward_mode}"
        env = gym.make("LunarLander-v2", render_mode=None)
        super().__init__(env)

        self.mode = reward_mode
        self.max_steps = max_steps
        self.steps = 0
        self.cumulative_reward = 0.0

        # Persona reward configuration
        if self.mode == "cautious":
            self.cfg = dict(
                w_alive=0.5, # survival reward per step
                w_center=1.5, # stay near x=0
                w_height=1.0, # descend steadily
                w_stability=1.3, # upright and low angular velocity
                w_soft_landing=20.0, # reward smooth touchdown
                w_touch=15.0, # reward for any landing contact
                w_crash=-40.0, # strong penalty for crashing
                w_native=0.2, # blend in native reward
                w_speed=-0.3, # penalize excessive linear speed
            )
        else: # aggressive persona
            self.cfg = dict(
                w_alive=0.3,
                w_center=0.8,
                w_height=0.6,
                w_stability=0.8,
                w_soft_landing=10.0,
                w_touch=12.0,
                w_crash=-25.0,
                w_native=0.3,
                w_speed=-0.15,
            )

    def reset(self, **kwargs):
        """Reset the environment and episode statistics."""
        obs, info = self.env.reset(**kwargs)
        self.steps = 0
        self.cumulative_reward = 0.0
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        """Apply action and compute persona-shaped reward."""
        obs, native_r, terminated, truncated, info = self.env.step(action)
        x, y, x_dot, y_dot, theta, theta_dot, left_contact, right_contact = obs
        done = terminated or truncated
        cfg = self.cfg

        # === Reward Shaping ===
        # 1️. Base survival reward
        reward = cfg["w_alive"]

        # 2. Center alignment (x≈0)
        center_r = np.exp(-abs(x) * 2.0) * cfg["w_center"]

        # 3. Height control (steady descent)
        height_r = np.exp(-abs(y - 0.3)) * cfg["w_height"]

        # 4. Stability (upright and low angular velocity)
        stability_r = np.exp(-abs(theta)) * cfg["w_stability"]
        stability_r += np.exp(-abs(theta_dot) * 0.5) * (cfg["w_stability"] / 2)

        # 5. Speed penalty (reduce excessive velocity)
        speed_penalty = cfg["w_speed"] * (abs(x_dot) + abs(y_dot))

        # 6. Landing rewards
        if left_contact or right_contact:
            reward += cfg["w_touch"]
            if abs(x_dot) < 0.3 and abs(y_dot) < 0.3 and abs(theta) < 0.2:
                reward += cfg["w_soft_landing"]

        # 7. Crash penalties
        if done and not (left_contact or right_contact):
            reward += cfg["w_crash"]

        # 8. Combine all components
        shaped = reward + center_r + height_r + stability_r + speed_penalty

        # 9. Add small portion of native reward for realism
        shaped += cfg["w_native"] * native_r

        self.steps += 1
        self.cumulative_reward += shaped
        if self.steps >= self.max_steps:
            truncated = True
            done = True

        info.update({
            "episode_steps": self.steps,
            "cumulative_reward": self.cumulative_reward,
            "x_position": x,
            "y_position": y,
            "angle": theta,
        })

        return np.array(obs, dtype=np.float32), float(shaped), bool(terminated), bool(truncated), info