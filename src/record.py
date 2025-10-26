# # from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
# # from stable_baselines3 import PPO

# # # --- Load model ---
# # model = PPO.load("models/snake/a2c_snake_survivo_seed7.zip")

# # # --- Create environment ---
# # from envs.snake.snake_env import SnakeEnv
# # env = DummyVecEnv([lambda: SnakeEnv(render_mode="rgb_array", reward_mode="survivo")])

# # # --- Record video for 1 episode ---
# # video_path = "media/snake/a2c_survivo_demo.mp4"
# # env = VecVideoRecorder(env, video_path,
# #                        record_video_trigger=lambda x: x == 0,
# #                        video_length=1000,
# #                        name_prefix="snake_survivo")

# # obs = env.reset()
# # for _ in range(1000):
# #     action, _ = model.predict(obs)
# #     obs, _, done, _ = env.step(action)
# #     if done.any():
# #         break

# # env.close()

# ## 

# # from stable_baselines3 import PPO
# # from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
# # from envs.miniworld.maze_env import MiniWorldPersonaEnv

# # # --- Configuration ---
# # algo = "a2c"
# # persona = "explorer"   # change to "hunter" for hunter video
# # model_path = f"models/miniworld/{algo}_miniworld_{persona}_seed7.zip"
# # video_path = f"media/miniworld/{algo}_{persona}_demo.mp4"

# # # --- Load model ---
# # model = PPO.load(model_path)

# # # --- Create environment ---
# # def make_env():
# #     return MiniWorldPersonaEnv(persona=persona, render_mode="rgb_array", max_steps=300)

# # env = DummyVecEnv([make_env])

# # # --- Attach video recorder ---
# # env = VecVideoRecorder(
# #     env,
# #     video_folder="media/miniworld",
# #     record_video_trigger=lambda x: x == 0,  # record first episode
# #     video_length=300,                       # 300 steps (~20s at 15fps)
# #     name_prefix=f"{algo}_{persona}"
# # )

# # --- Run episode ---
# # obs = env.reset()
# # for _ in range(300):
# #     action, _ = model.predict(obs)
# #     obs, _, done, _ = env.step(action)
# #     if done.any():
# #         break

# # env.close()
# # print(f"✅ Video saved at: {video_path}")

# from stable_baselines3 import PPO, A2C
# from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
# from envs.cartpole.cartpole_env import CartPolePersonaEnv

# # -------------------------------
# # Configuration
# # -------------------------------
# algo = "ppo"              # or "a2c"
# persona = "fuzzer"        # or "fuzzer"
# model_path = f"models/cartpole/{algo}_cartpole_{persona}_seed7.zip"
# video_folder = "media/cartpole"
# video_name = f"{algo}_{persona}_demo"

# # -------------------------------
# # Load trained model
# # -------------------------------
# if algo == "ppo":
#     model = PPO.load(model_path)
# else:
#     model = A2C.load(model_path)

# # -------------------------------
# # Create the environment
# # -------------------------------
# def make_env():
#     # Use rgb_array rendering for video recording
#     return CartPolePersonaEnv(reward_mode=persona, render_mode="rgb_array")

# env = DummyVecEnv([make_env])

# # -------------------------------
# # Attach video recorder
# # -------------------------------
# env = VecVideoRecorder(
#     env,
#     video_folder=video_folder,
#     record_video_trigger=lambda x: x == 0,   # record first episode
#     video_length=500,                        # ~20 seconds
#     name_prefix=video_name
# )

# # -------------------------------
# # Run 1 episode and record
# # -------------------------------
# obs = env.reset()
# for step in range(500):
#     action, _ = model.predict(obs)
#     obs, _, done, _ = env.step(action)
#     if done.any():
#         break

# env.close()
# print(f"✅ Video saved to {video_folder}/{video_name}.mp4")


from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from envs.lunarlander.lunarlander_env import LunarLanderPersonaEnv

# -------------------------------
# Configuration
# -------------------------------
algo = "ppo"              # or "a2c"
persona = "cautious"      # or "aggressive"
model_path = f"models/lunarlander/{algo}_{persona}_seed10.zip"
video_folder = "media/lunarlander"
video_name = f"{algo}_{persona}_demo"

# -------------------------------
# Load trained model
# -------------------------------
if algo == "ppo":
    model = PPO.load(model_path)
else:
    model = A2C.load(model_path)

# -------------------------------
# Create the environment
# -------------------------------
def make_env():
    return LunarLanderPersonaEnv(reward_mode=persona, max_steps=1000)

env = DummyVecEnv([make_env])

# -------------------------------
# Attach video recorder
# -------------------------------
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: x == 0,   # record first episode
    video_length=1000,                       # number of steps (~30 seconds)
    name_prefix=video_name
)

# -------------------------------
# Run one episode and record
# -------------------------------
obs = env.reset()
for step in range(1000):
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    if done.any():
        break

env.close()
print(f"✅ Video saved to {video_folder}/{video_name}.mp4")
