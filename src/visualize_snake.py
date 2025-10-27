


# src/visualize_snake.py
import time
import argparse
from stable_baselines3 import PPO, A2C
from envs.snake.snake_env import SnakeEnv

"""
Visualize a trained PPO or A2C agent playing the Snake game.

Displays:
---------
- Real-time pygame window
- Console output with step count, score, and total reward
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo", "a2c"], required=True,
                        help="Algorithm used (ppo or a2c)")
    parser.add_argument("--persona", type=str, choices=["survivor", "hunter"], required=True,
                        help="Persona or reward mode")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used during training")
    parser.add_argument("--grid_size", type=int, default=10, help="Snake grid size")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps to run the demo")
    parser.add_argument("--model_dir", type=str, default="models/snake", help="Base directory for saved models")
    args = parser.parse_args()

    # Determine model path
    model_path = f"{args.model_dir}/{args.algo}_snake_{args.persona}_seed{args.seed}.zip"
    print(f"Loading model from: {model_path}")

    # Create environment with human rendering
    env = SnakeEnv(grid_size=args.grid_size, reward_mode=args.persona, render_mode="human")

    # Load the model
    ModelClass = PPO if args.algo == "ppo" else A2C
    model = ModelClass.load(model_path, env=env)

    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    print("\n🐍 Snake Agent Playing...")
    print("-" * 40)

    while not done and steps < args.max_steps:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(int(action))
        total_reward += reward
        steps += 1
        env.render()

        if steps % 5 == 0 or done:
            print(f"Step: {steps:>4} | Score: {info['score']:>2} | Total Reward: {total_reward:>6.2f}", end="\r")

        time.sleep(0.1) 

    print("\n\nEpisode finished.")
    print(f"Total Steps: {steps}, Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
