import argparse
import imageio
import os
import numpy as np
import pygame

if not os.environ.get("DISPLAY"):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from stable_baselines3 import PPO
from environment import DoublePendulumEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gif_path', type=str, default=None)
    args = parser.parse_args()

    env = DoublePendulumEnv(reward_type='shaped')
    model = PPO.load(args.model_path, env=env)

    obs, info = env.reset()
    frames = []

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if args.gif_path:
            # Capture pygame screen for GIF
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            # Swap axes: (width, height, channels) -> (height, width, channels)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    if args.gif_path and frames:
        os.makedirs(os.path.dirname(args.gif_path) or '.', exist_ok=True)
        imageio.mimsave(args.gif_path, frames, fps=60)
        print(f"GIF saved to {args.gif_path}")


if __name__ == '__main__':
    main()
