import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import DoublePendulumEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--reward_type', type=str, default='shaped', choices=['baseline', 'shaped'])
    parser.add_argument('--save_path', type=str, default='models/ppo_model')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    # Wrap env with Monitor to log training metrics
    log_dir = f"logs/{args.reward_type}_reward"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(DoublePendulumEnv(reward_type=args.reward_type), log_dir)

    print(f"Starting training with {args.reward_type} reward for {args.timesteps} timesteps...")

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")


if __name__ == '__main__':
    main()
