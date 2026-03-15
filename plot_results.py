import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_learning_curves():
    try:
        df_baseline = load_results('logs/baseline_reward')
        df_shaped = load_results('logs/shaped_reward')

        x_base, y_base = ts2xy(df_baseline, 'timesteps')
        x_shaped, y_shaped = ts2xy(df_shaped, 'timesteps')

        plt.figure(figsize=(10, 5))
        plt.plot(x_base, y_base, label='Baseline Reward')
        plt.plot(x_shaped, y_shaped, label='Shaped Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.title('PPO Learning Curve Comparison: Double Pendulum')
        plt.legend()
        plt.grid(True)
        plt.savefig('reward_comparison.png')
        print("Plot saved as reward_comparison.png")
    except Exception as e:
        print(f"Ensure you have trained both models and logs exist. Error: {e}")

if __name__ == '__main__':
    plot_learning_curves()
