import os
import numpy as np
import matplotlib.pyplot as plt

# File paths
REWARDS_PATH = "outputs/train_rewards_lstm.npy"
OUTPUT_PLOT = "outputs/training_curve.png"

def plot_training_curve():
    if not os.path.exists(REWARDS_PATH):
        print(f"Error: Could not find {REWARDS_PATH}")
        return

    # Load rewards
    rewards = np.load(REWARDS_PATH)
    episodes = np.arange(len(rewards))

    # Compute moving average
    window_size = min(50, len(rewards) // 10) # Dynamic window size
    if window_size < 1:
        window_size = 1
        
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    moving_avg_episodes = np.arange(window_size - 1, len(rewards))

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Raw rewards (light, transparent)
    plt.plot(episodes, rewards, alpha=0.3, color='steelblue', label='Raw Episode Reward')
    
    # Moving average (bold)
    plt.plot(moving_avg_episodes, moving_avg, color='navy', linewidth=2, label=f'{window_size}-Episode Moving Avg')

    plt.title('PPO LSTM Training Performance', fontsize=16, pad=15)
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save and show
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot successfully saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_training_curve()
