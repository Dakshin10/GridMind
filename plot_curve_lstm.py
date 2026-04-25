import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs("plots", exist_ok=True)
    
    file_path = "outputs/train_rewards_lstm.npy"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        print("Note: The previous training script may not have saved this file if the RewardLogger was not used.")
        return

    rewards = np.load(file_path)
    
    # Apply moving average
    window = 20
    if len(rewards) < window:
        print("Not enough data to compute moving average.")
        window = 1
        
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 6))
    
    # Plot original rewards with low alpha
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average
    plt.plot(np.arange(window-1, len(rewards)), moving_avg, color='blue', linewidth=2, label=f'Moving Average (window={window})')
    
    plt.title("LSTM PPO Learning Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "plots/training_curve_lstm.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
