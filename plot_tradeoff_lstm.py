import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

def get_lstm_metrics(episodes=20):
    try:
        def make_env():
            return GridOpsEnvWrapper()
            
        env = DummyVecEnv([make_env])
        env = VecNormalize.load("models/vecnormalize_lstm.pkl", env)
        
        env.training = False
        env.norm_reward = False

        model = RecurrentPPO.load("models/ppo_lstm", env=env)

        rewards = []
        blackouts = []
        total_steps = 0

        for _ in range(episodes):
            obs = env.reset()
            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            done = False
            
            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                obs, reward, done_array, info = env.step(action)
                episode_starts = done_array
                total_steps += 1
                
                if done_array[0]:
                    done = True
                    ep_info = info[0].get("episode_summary")
                    if not ep_info and "terminal_info" in info[0]:
                        ep_info = info[0]["terminal_info"].get("episode_summary")
                    
                    if ep_info:
                        rewards.append(ep_info["total_reward"])
                        blackouts.append(ep_info["total_blackouts"])

        return np.sum(rewards) / total_steps, np.mean(blackouts)
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        return 0.0, 0.0

def main():
    os.makedirs("plots", exist_ok=True)
    
    lstm_reward, lstm_blackouts = get_lstm_metrics(20)
    
    agents = ["PPO-MLP", "PPO-LSTM", "Advanced"]
    blackouts = [50.8, lstm_blackouts, 0.0]
    rewards = [0.246, lstm_reward, 5.255]
    
    plt.figure(figsize=(8, 6))
    
    # Define colors for the points
    colors = ['blue', 'green', 'orange']
    
    for i, agent in enumerate(agents):
        plt.scatter(blackouts[i], rewards[i], color=colors[i], s=150, label=agent, edgecolors='black', zorder=5)
        
        # Add a small offset to prevent text from overlapping the point
        plt.annotate(
            agent, 
            (blackouts[i], rewards[i]),
            xytext=(10, 10), 
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.title("Risk vs. Reward Trade-off", fontsize=14, fontweight='bold')
    plt.xlabel("Total Blackouts (Risk)", fontsize=12)
    plt.ylabel("Reward per Step", fontsize=12)
    
    # Add a grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # Invert x-axis if preferred (fewer blackouts is better, so 0 should be on right, but standard is fine)
    # plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plot_path = "plots/tradeoff_lstm.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
