import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

def get_lstm_reward(episodes=20):
    try:
        def make_env():
            return GridOpsEnvWrapper()
            
        env = DummyVecEnv([make_env])
        env = VecNormalize.load("models/vecnormalize_lstm.pkl", env)
        
        env.training = False
        env.norm_reward = False

        model = RecurrentPPO.load("models/ppo_lstm", env=env)

        rewards = []
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

        return np.sum(rewards) / total_steps
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        return 0.0

def main():
    os.makedirs("plots", exist_ok=True)
    
    lstm_reward = get_lstm_reward(20)
    
    agents = ["Random", "PPO-MLP", "PPO-LSTM", "Advanced"]
    rewards = [-0.2, 0.246, lstm_reward, 5.255]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, rewards, color=['gray', 'blue', 'green', 'orange'])
    
    plt.title("Policy Performance Comparison")
    plt.ylabel("Reward per Step")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        offset = 0.1 if yval >= 0 else -0.3
        va = 'bottom' if yval >= 0 else 'top'
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va=va)
        
    plt.tight_layout()
    plot_path = "plots/final_comparison_lstm.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
