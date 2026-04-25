import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from env.gridops_env import GridOpsEnv

def gather_data(policy_name, episodes=50):
    if policy_name == "PPO":
        model = PPO.load("models/ppo_gridops_final")
        env = GridOpsEnvWrapper()
    else:
        env = GridOpsEnv(mode="advanced")
        env.set_reward_mode("global")
        
    rewards = []
    blackouts = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            if policy_name == "PPO":
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(None)
                
            done = terminated or truncated
            
            if done:
                if "episode_summary" in info:
                    rewards.append(info["episode_summary"]["total_reward"])
                    blackouts.append(info["episode_summary"]["total_blackouts"])
                else:
                    rewards.append(env.episode_stats["total_reward"])
                    blackouts.append(env.episode_stats["total_blackouts"])

    return blackouts, rewards

def main():
    os.makedirs("plots", exist_ok=True)
    
    print("Gathering data for PPO...")
    ppo_blackouts, ppo_rewards = gather_data("PPO", 50)
    
    print("Gathering data for Advanced...")
    adv_blackouts, adv_rewards = gather_data("Advanced", 50)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(ppo_blackouts, ppo_rewards, c='blue', label='PPO', alpha=0.7, marker='o')
    plt.scatter(adv_blackouts, adv_rewards, c='orange', label='Advanced', alpha=0.7, marker='x')
    
    plt.title("Reward vs Blackouts: PPO vs Advanced")
    plt.xlabel("Total Blackouts per Episode")
    plt.ylabel("Total Reward per Episode")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("plots/scatter_ppo_vs_adv.png")
    print("Saved scatter plot to plots/scatter_ppo_vs_adv.png")

if __name__ == "__main__":
    main()
