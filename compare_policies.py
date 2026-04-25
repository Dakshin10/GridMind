import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from env.gridops_env import GridOpsEnv

def eval_policy(policy_name, episodes=20):
    if policy_name == "PPO":
        model = PPO.load("models/ppo_gridops_final")
        env = GridOpsEnvWrapper()
    elif policy_name == "Random":
        env = GridOpsEnv(mode="baseline")
        env.set_reward_mode("global")
    elif policy_name == "Advanced":
        env = GridOpsEnv(mode="advanced")
        env.set_reward_mode("global")
        
    rewards = []
    total_steps = 0
    
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
            total_steps += 1
            
            if done:
                if "episode_summary" in info:
                    rewards.append(info["episode_summary"]["total_reward"])
                else:
                    rewards.append(env.episode_stats["total_reward"])

    return np.sum(rewards) / total_steps

def main():
    os.makedirs("plots", exist_ok=True)
    policies = ["Random", "PPO", "Advanced"]
    scores = []
    
    for p in policies:
        print(f"Evaluating {p}...")
        score = eval_policy(p, 20)
        scores.append(score)
        print(f"{p} Reward/Step: {score:.3f}")
        
    plt.figure(figsize=(8, 6))
    bars = plt.bar(policies, scores, color=['gray', 'blue', 'orange'])
    
    plt.title("Policy Comparison: Reward per step")
    plt.ylabel("Reward per step")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        # Handle negative values for text placement
        offset = 0.05 if yval > 0 else -0.15
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va='bottom' if yval > 0 else 'top')

    plt.tight_layout()
    plt.savefig("plots/policy_comparison.png")
    print("Plot saved to plots/policy_comparison.png")

if __name__ == "__main__":
    main()
