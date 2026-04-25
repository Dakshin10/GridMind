import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import GridOpsEnvWrapper

def evaluate_agent(episodes=20):
    model = PPO.load("models/ppo_gridops_final")
    env = GridOpsEnvWrapper()

    rewards = []
    blackouts = []
    stabilities = []
    total_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            
            if done:
                summary = info["episode_summary"]
                rewards.append(summary["total_reward"])
                blackouts.append(summary["total_blackouts"])
                stabilities.append(summary["avg_stability"])

    avg_reward_per_step = np.sum(rewards) / total_steps
    avg_blackouts = np.mean(blackouts)
    avg_stability = np.mean(stabilities)

    print("-" * 40)
    print("TRAINED AGENT RESULTS")
    print("-" * 40)
    print(f"Reward/Step: {avg_reward_per_step:.3f}")
    print(f"Blackouts: {avg_blackouts:.3f}")
    print(f"Stability: {avg_stability:.3f}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_agent(20)
