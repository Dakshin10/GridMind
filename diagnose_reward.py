import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper

def main():
    env = GridOpsEnvWrapper()
    env.reset()
    
    rewards = []
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            env.reset()
            
    rewards = np.array(rewards)
    mean_r = np.mean(rewards)
    min_r = np.min(rewards)
    max_r = np.max(rewards)
    
    print("----------------------------------------")
    print("REWARD DIAGNOSTICS")
    print("----------------------------------------")
    print(f"Mean reward: {mean_r:.3f}")
    print(f"Min reward: {min_r:.3f}")
    print(f"Max reward: {max_r:.3f}")
    print("----------------------------------------")
    
    print("First 20 rewards:")
    print([round(r, 3) for r in rewards[:20]])
    print("----------------------------------------")
    
    # Check criteria
    mostly_negative = mean_r < -0.1  # Random actions likely cause some negative, but we evaluate if it's overly harsh
    high_variance = np.std(rewards) > 2.0
    
    if mostly_negative or high_variance:
        print("Output summary:\nReward signal is too negative / unstable")
    else:
        print("Output summary:\nReward signal is usable")

if __name__ == "__main__":
    main()
