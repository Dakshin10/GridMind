import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def evaluate_improved_ppo(episodes=20):
    def make_env():
        return GridOpsEnvWrapper()
        
    env = DummyVecEnv([make_env])
    env = VecNormalize.load("models/vecnormalize_improved.pkl", env)
    
    # Crucial: do not update stats during evaluation
    env.training = False
    env.norm_reward = False

    model = PPO.load("models/ppo_improved", env=env)

    rewards = []
    blackouts = []
    stabilities = []
    total_steps = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done_array, info = env.step(action)
            total_steps += 1
            if done_array[0]:
                done = True
                ep_info = info[0].get("episode_summary")
                if not ep_info and "terminal_info" in info[0]:
                    ep_info = info[0]["terminal_info"].get("episode_summary")
                
                if ep_info:
                    rewards.append(ep_info["total_reward"])
                    blackouts.append(ep_info["total_blackouts"])
                    stabilities.append(ep_info["avg_stability"])

    avg_reward_per_step = np.sum(rewards) / total_steps
    avg_blackouts = np.mean(blackouts)
    avg_stability = np.mean(stabilities)

    print("-" * 40)
    print("IMPROVED PPO RESULTS")
    print("-" * 40)
    print(f"Reward/Step: {avg_reward_per_step:.3f}")
    print(f"Blackouts: {avg_blackouts:.3f}")
    print(f"Stability: {avg_stability:.3f}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_improved_ppo(20)
