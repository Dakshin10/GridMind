import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper, RewardLogger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def train_ppo_improved(total_timesteps=300000):
    print("Setting up Improved PPO Environment...")
    env = DummyVecEnv([lambda: Monitor(GridOpsEnvWrapper())])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1
    )
    
    logger = RewardLogger()
    
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=logger
    )
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    model.save("models/ppo_improved")
    env.save("models/vecnormalize_improved.pkl")
    np.save("outputs/train_rewards_improved.npy", logger.rewards)
    print("Training complete. Model saved to 'models/ppo_improved', env stats to 'models/vecnormalize_improved.pkl'.")

if __name__ == "__main__":
    train_ppo_improved(300000)
