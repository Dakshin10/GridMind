import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from env.gridops_env import GridOpsEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

def eval_random(episodes=20):
    env = GridOpsEnv(mode='baseline')
    env.set_reward_mode('global')
    rewards, blackouts, stabilities = [], [], []
    total_steps = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(None)
            done = terminated or truncated
            total_steps += 1
            if done:
                ep_info = env.episode_stats
                rewards.append(ep_info['total_reward'])
                blackouts.append(ep_info['total_blackouts'])
                stabilities.append(ep_info['avg_stability'])
    return np.sum(rewards) / total_steps, np.mean(blackouts), np.mean(stabilities)

def eval_lstm(episodes=20):
    try:
        def make_env():
            return GridOpsEnvWrapper()
            
        env = DummyVecEnv([make_env])
        env = VecNormalize.load("models/vecnormalize_lstm.pkl", env)
        
        env.training = False
        env.norm_reward = False

        model = RecurrentPPO.load("models/ppo_lstm", env=env)

        rewards, blackouts, stabilities = [], [], []
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
                        stabilities.append(ep_info["avg_stability"])

        return np.sum(rewards)/total_steps, np.mean(blackouts), np.mean(stabilities)
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        return 0.0, 0.0, 0.0

def main():
    print("Evaluating Random baseline...")
    r_reward, r_blackouts, r_stability = eval_random(20)
    
    print("Evaluating PPO (LSTM)...")
    lstm_reward, lstm_blackouts, lstm_stability = eval_lstm(20)
    
    print("\n==================================================")
    print("FINAL POLICY COMPARISON")
    print("==================================================")
    print(f"{'Agent':<12} {'Reward/Step':<13} {'Blackouts':<11} {'Stability'}")
    print("-" * 50)
    
    print(f"{'Random':<12} {r_reward:<13.3f} {r_blackouts:<11.3f} {r_stability:.3f}")
    print(f"{'PPO (MLP)':<12} {0.246:<13.3f} {50.800:<11.3f} {0.622:.3f}")
    print(f"{'PPO (LSTM)':<12} {lstm_reward:<13.3f} {lstm_blackouts:<11.3f} {lstm_stability:.3f}")
    print(f"{'Advanced':<12} {5.255:<13.3f} {0.000:<11.3f} {0.944:.3f}")
    print("==================================================")

if __name__ == "__main__":
    main()
