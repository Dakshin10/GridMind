import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

MODEL_PATH  = "models/ppo_lstm_final"
VECNORM_PATH = "models/vecnormalize_lstm_final.pkl"
EPISODES    = 50

def run_evaluation(env, model=None, episodes=50, is_random=False):
    total_reward    = 0.0
    total_blackouts = 0.0
    stability_list  = []

    for _ in range(episodes):
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        done = np.zeros((env.num_envs,), dtype=bool)
        
        ep_reward = 0.0
        ep_blackouts = 0.0
        ep_stability = []

        while not done[0]:
            if is_random:
                action = [env.action_space.sample()]
            else:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )

            obs, reward, done, info = env.step(action)
            episode_starts = done.copy()

            ep_reward += reward[0]
            
            if "blackouts" in info[0]:
                ep_blackouts += info[0]["blackouts"]
            elif "fault_count" in info[0]:
                ep_blackouts += info[0]["fault_count"]
                
            if "stability_score" in info[0]:
                ep_stability.append(info[0]["stability_score"])

            if done[0]:
                total_reward += ep_reward
                total_blackouts += ep_blackouts
                if ep_stability:
                    stability_list.append(float(np.mean(ep_stability)))

    avg_reward = total_reward / episodes
    avg_blackouts = total_blackouts / episodes
    avg_stability = float(np.mean(stability_list)) if stability_list else 0.0
    
    return avg_reward, avg_blackouts, avg_stability

def evaluate():
    print(f"Running evaluation over {EPISODES} episodes...")
    
    # ── Load Environment and Model ──────────────────────────
    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    
    # Random Baseline
    rand_reward, rand_blackouts, rand_stability = run_evaluation(env, episodes=EPISODES, is_random=True)
    
    # PPO LSTM Model
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False
    env.norm_reward = False

    model = RecurrentPPO.load(MODEL_PATH, env=env)
    
    lstm_reward, lstm_blackouts, lstm_stability = run_evaluation(env, model=model, episodes=EPISODES, is_random=False)

    # ── Report ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Random Policy':<15} | {'PPO LSTM':<15}")
    print("-" * 50)
    print(f"{'Avg Reward/Episode':<20} | {rand_reward:<15.3f} | {lstm_reward:<15.3f}")
    print(f"{'Avg Blackouts':<20} | {rand_blackouts:<15.3f} | {lstm_blackouts:<15.3f}")
    print(f"{'Avg Stability':<20} | {rand_stability:<15.3f} | {lstm_stability:<15.3f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
