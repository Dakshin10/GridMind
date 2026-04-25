"""
Clean evaluation script for ppo_lstm_final on the upgraded GridOpsEnv.
- Loads VecNormalize correctly (training=False, norm_reward=False)
- Runs 20 deterministic episodes with proper LSTM state handling
- Reports Reward/Step, Blackouts, Stability
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

MODEL_PATH  = "models/ppo_lstm_final"
VECNORM_PATH = "models/vecnormalize_lstm_final.pkl"
EPISODES    = 20


def evaluate(episodes: int = EPISODES) -> None:
    # ── Load ────────────────────────────────────────────────
    # CRITICAL:
    # Evaluation must use the same VecNormalize statistics as training.
    # Otherwise observations are mismatched and performance collapses.
    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    env = VecNormalize.load("models/vecnormalize_lstm_final.pkl", env)
    env.training = False
    env.norm_reward = False

    model = RecurrentPPO.load(MODEL_PATH, env=env)

    # ── Evaluate ────────────────────────────────────────────
    total_reward    = 0.0
    total_steps     = 0
    total_blackouts = 0.0
    stability_list  = []

    for _ in range(episodes):
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        done = np.zeros((env.num_envs,), dtype=bool)
        
        # NOTE:
        # Metrics are computed manually because env does not expose episode_stats
        ep_reward = 0.0
        ep_blackouts = 0.0
        ep_stability = []

        while not done[0]:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )

            obs, reward, done, info = env.step(action)
            episode_starts = done.copy()
            total_steps += 1

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

    # ── Report ──────────────────────────────────────────────
    reward_per_step = total_reward    / total_steps if total_steps > 0 else 0.0
    avg_blackouts   = total_blackouts / episodes
    avg_stability   = float(np.mean(stability_list)) if stability_list else 0.0

    print("\n" + "-" * 40)
    print("FINAL RESULTS")
    print("-" * 40)
    print(f"Reward/Step: {reward_per_step:.3f}")
    print(f"Blackouts:   {avg_blackouts:.3f}")
    print(f"Stability:   {avg_stability:.3f}")
    print("-" * 40)


if __name__ == "__main__":
    evaluate()
