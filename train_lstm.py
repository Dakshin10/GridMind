import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Wrap with Monitor to ensure episode info is logged correctly
    env = DummyVecEnv([lambda: Monitor(GridOpsEnvWrapper())])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=1e-4,          # Constant LR to avoid premature decay
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        vf_coef=1.0,                 # Strengthen value function learning
        max_grad_norm=0.5,
        target_kl=None,
        policy_kwargs=dict(
            net_arch=[256, 256],
            enable_critic_lstm=True,
        ),
        verbose=1
    )

    logger = RewardLogger()

    model.learn(
        total_timesteps=200_000,
        callback=logger
    )

    model.save("models/ppo_lstm_final")
    env.save("models/vecnormalize_lstm_final.pkl")

    rewards_to_save = logger.rewards if logger.rewards else []
    np.save("outputs/train_rewards_lstm.npy", rewards_to_save)
    print(f"Rewards logged: {len(rewards_to_save)} episodes")
    print("Saved: outputs/train_rewards_lstm.npy")

if __name__ == "__main__":
    train()
