"""
Fast Retraining Cycle — LSTM PPO on upgraded GridOpsEnv.

Key changes vs train_lstm.py:
  - total_timesteps = 100_000  (fast cycle)
  - saves to: models/ppo_lstm_final + vecnormalize_lstm_final.pkl
  - same architecture, same hyperparameters, same VecNormalize setup
  - obs shape 21 (updated wrapper: +prev_demand + fuel_remaining)
  - HealthMonitor callback: warns if training goes unstable
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Captures per-episode reward from Monitor info dict."""
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
        return True


class HealthMonitor(BaseCallback):
    """
    Prints a health snapshot every `check_freq` timesteps.
    Warns (but does NOT stop) if training looks unstable:
      - explained_variance < 0.0
      - approx_kl > 0.1
    """

    def __init__(self, check_freq: int = 10_000):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            log = self.model.logger.name_to_value
            ev  = log.get("train/explained_variance", float("nan"))
            kl  = log.get("train/approx_kl",          float("nan"))
            ent = log.get("train/entropy_loss",        float("nan"))

            status = "HEALTHY"
            if (not np.isnan(ev) and ev < 0.0) or (not np.isnan(kl) and kl > 0.1):
                status = "WARNING — check hyperparameters"

            print(
                f"  [Health @ {self.num_timesteps:>7d}]  "
                f"EV={ev:.3f}  KL={kl:.4f}  Entropy={ent:.4f}  -> {status}"
            )
        return True


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 100_000
MODEL_PATH      = "models/ppo_lstm_final"
VECNORM_PATH    = "models/vecnormalize_lstm_final.pkl"
REWARDS_PATH    = "outputs/train_rewards_lstm_final.npy"


def train(total_timesteps: int = TOTAL_TIMESTEPS) -> None:
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LSTM PPO — Fast Retraining Cycle")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Model out  : {MODEL_PATH}")
    print(f"  VecNorm out: {VECNORM_PATH}")
    print("=" * 55)

    # ── Environment ───────────────────────────────────────────
    # GridOpsEnvWrapper now exposes obs shape (21,):
    #   demand(3) + supply(3) + reputation(3) + faults(3) +
    #   time_step(1) + priority(3) + total_power(1) +
    #   prev_demand(3) + fuel_remaining(1) = 21
    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    obs_shape = env.observation_space.shape
    print(f"  Obs shape  : {obs_shape}")
    assert obs_shape == (21,), f"Expected (21,), got {obs_shape}"

    # ── Model — identical hyperparameters to improved run ─────
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate  = lambda progress: 3e-4 * progress,
        n_steps        = 1024,
        batch_size     = 64,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        max_grad_norm  = 0.5,
        policy_kwargs  = dict(net_arch=[256, 256]),
        verbose        = 1,
    )

    lr_at_start = model.lr_schedule(1.0)
    lr_at_end   = model.lr_schedule(0.0)
    print(f"  LR schedule: {lr_at_start:.2e} -> {lr_at_end:.2e}")

    # ── Callbacks ─────────────────────────────────────────────
    reward_logger   = RewardLogger()
    health_monitor  = HealthMonitor(check_freq=10_000)

    # ── Train ─────────────────────────────────────────────────
    print("\nStarting training …\n")
    model.learn(
        total_timesteps = total_timesteps,
        callback        = [reward_logger, health_monitor],
    )

    # ── Save ──────────────────────────────────────────────────
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)

    rewards = reward_logger.rewards
    np.save(REWARDS_PATH, rewards if rewards else [])

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    if rewards:
        print(f"  Episodes logged : {len(rewards)}")
        print(f"  Mean reward     : {np.mean(rewards):.3f}")
        print(f"  Best reward     : {np.max(rewards):.3f}")
        print(f"  Last 10 mean    : {np.mean(rewards[-10:]):.3f}")
    print(f"  Saved model     : {MODEL_PATH}.zip")
    print(f"  Saved VecNorm   : {VECNORM_PATH}")
    print(f"  Saved rewards   : {REWARDS_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    train()
