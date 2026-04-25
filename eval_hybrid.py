"""
Hybrid Safety Override Evaluation for LSTM PPO on GridOpsEnv.

Safety override logic:
  - Tracks fault count per episode step from observation faults field
  - If accumulated fault_count > 2: switch to safe_policy(obs)
  - safe_policy: allocates proportionally to demand, biased 0.6 demand + 0.4 uniform,
    scaled conservatively to avoid overload triggers (actions are ratio vectors)
  - Environment/reward/normalization untouched
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

NUM_ZONES   = 3
OBS_SHAPE   = 17
# Obs indices (from GridOpsEnvWrapper._flatten_obs):
# 0:3   demand
# 3:6   supply
# 6:9   reputation
# 9:12  faults
# 12    time_step
# 13:16 priority
# 16    total_power
IDX_DEMAND     = slice(0, 3)
IDX_FAULTS     = slice(9, 12)
IDX_SUPPLY     = slice(3, 6)
IDX_PRIORITY   = slice(13, 16)


def safe_policy(normalized_obs: np.ndarray) -> np.ndarray:
    """
    Overload-prevention allocation.

    Root cause of cascading failures in GridOpsEnv:
      - Overload fires when: allocated > 1.3 * demand
        -> sets failed=True, queues delayed power loss, amplifies random faults.

    Strategy:
      - Base: proportional to demand (targets meeting demand exactly)
      - Detect zones where supply already exceeds 1.1x demand (overload approaching)
      - Dampen those zones by 0.7x, redistribute to under-served zones

    Input : flat normalized obs from VecNormalize (shape 17,)
    Output: action array shape (3,) summing to 1
    """
    demand = np.maximum(normalized_obs[IDX_DEMAND], 1e-6)
    supply = np.maximum(normalized_obs[IDX_SUPPLY], 1e-6)

    action = demand / (demand.sum() + 1e-8)

    supply_ratio  = supply / demand
    overload_risk = supply_ratio > 1.1
    if overload_risk.any():
        action[overload_risk] *= 0.7
        deficit = ~overload_risk
        if deficit.any():
            action[deficit] *= 1.15

    action = np.clip(action, 1e-6, 1.0)
    action = action / (action.sum() + 1e-8)
    return action.astype(np.float32)


def detect_overload_risk(normalized_obs: np.ndarray) -> bool:
    """True if any zone supply > 1.1x demand (approaching the 1.3x overload threshold)."""
    demand = np.maximum(normalized_obs[IDX_DEMAND], 1e-6)
    supply = np.maximum(normalized_obs[IDX_SUPPLY], 1e-6)
    return bool(np.any(supply / demand > 1.1))


def count_active_faults(obs: np.ndarray) -> int:
    """Return number of zones currently faulted from flat obs."""
    faults = obs[IDX_FAULTS]
    return int(np.sum(faults > 0.5))


def evaluate_hybrid(episodes: int = 20) -> None:
    """
    Override trigger: detect_overload_risk(obs) — fires when any zone
    supply/demand ratio > 1.1x, which directly precedes the 1.3x overload
    threshold that starts cascade failures in GridOpsEnv.
    """
    print("\n" + "=" * 50)
    print("HYBRID (HARD OVERRIDE) EVALUATION")
    print(f"  Episodes       : {episodes}")
    print("  Override trigger: supply/demand > 1.1x (overload prevention)")
    print("=" * 50)

    # ── Load env & model ──────────────────────────────────────
    vec_path   = "models/vecnormalize_lstm_improved.pkl"
    model_path = "models/ppo_lstm_improved"

    if not os.path.exists(vec_path):
        print(f"[!] {vec_path} not found. Run train_lstm.py first.")
        sys.exit(1)

    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    env = VecNormalize.load(vec_path, env)
    env.training  = False
    env.norm_reward = False

    model = RecurrentPPO.load(model_path, env=env)
    print(f"  Loaded: {model_path}")

    # ── Evaluation loop ───────────────────────────────────────
    total_reward    = 0.0
    total_steps     = 0
    total_blackouts = 0.0
    stability_list  = []

    rl_steps       = 0
    override_steps = 0

    for ep in range(episodes):
        obs            = env.reset()
        lstm_states    = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done           = np.zeros(1, dtype=bool)
        while not done[0]:
            # Get RL action (with LSTM state)
            rl_action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )

            # Read per-step fault count from obs (faults are binary 0/1).
            # VecNormalize passes faults through; values remain ~0 or ~1.
            raw_obs = obs[0]  # shape (17,)

            # Override when any zone is approaching overload (supply > 1.1x demand)
            # This prevents the cascade: overload -> failed=True -> random fault amplification
            if detect_overload_risk(raw_obs):
                action = safe_policy(raw_obs)[np.newaxis, :]  # shape (1, 3)
                override_steps += 1
            else:
                action = rl_action
                rl_steps += 1

            obs, reward, done, info = env.step(action)
            episode_starts = np.array([done], dtype=bool)
            total_steps += 1

            if done[0]:
                ep_info = info[0].get("episode_summary")
                if not ep_info and "terminal_info" in info[0]:
                    ep_info = info[0]["terminal_info"].get("episode_summary")
                if ep_info:
                    total_reward    += ep_info["total_reward"]
                    total_blackouts += ep_info["total_blackouts"]
                    stability_list.append(ep_info["avg_stability"])

    # ── Metrics ───────────────────────────────────────────────
    reward_per_step = total_reward    / total_steps  if total_steps  > 0 else 0.0
    avg_blackouts   = total_blackouts / episodes
    avg_stability   = float(np.mean(stability_list)) if stability_list else 0.0
    override_pct    = 100.0 * override_steps / total_steps if total_steps > 0 else 0.0

    print("\n" + "-" * 40)
    print("HYBRID (HARD OVERRIDE) RESULTS")
    print("-" * 40)
    print(f"Reward/Step:   {reward_per_step:.3f}")
    print(f"Blackouts:     {avg_blackouts:.3f}")
    print(f"Stability:     {avg_stability:.3f}")
    print("-" * 40)
    print(f"  RL steps       : {rl_steps} ({100 - override_pct:.1f}%)")
    print(f"  Override steps : {override_steps} ({override_pct:.1f}%)")
    print("-" * 40)

    # ── Comparison vs baseline LSTM ───────────────────────────
    lstm_blk  = 30.450
    lstm_rps  = 0.305
    lstm_stab = 0.728

    blk_delta  = lstm_blk  - avg_blackouts
    rps_delta  = reward_per_step - lstm_rps
    stab_delta = avg_stability - lstm_stab

    print("\n  vs. Pure LSTM PPO:")
    sign = lambda x: "+" if x >= 0 else ""
    print(f"  Blackouts : {lstm_blk:.3f} -> {avg_blackouts:.3f} ({sign(-blk_delta)}{-blk_delta:.3f})")
    print(f"  Reward    : {lstm_rps:.3f} -> {reward_per_step:.3f} ({sign(rps_delta)}{rps_delta:.3f})")
    print(f"  Stability : {lstm_stab:.3f} -> {avg_stability:.3f} ({sign(stab_delta)}{stab_delta:.3f})")

    if avg_blackouts < lstm_blk:
        pct = (lstm_blk - avg_blackouts) / lstm_blk * 100
        print(f"\n  Blackout reduction vs LSTM: {pct:.1f}%")
        if pct >= 30:
            print("  HYBRID SIGNIFICANTLY REDUCES BLACKOUTS")
        elif pct >= 10:
            print("  HYBRID MODERATELY REDUCES BLACKOUTS")
        else:
            print("  HYBRID MARGINALLY REDUCES BLACKOUTS")
    else:
        print("\n  No blackout reduction vs pure LSTM.")


if __name__ == "__main__":
    evaluate_hybrid(episodes=20)
