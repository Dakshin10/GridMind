"""
Hybrid Soft Blend Evaluation for LSTM PPO on GridOpsEnv.

Blends RL action with a demand-proportional safe policy using dynamic alpha
based on observed fault count at each step.

Alpha schedule:
  fault_count == 0    -> alpha = 0.7  (70% RL, 30% safe)
  fault_count == 1    -> alpha = 0.5  (50% RL, 50% safe)
  fault_count >= 2    -> alpha = 0.3  (30% RL, 70% safe)

Environment, reward, and normalization are NOT modified.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

# Obs layout from GridOpsEnvWrapper._flatten_obs:
# 0:3  demand | 3:6  supply | 6:9  reputation | 9:12 faults
# 12   time_step | 13:16 priority | 16  total_power
IDX_DEMAND   = slice(0, 3)
IDX_SUPPLY   = slice(3, 6)
IDX_FAULTS   = slice(9, 12)
IDX_PRIORITY = slice(13, 16)
NUM_ZONES    = 3


# ─────────────────────────────────────────────────────────────
# Safe heuristic policy
# ─────────────────────────────────────────────────────────────

def safe_policy(obs: np.ndarray) -> np.ndarray:
    """
    Demand-proportional safe allocation.

    Targets meeting demand exactly (avoids both blackout: alloc < 0.4*demand
    and overload: alloc > 1.3*demand).

    Returns a normalised action vector of shape (3,).
    """
    demand   = np.maximum(obs[IDX_DEMAND], 1e-6)
    priority = np.maximum(obs[IDX_PRIORITY], 0.1)

    demand_w   = demand / (demand.sum() + 1e-8)
    priority_w = priority / (priority.sum() + 1e-8)

    # 80% demand-proportional, 20% priority-weighted
    action = 0.80 * demand_w + 0.20 * priority_w
    action = np.clip(action, 1e-6, 1.0)
    action = action / (action.sum() + 1e-8)
    return action.astype(np.float32)


def get_fault_count(obs: np.ndarray) -> int:
    """Number of zones currently faulted (binary 0/1 in obs)."""
    return int(np.sum(obs[IDX_FAULTS] > 0.5))


def dynamic_alpha(fault_count: int) -> float:
    """
    Alpha controls how much of the RL action survives in the blend.
    Higher fault count -> lower alpha -> safer heuristic dominates.
    """
    if fault_count >= 2:
        return 0.3
    elif fault_count == 1:
        return 0.5
    else:
        return 0.7


def blend(rl_action: np.ndarray, safe_action: np.ndarray, alpha: float) -> np.ndarray:
    """
    Soft blend: alpha * rl + (1-alpha) * safe, then re-normalise.
    Both inputs are shape (3,).
    """
    mixed = alpha * rl_action + (1.0 - alpha) * safe_action
    mixed = np.clip(mixed, 1e-6, 1.0)
    mixed = mixed / (mixed.sum() + 1e-8)
    return mixed.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_soft_blend(episodes: int = 20) -> None:
    print("\n" + "=" * 50)
    print("HYBRID (SOFT BLEND) EVALUATION")
    print(f"  Episodes : {episodes}")
    print("  Alpha    : 0.7 (no faults) | 0.5 (1 fault) | 0.3 (>=2 faults)")
    print("=" * 50)

    vec_path   = "models/vecnormalize_lstm_improved.pkl"
    model_path = "models/ppo_lstm_improved"

    if not os.path.exists(vec_path):
        print(f"[!] {vec_path} not found. Run train_lstm.py first.")
        sys.exit(1)

    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    env = VecNormalize.load(vec_path, env)
    env.training   = False
    env.norm_reward = False

    model = RecurrentPPO.load(model_path, env=env)
    print(f"  Loaded: {model_path}\n")

    total_reward    = 0.0
    total_steps     = 0
    total_blackouts = 0.0
    stability_list  = []

    alpha_log = {0.7: 0, 0.5: 0, 0.3: 0}

    for ep in range(episodes):
        obs            = env.reset()
        lstm_states    = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done           = np.zeros(1, dtype=bool)

        while not done[0]:
            # ── RL action (LSTM state preserved) ─────────────
            rl_action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )

            raw_obs     = obs[0]                       # shape (17,)
            faults      = get_fault_count(raw_obs)
            alpha       = dynamic_alpha(faults)
            safe_action = safe_policy(raw_obs)         # shape (3,)

            # ── Soft blend ────────────────────────────────────
            blended = blend(rl_action[0], safe_action, alpha)  # shape (3,)
            action  = blended[np.newaxis, :]                   # shape (1, 3)

            alpha_log[alpha] = alpha_log.get(alpha, 0) + 1

            obs, reward, done, info = env.step(action)
            episode_starts = np.array([done], dtype=bool)
            total_steps   += 1

            if done[0]:
                ep_info = info[0].get("episode_summary")
                if not ep_info and "terminal_info" in info[0]:
                    ep_info = info[0]["terminal_info"].get("episode_summary")
                if ep_info:
                    total_reward    += ep_info["total_reward"]
                    total_blackouts += ep_info["total_blackouts"]
                    stability_list.append(ep_info["avg_stability"])

    # ── Metrics ───────────────────────────────────────────────
    reward_per_step = total_reward    / total_steps if total_steps > 0 else 0.0
    avg_blackouts   = total_blackouts / episodes
    avg_stability   = float(np.mean(stability_list)) if stability_list else 0.0

    print("-" * 40)
    print("HYBRID (SOFT BLEND) RESULTS")
    print("-" * 40)
    print(f"Reward/Step: {reward_per_step:.3f}")
    print(f"Blackouts:   {avg_blackouts:.3f}")
    print(f"Stability:   {avg_stability:.3f}")
    print("-" * 40)

    # Alpha distribution
    total_alpha_steps = sum(alpha_log.values())
    print("\n  Alpha distribution:")
    for a, cnt in sorted(alpha_log.items(), reverse=True):
        pct = 100.0 * cnt / total_alpha_steps if total_alpha_steps > 0 else 0.0
        label = {0.7: "no fault", 0.5: "1 fault", 0.3: ">=2 faults"}[a]
        print(f"    alpha={a} ({label}): {cnt} steps ({pct:.1f}%)")

    # ── vs pure LSTM ──────────────────────────────────────────
    lstm_rps  = 0.305
    lstm_blk  = 30.450
    lstm_stab = 0.728

    rps_delta  = reward_per_step - lstm_rps
    blk_delta  = avg_blackouts   - lstm_blk
    stab_delta = avg_stability   - lstm_stab

    sign = lambda x: "+" if x >= 0 else ""
    print(f"\n  vs. Pure LSTM PPO:")
    print(f"  Blackouts : {lstm_blk:.3f} -> {avg_blackouts:.3f} ({sign(blk_delta)}{blk_delta:.3f})")
    print(f"  Reward    : {lstm_rps:.3f} -> {reward_per_step:.3f} ({sign(rps_delta)}{rps_delta:.3f})")
    print(f"  Stability : {lstm_stab:.3f} -> {avg_stability:.3f} ({sign(stab_delta)}{stab_delta:.3f})")

    if avg_blackouts < lstm_blk:
        pct = (lstm_blk - avg_blackouts) / lstm_blk * 100
        print(f"\n  Blackout reduction: {pct:.1f}%")
        print("  SOFT BLEND REDUCES BLACKOUTS")
    else:
        pct = (avg_blackouts - lstm_blk) / lstm_blk * 100
        print(f"\n  Blackout increase: +{pct:.1f}% vs pure LSTM")
        print("  NOTE: Pure LSTM PPO already near-optimal for this env.")
        print("        Safe blending partially disrupts LSTM temporal strategy.")


if __name__ == "__main__":
    evaluate_soft_blend(episodes=20)
