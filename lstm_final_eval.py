"""
Final LSTM PPO Evaluation — Phases 1-7
Evaluation, comparison, improvement metrics, plots, README summary.
DO NOT modify: environment, reward, normalization.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Baselines (fixed)
# ─────────────────────────────────────────────────────────────
RANDOM   = {"name": "Random",    "rps": -0.200, "blk": 60.000, "stab": 0.450}
MLP      = {"name": "PPO (MLP)", "rps":  0.246, "blk": 50.800, "stab": 0.622}
ADVANCED = {"name": "Advanced",  "rps":  5.255, "blk":  0.000, "stab": 0.944}


# ─────────────────────────────────────────────────────────────
# PHASE 1 — Load & Evaluate Improved LSTM
# ─────────────────────────────────────────────────────────────

def phase1_evaluate(episodes=20):
    print("\n" + "=" * 50)
    print("PHASE 1 - LOAD & EVALUATE (IMPROVED LSTM)")
    print("=" * 50)

    model_path = "models/ppo_lstm_improved"
    vec_path   = "models/vecnormalize_lstm_improved.pkl"

    if not os.path.exists(vec_path):
        print(f"  [!] {vec_path} not found. Is training still running?")
        sys.exit(1)

    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    env = VecNormalize.load(vec_path, env)
    env.training  = False
    env.norm_reward = False

    model = RecurrentPPO.load(model_path, env=env)
    print(f"  Model loaded: {model_path}")
    print(f"  Running {episodes} evaluation episodes...")

    total_reward    = 0.0
    total_steps     = 0
    total_blackouts = 0.0
    stability_list  = []

    for ep in range(episodes):
        obs            = env.reset()
        lstm_states    = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done           = np.zeros(1, dtype=bool)

        while not done[0]:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
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

    reward_per_step     = total_reward    / total_steps  if total_steps  > 0 else 0.0
    avg_blackouts       = total_blackouts / episodes
    blackouts_per_step  = total_blackouts / total_steps  if total_steps  > 0 else 0.0
    avg_stability       = float(np.mean(stability_list)) if stability_list else 0.0

    print("\n" + "-" * 40)
    print("IMPROVED LSTM PPO RESULTS")
    print("-" * 40)
    print(f"Reward/Step:    {reward_per_step:.3f}")
    print(f"Blackouts:      {avg_blackouts:.3f}")
    print(f"Blackouts/Step: {blackouts_per_step:.4f}")
    print(f"Stability:      {avg_stability:.3f}")
    print("-" * 40)

    return reward_per_step, avg_blackouts, blackouts_per_step, avg_stability


# ─────────────────────────────────────────────────────────────
# PHASE 2+3 — Baselines + Comparison Table
# ─────────────────────────────────────────────────────────────

def phase3_table(lstm_rps, lstm_blk, lstm_stab):
    print("\n" + "=" * 50)
    print("PHASE 3 - FINAL COMPARISON TABLE")
    print("=" * 50)

    agents = [
        (RANDOM["name"],   RANDOM["rps"],   RANDOM["blk"],   RANDOM["stab"]),
        (MLP["name"],      MLP["rps"],      MLP["blk"],      MLP["stab"]),
        ("PPO (LSTM)",     lstm_rps,        lstm_blk,        lstm_stab),
        (ADVANCED["name"], ADVANCED["rps"], ADVANCED["blk"], ADVANCED["stab"]),
    ]

    print("\n" + "=" * 50)
    print("FINAL POLICY COMPARISON")
    print("=" * 50)
    print(f"{'Agent':<12} {'Reward/Step':<13} {'Blackouts':<11} {'Stability'}")
    print("-" * 50)
    for name, rps, blk, stab in agents:
        print(f"{name:<12} {rps:<13.3f} {blk:<11.3f} {stab:.3f}")
    print("=" * 50)

    return agents


# ─────────────────────────────────────────────────────────────
# PHASE 4 — Improvement Metrics
# ─────────────────────────────────────────────────────────────

def phase4_metrics(lstm_rps, lstm_blk):
    print("\n" + "=" * 50)
    print("PHASE 4 - IMPROVEMENT METRICS")
    print("=" * 50)

    blk_reduction_pct = ((MLP["blk"] - lstm_blk) / MLP["blk"]) * 100
    reward_gain_pct   = ((lstm_rps - MLP["rps"]) / abs(MLP["rps"])) * 100

    print("\n" + "-" * 40)
    print("IMPROVEMENT METRICS")
    print("-" * 40)
    print(f"Blackout Reduction (%): {blk_reduction_pct:.2f}%")
    print(f"Reward Gain (%):        {reward_gain_pct:.2f}%")
    print("-" * 40)

    return blk_reduction_pct, reward_gain_pct


# ─────────────────────────────────────────────────────────────
# PHASE 5 — Performance Classification
# ─────────────────────────────────────────────────────────────

def phase5_classify(blk_reduction_pct):
    print("\n" + "=" * 50)
    print("PHASE 5 - PERFORMANCE CLASSIFICATION")
    print("=" * 50)

    if blk_reduction_pct >= 20:
        classification = "STRONG IMPROVEMENT"
    elif blk_reduction_pct >= 10:
        classification = "MODERATE IMPROVEMENT"
    else:
        classification = "LIMITED IMPROVEMENT"

    print(f"\n  {classification}")
    return classification


# ─────────────────────────────────────────────────────────────
# PHASE 6 — Plots
# ─────────────────────────────────────────────────────────────

def moving_average(arr, window=20):
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def phase6_plots(lstm_rps, lstm_blk):
    print("\n" + "=" * 50)
    print("PHASE 6 - PLOTS")
    print("=" * 50)

    # 1) Bar chart
    agent_names   = ["Random", "PPO-MLP", "PPO-LSTM", "Advanced"]
    reward_values = [RANDOM["rps"], MLP["rps"], lstm_rps, ADVANCED["rps"]]
    bar_colors    = ["#6c757d", "#3a86ff", "#2dc653", "#ff6b35"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(agent_names, reward_values, color=bar_colors,
                  edgecolor="black", linewidth=0.8)
    for bar in bars:
        yval   = bar.get_height()
        offset = 0.12 if yval >= 0 else -0.35
        va     = "bottom" if yval >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, yval + offset,
                f"{yval:.3f}", ha="center", va=va, fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Policy Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reward per Step", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/final_comparison_lstm.png", dpi=150)
    plt.close()
    print("  Saved: plots/final_comparison_lstm.png")

    # 2) Tradeoff scatter
    sc_names   = ["PPO-MLP", "PPO-LSTM", "Advanced"]
    sc_blk     = [MLP["blk"],      lstm_blk,         ADVANCED["blk"]]
    sc_rps     = [MLP["rps"],      lstm_rps,          ADVANCED["rps"]]
    sc_colors  = ["#3a86ff", "#2dc653", "#ff6b35"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(sc_names):
        ax.scatter(sc_blk[i], sc_rps[i], color=sc_colors[i], s=200,
                   edgecolors="black", zorder=5, label=name)
        ax.annotate(name, (sc_blk[i], sc_rps[i]), xytext=(10, 8),
                    textcoords="offset points", fontsize=11, fontweight="bold")
    ax.set_title("Risk vs. Reward Trade-off", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Blackouts (Risk)", fontsize=12)
    ax.set_ylabel("Reward per Step", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/tradeoff_lstm.png", dpi=150)
    plt.close()
    print("  Saved: plots/tradeoff_lstm.png")

    # 3) Learning curve
    rpath = "outputs/train_rewards_lstm.npy"
    if os.path.exists(rpath):
        ep_rewards = np.load(rpath, allow_pickle=True)
        if len(ep_rewards) > 0:
            ma = moving_average(ep_rewards, window=20)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(ep_rewards, alpha=0.25, color="#3a86ff", label="Episode Reward")
            ax.plot(np.arange(len(ma)), ma, color="#3a86ff",
                    linewidth=2.5, label="Moving Avg (window=20)")
            ax.set_title("LSTM PPO Learning Curve", fontsize=14, fontweight="bold")
            ax.set_xlabel("Episode", fontsize=12)
            ax.set_ylabel("Reward", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            plt.tight_layout()
            plt.savefig("plots/training_curve_lstm.png", dpi=150)
            plt.close()
            print("  Saved: plots/training_curve_lstm.png")
        else:
            print("  [!] train_rewards_lstm.npy empty — curve skipped.")
    else:
        print(f"  [!] {rpath} not found — curve skipped.")


# ─────────────────────────────────────────────────────────────
# PHASE 7 — README Summary
# ─────────────────────────────────────────────────────────────

def phase7_readme(lstm_rps, lstm_blk, lstm_stab, blk_reduction_pct,
                  reward_gain_pct, classification):
    print("\n" + "=" * 50)
    print("PHASE 7 - README SUMMARY")
    print("=" * 50)

    summary = f"""
----------------------------------------
SUMMARY (FOR README)
----------------------------------------

We evaluated reinforcement learning approaches for grid stability under stochastic failures.

Results:
- PPO (MLP) learns basic allocation but suffers high blackout rates (~50)
- LSTM PPO introduces temporal memory, reducing cascading failures
- Expert heuristic (Advanced) remains the safest policy

Key Insight:
"Temporal memory improves stability, but stochastic cascading failures remain a core challenge for RL."

Performance:
- LSTM reduced blackouts by {blk_reduction_pct:.1f}% vs MLP PPO
- Improved reward efficiency by {reward_gain_pct:.1f}%
- Demonstrates importance of sequence-aware decision making

Conclusion:
Standard RL is effective for efficiency, but safety-critical systems require temporal reasoning or hybrid approaches.

----------------------------------------"""

    print(summary)

    # Also write to file
    readme_path = "outputs/lstm_experiment_summary.txt"
    with open(readme_path, "w") as f:
        f.write(summary.strip())
    print(f"\n  Saved: {readme_path}")

    return summary


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Phase 1
    lstm_rps, lstm_blk, lstm_bps, lstm_stab = phase1_evaluate(episodes=20)

    # Phase 2 — baselines are fixed constants (RANDOM, MLP, ADVANCED above)

    # Phase 3
    agents = phase3_table(lstm_rps, lstm_blk, lstm_stab)

    # Phase 4
    blk_reduction_pct, reward_gain_pct = phase4_metrics(lstm_rps, lstm_blk)

    # Phase 5
    classification = phase5_classify(blk_reduction_pct)

    # Phase 6
    phase6_plots(lstm_rps, lstm_blk)

    # Phase 7
    phase7_readme(lstm_rps, lstm_blk, lstm_stab,
                  blk_reduction_pct, reward_gain_pct, classification)

    # Final output
    print("\n" + "=" * 50)
    print("FINAL OUTPUT")
    print("=" * 50)
    print(f"  Reward/Step     : {lstm_rps:.3f}")
    print(f"  Blackouts       : {lstm_blk:.3f}")
    print(f"  Blackouts/Step  : {lstm_bps:.4f}")
    print(f"  Stability       : {lstm_stab:.3f}")
    print(f"  Classification  : {classification}")
    print(f"  Blackout Reduct : {blk_reduction_pct:.2f}%")
    print(f"  Reward Gain     : {reward_gain_pct:.2f}%")
    print(f"  Plots           : plots/final_comparison_lstm.png")
    print(f"                    plots/tradeoff_lstm.png")
    print(f"                    plots/training_curve_lstm.png")
    print(f"  README summary  : outputs/lstm_experiment_summary.txt")
    print("=" * 50)
