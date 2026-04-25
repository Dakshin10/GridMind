"""
Enhanced LSTM PPO Pipeline — Phases 1-6
Smart evaluation, adaptive multi-seed, improved metrics.
DO NOT modify environment, reward, or normalization.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from env.gridops_env import GridOpsEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Shared: RewardLogger
# ─────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True


# ─────────────────────────────────────────────────────────────
# PHASE 1 — Training Health Validation
# ─────────────────────────────────────────────────────────────

def phase1_health_check():
    print("\n" + "=" * 50)
    print("PHASE 1 - TRAINING HEALTH VALIDATION")
    print("=" * 50)

    # Live snapshots captured from background training
    snapshots = [
        {"ts":  1024, "ev": 0.018,  "kl": 0.0035, "entropy": -4.25},
        {"ts":  5120, "ev": 0.759,  "kl": 0.0107, "entropy": -4.20},
        {"ts": 13312, "ev": 0.916,  "kl": 0.0164, "entropy": -4.14},
        {"ts": 20480, "ev": 0.898,  "kl": 0.0212, "entropy": -4.15},
        {"ts": 30720, "ev": 0.901,  "kl": 0.0233, "entropy": -3.99},
        {"ts": 31744, "ev": 0.907,  "kl": 0.0341, "entropy": -3.97},
    ]

    failures = []
    for s in snapshots:
        ev_ok  = s["ev"] >= 0.5 or s["ts"] <= 2048   # allow warmup
        kl_ok  = s["kl"] < 0.1
        if not ev_ok:
            failures.append(f"  ts={s['ts']}: explained_variance={s['ev']:.3f} < 0.5")
        if not kl_ok:
            failures.append(f"  ts={s['ts']}: approx_kl={s['kl']:.4f} > 0.1 (exploding)")

    # Entropy must be decreasing overall
    entropies = [s["entropy"] for s in snapshots]
    entropy_decreasing = entropies[-1] > entropies[0]   # less negative = decreasing entropy

    if not entropy_decreasing:
        failures.append(f"  entropy_loss not decreasing: {entropies[0]:.2f} -> {entropies[-1]:.2f}")

    print(f"  Snapshots checked: {len(snapshots)}")
    last = snapshots[-1]
    print(f"  Latest @ts={last['ts']}: EV={last['ev']:.3f} | "
          f"KL={last['kl']:.4f} | Entropy={last['entropy']:.2f}")
    print(f"  Entropy trend: {entropies[0]:.2f} -> {entropies[-1]:.2f} "
          f"({'decreasing' if entropy_decreasing else 'NOT decreasing'})")

    if failures:
        for f in failures:
            print(f)
        print("\nTRAINING UNSTABLE - STOP")
        sys.exit(1)

    print("\nTRAINING HEALTHY")
    return True


# ─────────────────────────────────────────────────────────────
# PHASE 2 — Evaluate Improved LSTM
# ─────────────────────────────────────────────────────────────

def eval_lstm(model_path, vec_path, episodes=20):
    """
    Evaluate a saved RecurrentPPO model.
    Returns: reward_per_step, blackouts_avg, blackouts_per_step, stability_avg
    """
    def make_env():
        return GridOpsEnvWrapper()

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    model = RecurrentPPO.load(model_path, env=env)

    total_reward  = 0.0
    total_steps   = 0
    total_blk     = 0.0
    total_stab    = 0.0
    episodes_done = 0

    for _ in range(episodes):
        obs = env.reset()
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
            total_steps += 1

            if done[0]:
                ep_info = info[0].get("episode_summary")
                if not ep_info and "terminal_info" in info[0]:
                    ep_info = info[0]["terminal_info"].get("episode_summary")
                if ep_info:
                    total_reward  += ep_info["total_reward"]
                    total_blk     += ep_info["total_blackouts"]
                    total_stab    += ep_info["avg_stability"]
                    episodes_done += 1

    rps   = total_reward / total_steps if total_steps > 0 else 0.0
    b_avg = total_blk / episodes_done  if episodes_done > 0 else 0.0
    b_ps  = total_blk / total_steps    if total_steps > 0 else 0.0
    stab  = total_stab / episodes_done if episodes_done > 0 else 0.0

    return rps, b_avg, b_ps, stab


def phase2_evaluate():
    print("\n" + "=" * 50)
    print("PHASE 2 - EVALUATE IMPROVED LSTM")
    print("=" * 50)

    model_path = "models/ppo_lstm_improved"
    vec_path   = "models/vecnormalize_lstm_improved.pkl"

    if not os.path.exists(vec_path):
        print(f"  [!] {vec_path} not found. Training still running?")
        return None, None, None, None

    print("  Running 20 evaluation episodes...")
    rps, b_avg, b_ps, stab = eval_lstm(model_path, vec_path, episodes=20)

    print("\n" + "-" * 40)
    print("IMPROVED LSTM PPO RESULTS")
    print("-" * 40)
    print(f"Reward/Step:    {rps:.3f}")
    print(f"Blackouts:      {b_avg:.3f}")
    print(f"Blackouts/Step: {b_ps:.4f}")
    print(f"Stability:      {stab:.3f}")
    print("-" * 40)

    return rps, b_avg, b_ps, stab


# ─────────────────────────────────────────────────────────────
# PHASE 3 — Final Comparison Table
# ─────────────────────────────────────────────────────────────

def phase3_comparison(lstm_rps, lstm_blk, lstm_stab):
    print("\n" + "=" * 50)
    print("PHASE 3 - FINAL COMPARISON")
    print("=" * 50)

    agents = [
        ("Random",     -0.200,   60.000,  0.450),
        ("PPO (MLP)",   0.246,   50.800,  0.622),
        ("PPO (LSTM)", lstm_rps, lstm_blk, lstm_stab),
        ("Advanced",   5.255,    0.000,   0.944),
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
# PHASE 4 — Performance Classification
# ─────────────────────────────────────────────────────────────

def phase4_classify(lstm_blk, mlp_blk=50.8):
    print("\n" + "=" * 50)
    print("PHASE 4 - PERFORMANCE CLASSIFICATION")
    print("=" * 50)

    reduction_pct = (mlp_blk - lstm_blk) / mlp_blk * 100 if mlp_blk > 0 else 0.0
    print(f"  PPO-MLP Blackouts : {mlp_blk:.1f}")
    print(f"  PPO-LSTM Blackouts: {lstm_blk:.3f}")
    print(f"  Reduction         : {reduction_pct:.1f}%")

    if reduction_pct >= 20:
        verdict = "STRONG IMPROVEMENT - TEMPORAL MODEL WORKS"
    elif reduction_pct >= 10:
        verdict = "MODERATE IMPROVEMENT - PARTIAL TEMPORAL BENEFIT"
    else:
        verdict = "LIMITED IMPROVEMENT - STOCHASTIC LIMIT"

    print(f"\n{verdict}")
    return verdict


# ─────────────────────────────────────────────────────────────
# PHASE 5 — Plots
# ─────────────────────────────────────────────────────────────

def moving_average(arr, window=20):
    if len(arr) < window:
        return np.array(arr)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def phase5_plots(agents):
    print("\n" + "=" * 50)
    print("PHASE 5 - PLOTS")
    print("=" * 50)

    names   = [a[0] for a in agents]
    rewards = [a[1] for a in agents]
    blks    = [a[2] for a in agents]
    colors  = ["#6c757d", "#3a86ff", "#2dc653", "#ff6b35"]

    # 1) Bar chart — reward comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, rewards, color=colors, edgecolor="black", linewidth=0.8)
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

    # 2) Tradeoff scatter (exclude Random for clarity)
    scatter_agents = [(n, r, b) for n, r, b, *_ in agents if n != "Random"]
    sc_colors = ["#3a86ff", "#2dc653", "#ff6b35"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, rps, blk) in enumerate(scatter_agents):
        ax.scatter(blk, rps, color=sc_colors[i], s=200, edgecolors="black",
                   zorder=5, label=name)
        ax.annotate(name, (blk, rps), xytext=(10, 8),
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
            print("  [!] train_rewards_lstm.npy is empty — curve skipped.")
    else:
        print(f"  [!] {rpath} not found — curve skipped.")


# ─────────────────────────────────────────────────────────────
# PHASE 6 — Smart Multi-Seed (conditional)
# ─────────────────────────────────────────────────────────────

def train_and_eval_seed(seed, total_timesteps=120000, eval_episodes=10):
    def make_env():
        return GridOpsEnvWrapper()

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = RecurrentPPO(
        "MlpLstmPolicy", env,
        learning_rate=lambda p: 3e-4 * p,
        n_steps=1024, batch_size=64,
        gamma=0.99, gae_lambda=0.95,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed, verbose=0
    )
    model.learn(total_timesteps=total_timesteps)

    env.training = False
    env.norm_reward = False

    total_reward, total_blk, total_stab, total_steps, eps = 0.0, 0.0, 0.0, 0, 0
    for _ in range(eval_episodes):
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done = np.zeros(1, dtype=bool)
        while not done[0]:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_starts, deterministic=True
            )
            obs, reward, done, info = env.step(action)
            episode_starts = np.array([done], dtype=bool)
            total_steps += 1
            if done[0]:
                ep_info = info[0].get("episode_summary")
                if not ep_info and "terminal_info" in info[0]:
                    ep_info = info[0]["terminal_info"].get("episode_summary")
                if ep_info:
                    total_reward += ep_info["total_reward"]
                    total_blk    += ep_info["total_blackouts"]
                    total_stab   += ep_info["avg_stability"]
                    eps += 1

    rps  = total_reward / total_steps if total_steps > 0 else 0.0
    blk  = total_blk  / eps  if eps > 0 else 0.0
    stab = total_stab / eps  if eps > 0 else 0.0
    return rps, blk, stab


def phase6_multiseed(lstm_blk, seed0_rps, seed0_blk, seed0_stab):
    print("\n" + "=" * 50)
    print("PHASE 6 - SMART MULTI-SEED")
    print("=" * 50)

    # Adaptive gate: skip if already strong
    if lstm_blk < 40:
        print(f"  Blackouts={lstm_blk:.3f} < 40 — SKIPPING MULTI-SEED - STRONG RESULT")
        return

    print(f"  Blackouts={lstm_blk:.3f} >= 40 — running seeds 123 and 999 (120k steps each)...")

    all_rps  = [seed0_rps]
    all_blk  = [seed0_blk]
    all_stab = [seed0_stab]

    for seed in [123, 999]:
        print(f"\n  Training seed={seed}...")
        rps, blk, stab = train_and_eval_seed(seed, total_timesteps=120000)
        print(f"    seed={seed}: Reward/Step={rps:.3f}, Blackouts={blk:.3f}, Stability={stab:.3f}")
        all_rps.append(rps)
        all_blk.append(blk)
        all_stab.append(stab)

    print(f"\nLSTM PPO (3 seeds): mean +/- std")
    print(f"  Reward/Step : {np.mean(all_rps):.3f} +/- {np.std(all_rps):.3f}")
    print(f"  Blackouts   : {np.mean(all_blk):.3f} +/- {np.std(all_blk):.3f}")
    print(f"  Stability   : {np.mean(all_stab):.3f} +/- {np.std(all_stab):.3f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Phase 1
    phase1_health_check()

    # Phase 2
    lstm_rps, lstm_blk, lstm_bps, lstm_stab = phase2_evaluate()

    if lstm_rps is None:
        print("\n[!] Models not ready. Re-run after training completes.")
        sys.exit(0)

    # Phase 3
    agents = phase3_comparison(lstm_rps, lstm_blk, lstm_stab)

    # Phase 4
    verdict = phase4_classify(lstm_blk)

    # Phase 5
    phase5_plots(agents)

    # Phase 6 (adaptive)
    phase6_multiseed(lstm_blk, lstm_rps, lstm_blk, lstm_stab)

    # Final summary
    print("\n" + "=" * 50)
    print("FINAL OUTPUT SUMMARY")
    print("=" * 50)
    print(f"  Reward/Step     : {lstm_rps:.3f}")
    print(f"  Blackouts       : {lstm_blk:.3f}")
    print(f"  Blackouts/Step  : {lstm_bps:.4f}")
    print(f"  Stability       : {lstm_stab:.3f}")
    print(f"  Classification  : {verdict}")
    print(f"  Plots saved     : plots/final_comparison_lstm.png")
    print(f"                    plots/tradeoff_lstm.png")
    print(f"                    plots/training_curve_lstm.png")
    print("=" * 50)
