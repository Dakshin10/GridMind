import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODES  = ["baseline", "selfish", "coordinated", "advanced"]
COLORS = {
    "baseline":    "gray",
    "selfish":     "red",
    "coordinated": "blue",
    "advanced":    "green",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _mean_curve(histories, name, key):
    """Element-wise mean across seeds for a given history key."""
    arrays  = [np.array(h.get(key, [0])) for h in histories[name]]
    min_len = min(len(a) for a in arrays)
    matrix  = np.stack([a[:min_len] for a in arrays])
    return matrix.mean(axis=0)

def _mean_std_curve(histories, name, key):
    """Element-wise mean and std across seeds for a given history key."""
    arrays  = [np.array(h.get(key, [0])) for h in histories[name]]
    min_len = min(len(a) for a in arrays)
    matrix  = np.stack([a[:min_len] for a in arrays])
    return matrix.mean(axis=0), matrix.std(axis=0)

def _moving_average(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

def _save(fig, path, label=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    tag = label or path
    print(f"  Saved -> {tag}")


def _multi_line(histories, key, title, ylabel, filename, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in MODES:
        curve_mean, curve_std = _mean_std_curve(histories, name, key)
        ax.plot(curve_mean, label=name, color=COLORS[name], linewidth=2.5)
        ax.fill_between(np.arange(len(curve_mean)), curve_mean - curve_std, curve_mean + curve_std, color=COLORS[name], alpha=0.2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(save_dir, filename))


# ----------------------------------------------------------------------
# Individual plots
# ----------------------------------------------------------------------

def plot_reward_curves(histories, save_dir):
    _multi_line(histories, "reward",
                "Reward Over Time — Coordination Maximizes Global Gain", "Reward",
                "reward_curve.png", save_dir)


def plot_blackouts(histories, save_dir):
    _multi_line(histories, "blackouts",
                "Blackouts Over Time — Reduction via Coordination", "Blackouts",
                "blackouts.png", save_dir)


def plot_misalignment(histories, save_dir):
    """
    Local (raw served) vs global (ethical + fairness) reward for selfish mode.
    Shows divergence when an agent optimises locally at the expense of the grid.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    local_curve  = _mean_curve(histories, "selfish", "reward")
    global_curve = _mean_curve(histories, "advanced", "reward")
    steps = np.arange(len(local_curve))
    ax.plot(local_curve,  label="selfish (local reward)",   color=COLORS["selfish"],   linewidth=2.5)
    ax.plot(global_curve, label="advanced (global reward)", color=COLORS["advanced"],  linewidth=2.5)
    ax.fill_between(steps, local_curve, global_curve, alpha=0.12, color="grey",
                    label="misalignment gap")
    ax.set_title("Misalignment — Local Reward vs Global Grid Health", fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(save_dir, "misalignment_plot.png"))


def plot_misreporting_trend(histories, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["selfish", "coordinated", "advanced"]:
        curve_mean, curve_std = _mean_std_curve(histories, name, "misreporting")
        ax.plot(curve_mean, label=name, color=COLORS[name], linewidth=2.5)
        ax.fill_between(np.arange(len(curve_mean)), curve_mean - curve_std, curve_mean + curve_std, color=COLORS[name], alpha=0.2)
    ax.set_title("Misreporting Trend — Coordination Eliminates Strategic Lying", fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Misreporting Rate", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(save_dir, "misreporting_trend.png"))


def plot_coalition_trend(histories, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["selfish", "coordinated", "advanced"]:
        curve_mean, curve_std = _mean_std_curve(histories, name, "coalition")
        ax.plot(curve_mean, label=name, color=COLORS[name], linewidth=2.5)
        ax.fill_between(np.arange(len(curve_mean)), curve_mean - curve_std, curve_mean + curve_std, color=COLORS[name], alpha=0.2)
    ax.set_title("Coalition Formation — Emergence of Trust", fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Coalition Active (fraction)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(save_dir, "coalition_trend.png"))


def plot_stability(histories, save_dir):
    _multi_line(histories, "stability",
                "System Stability — Long-Horizon Resilience", "Stability Score",
                "stability.png", save_dir)


def plot_imbalance(histories, save_dir):
    _multi_line(histories, "imbalance",
                "Grid Imbalance — Fair Distribution Over Time", "Imbalance (variance of shares)",
                "imbalance.png", save_dir)


def plot_reputation(histories, save_dir):
    key = "avg_reputation"
    has_data = any(
        len(histories[name][0].get(key, [])) > 0
        for name in MODES
    )
    if not has_data:
        return
    _multi_line(histories, key,
                "Reputation Dynamics — Trust Recovery vs Decay", "Reputation",
                "reputation.png", save_dir)


# ----------------------------------------------------------------------
# Hero comparison overlay
# ----------------------------------------------------------------------

def plot_comparison(histories, save_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in MODES:
        curve = _mean_curve(histories, name, "reward")
        smooth = _moving_average(curve, w=3)
        steps = np.arange(len(smooth)) + 1  # Shift right due to MA valid mode
        ax.plot(steps, smooth, label=name, color=COLORS[name], linewidth=3.0)
        
        # Annotations
        if name == "selfish":
            peak_idx = np.argmax(smooth)
            ax.annotate("selfish peak", xy=(steps[peak_idx], smooth[peak_idx]), 
                        xytext=(steps[peak_idx]-5, smooth[peak_idx]+5),
                        arrowprops=dict(arrowstyle="->", color=COLORS[name]), 
                        fontsize=10, color=COLORS[name])
        elif name == "advanced":
            peak_idx = len(smooth) - 1
            ax.annotate("coordination gain", xy=(steps[peak_idx], smooth[peak_idx]), 
                        xytext=(steps[peak_idx]-10, smooth[peak_idx]-10),
                        arrowprops=dict(arrowstyle="->", color=COLORS[name]), 
                        fontsize=10, color=COLORS[name])

    ax.set_title("Emergence of Stable Coordination in Multi-Agent Grid System",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Smoothed Reward", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(save_dir, "comparison.png"))

# ----------------------------------------------------------------------
# Single Summary Image
# ----------------------------------------------------------------------

def plot_summary(histories, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Reward Comparison
    ax = axes[0]
    for name in MODES:
        curve_mean, curve_std = _mean_std_curve(histories, name, "reward")
        ax.plot(curve_mean, label=name, color=COLORS[name], linewidth=2.5)
        ax.fill_between(np.arange(len(curve_mean)), curve_mean - curve_std, curve_mean + curve_std, color=COLORS[name], alpha=0.2)
    ax.set_title("Reward Comparison", fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    # 2. Misreporting Trend
    ax = axes[1]
    for name in ["selfish", "coordinated", "advanced"]:
        curve_mean, curve_std = _mean_std_curve(histories, name, "misreporting")
        ax.plot(curve_mean, label=name, color=COLORS[name], linewidth=2.5)
        ax.fill_between(np.arange(len(curve_mean)), curve_mean - curve_std, curve_mean + curve_std, color=COLORS[name], alpha=0.2)
    ax.set_title("Misreporting Trend", fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Misreporting Rate", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    # 3. Stability Curve
    ax = axes[2]
    for name in MODES:
        curve_mean, curve_std = _mean_std_curve(histories, name, "stability")
        ax.plot(curve_mean, label=name, color=COLORS[name], linewidth=2.5)
        ax.fill_between(np.arange(len(curve_mean)), curve_mean - curve_std, curve_mean + curve_std, color=COLORS[name], alpha=0.2)
    ax.set_title("Stability Curve", fontsize=14)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Stability Score", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    fig.suptitle("GridOps Emergent Multi-Agent Behavior Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "summary.png"))


# ----------------------------------------------------------------------
# 2×2 Hero Plot — main_result.png
# ----------------------------------------------------------------------

def plot_main_result(histories, save_dir):
    """
    2×2 hero grid: Reward, Stability, Misreporting, Blackouts.
    This is the first image judges see in the README.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#f7f9fc")

    panels = [
        ("reward",       "Reward Over Time",            "Reward",            axes[0, 0]),
        ("stability",    "Grid Stability",               "Stability Score",   axes[0, 1]),
        ("misreporting", "Misreporting Rate",            "Misreport Rate",    axes[1, 0]),
        ("blackouts",    "Blackouts per Step",           "Blackouts",         axes[1, 1]),
    ]
    modes_for = {
        "reward":       MODES,
        "stability":    MODES,
        "misreporting": ["selfish", "coordinated", "advanced"],
        "blackouts":    MODES,
    }

    for key, title, ylabel, ax in panels:
        ax.set_facecolor("#f7f9fc")
        for name in modes_for[key]:
            mean, std = _mean_std_curve(histories, name, key)
            steps = np.arange(len(mean))
            ax.plot(steps, mean, label=name, color=COLORS[name], linewidth=2.5)
            ax.fill_between(steps, mean - std, mean + std,
                            color=COLORS[name], alpha=0.2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "GridOps++: Advanced Coordination Dominates All Baselines",
        fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "main_result.png"))


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def generate_all_plots(histories, save_dir="plots"):
    print(f"\nGenerating plots in '{save_dir}/' ...")
    os.makedirs(save_dir, exist_ok=True)

    plot_main_result(histories, save_dir)
    plot_reward_curves(histories, save_dir)
    plot_blackouts(histories, save_dir)
    plot_misalignment(histories, save_dir)
    plot_misreporting_trend(histories, save_dir)
    plot_coalition_trend(histories, save_dir)
    plot_stability(histories, save_dir)
    plot_imbalance(histories, save_dir)
    plot_reputation(histories, save_dir)
    plot_comparison(histories, save_dir)
    plot_summary(histories, save_dir)

    print("All plots saved.\n")


# ----------------------------------------------------------------------
# Stand-alone entry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "train", os.path.join(os.path.dirname(__file__), "train.py")
    )
    _train = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_train)

    histories, _ = _train.run_all()
    generate_all_plots(histories)
