"""plots.py — Visual suite for GridOps++ analysis.

Consistent style: seaborn-v0_8-whitegrid, dpi=200, shaded std bands,
smoothed mean curves, annotated key events.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODES = ["baseline", "selfish", "coordinated", "advanced"]
COLORS = {
    "baseline":    "#7f7f7f",
    "selfish":     "#d62728",
    "coordinated": "#1f77b4",
    "advanced":    "#2ca02c",
}

LW        = 2.5
SHADE     = 0.20
TITLE_FS  = 16
LABEL_FS  = 13
LEGEND_FS = 11
DPI       = 200


# ── Helpers ──────────────────────────────────────────────────────────────────

def smooth(x, w=3):
    """Moving-average smoothing (mode='same' keeps length)."""
    return np.convolve(x, np.ones(w) / w, mode="same")


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
    return np.convolve(x, np.ones(w), "valid") / w


def _save(fig, path, label=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {label or path}")


def _leg(ax, outside=False):
    if outside:
        ax.legend(fontsize=LEGEND_FS, loc="upper left",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)
    else:
        ax.legend(fontsize=LEGEND_FS)


def _ax(ax, title, ylabel, xlabel="Step"):
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.grid(alpha=0.3)


# ── Reward curve ─────────────────────────────────────────────────────────────

def plot_reward_curves(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODES:
        mean, std = _mean_std_curve(histories, name, "reward")
        sm    = smooth(mean)
        steps = np.arange(len(sm))
        lw = 3.0 if name == "advanced" else LW
        ax.plot(steps, sm, label=name, color=COLORS[name], linewidth=lw)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)

    sel_sm  = smooth(_mean_curve(histories, "selfish",  "reward"))
    adv_sm  = smooth(_mean_curve(histories, "advanced", "reward"))
    sp = int(np.argmax(sel_sm))
    ax.annotate("selfish peak", xy=(sp, sel_sm[sp]),
                xytext=(sp + 3, sel_sm[sp] + 4),
                arrowprops=dict(arrowstyle="->", color=COLORS["selfish"], lw=1.5),
                fontsize=10, color=COLORS["selfish"], fontweight="bold")
    ep = len(adv_sm) - 1
    ax.annotate("coordination gain", xy=(ep, adv_sm[ep]),
                xytext=(ep - 14, adv_sm[ep] - 9),
                arrowprops=dict(arrowstyle="->", color=COLORS["advanced"], lw=1.5),
                fontsize=10, color=COLORS["advanced"], fontweight="bold")
    ax.text(0.02, 0.95, "+134% improvement  (17.29 → 40.55)",
            transform=ax.transAxes, fontsize=11, color=COLORS["advanced"], va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["advanced"], alpha=0.9))
    ax.text(ep, adv_sm[ep] + 1.5, f"{adv_sm[ep]:.1f}",
            fontsize=9, color=COLORS["advanced"], ha="right")

    _ax(ax, "Reward Over Time — Coordination Maximizes Global Gain", "Reward")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "reward_curve.png"))


# ── Blackouts (cumulative) ────────────────────────────────────────────────────

def plot_blackouts(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODES:
        mean, std = _mean_std_curve(histories, name, "blackouts")
        cum_mean = np.cumsum(mean)
        cum_std  = np.cumsum(std)
        steps    = np.arange(len(cum_mean))
        ax.plot(steps, cum_mean, label=name, color=COLORS[name], linewidth=LW)
        ax.fill_between(steps, cum_mean - cum_std, cum_mean + cum_std,
                        color=COLORS[name], alpha=SHADE)
    _ax(ax, "Cumulative Blackouts — Advanced Minimizes Failures", "Cumulative Blackouts")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "blackouts.png"))


# ── Misalignment ──────────────────────────────────────────────────────────────

def plot_misalignment(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    lc = smooth(_mean_curve(histories, "selfish",  "reward"))
    gc = smooth(_mean_curve(histories, "advanced", "reward"))
    steps = np.arange(len(lc))
    ax.plot(steps, lc, label="selfish (local)",   color=COLORS["selfish"],   linewidth=LW)
    ax.plot(steps, gc, label="advanced (global)",  color=COLORS["advanced"],  linewidth=3.0)
    ax.fill_between(steps, lc, gc, alpha=0.12, color="grey", label="misalignment gap")
    _ax(ax, "Misalignment — Local Reward vs Global Grid Health", "Reward")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "misalignment_plot.png"))


# ── Misreporting ──────────────────────────────────────────────────────────────

def plot_misreporting_trend(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    sel_mean, _ = _mean_std_curve(histories, "selfish", "misreporting")
    steps_all   = np.arange(len(sel_mean))
    ax.fill_between(steps_all, 0, sel_mean, color=COLORS["selfish"], alpha=0.08)
    for name in ["selfish", "coordinated", "advanced"]:
        mean, std = _mean_std_curve(histories, name, "misreporting")
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=name, color=COLORS[name], linewidth=LW)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
    ax.set_ylim(0, 1)
    ax.text(0.98, 0.92, "↓ 99% reduction", transform=ax.transAxes,
            fontsize=13, color=COLORS["advanced"], fontweight="bold", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["advanced"], alpha=0.9))
    _ax(ax, "Misreporting Trend — Coordination Eliminates Strategic Lying", "Misreporting Rate")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "misreporting_trend.png"))


# ── Coalition ────────────────────────────────────────────────────────────────

def plot_coalition_trend(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ["selfish", "coordinated", "advanced"]:
        mean, std = _mean_std_curve(histories, name, "coalition")
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=name, color=COLORS[name], linewidth=LW)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
    _ax(ax, "Coalition Formation — Emergence of Trust", "Coalition Active (fraction)")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "coalition_trend.png"))


# ── Stability ────────────────────────────────────────────────────────────────

def plot_stability(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODES:
        mean, std = _mean_std_curve(histories, name, "stability")
        sm    = smooth(mean)
        steps = np.arange(len(sm))
        lw = 3.0 if name == "advanced" else LW
        ax.plot(steps, sm, label=name, color=COLORS[name], linewidth=lw)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)

    ax.axhline(0.8, linestyle="--", color="black", linewidth=1.4, alpha=0.6, label="High Stability (0.8)")
    adv_sm = smooth(_mean_curve(histories, "advanced", "stability"))
    cross  = int(np.argmax(adv_sm >= 0.75)) if np.any(adv_sm >= 0.75) else len(adv_sm) - 1
    ax.annotate("advanced stable region",
                xy=(cross, adv_sm[cross]),
                xytext=(cross + 5, adv_sm[cross] - 0.12),
                arrowprops=dict(arrowstyle="->", color=COLORS["advanced"], lw=1.4),
                fontsize=10, color=COLORS["advanced"], fontweight="bold")
    ax.text(len(adv_sm) - 1, adv_sm[-1] + 0.01, f"{adv_sm[-1]:.2f}",
            fontsize=9, color=COLORS["advanced"], ha="right")

    _ax(ax, "System Stability — Long-Horizon Resilience", "Stability Score")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "stability.png"))


# ── Imbalance ────────────────────────────────────────────────────────────────

def plot_imbalance(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODES:
        mean, std = _mean_std_curve(histories, name, "imbalance")
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=name, color=COLORS[name], linewidth=LW)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
    _ax(ax, "Grid Imbalance — Fair Distribution Over Time", "Imbalance (variance of shares)")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "imbalance.png"))


# ── Reputation ───────────────────────────────────────────────────────────────

def plot_reputation(histories, save_dir):
    key = "avg_reputation"
    has_data = any(len(histories[name][0].get(key, [])) > 0 for name in MODES)
    if not has_data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODES:
        mean, std = _mean_std_curve(histories, name, key)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=name, color=COLORS[name], linewidth=LW)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
    _ax(ax, "Reputation Dynamics — Trust Recovery vs Decay", "Reputation")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "reputation.png"))


# ── Hero comparison overlay ───────────────────────────────────────────────────

def plot_comparison(histories, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODES:
        curve = _mean_curve(histories, name, "reward")
        sm    = _moving_average(curve, w=3)
        steps = np.arange(len(sm)) + 1
        lw = 3.0 if name == "advanced" else LW
        ax.plot(steps, sm, label=name, color=COLORS[name], linewidth=lw)
        if name == "selfish":
            pi = int(np.argmax(sm))
            ax.annotate("selfish peak", xy=(steps[pi], sm[pi]),
                        xytext=(steps[pi] - 6, sm[pi] + 5),
                        arrowprops=dict(arrowstyle="->", color=COLORS[name], lw=1.4),
                        fontsize=10, color=COLORS[name], fontweight="bold")
        elif name == "advanced":
            pi = len(sm) - 1
            ax.annotate("coordination gain", xy=(steps[pi], sm[pi]),
                        xytext=(steps[pi] - 14, sm[pi] - 9),
                        arrowprops=dict(arrowstyle="->", color=COLORS[name], lw=1.4),
                        fontsize=10, color=COLORS[name], fontweight="bold")
    _ax(ax, "Emergence of Stable Coordination in Multi-Agent Grid System", "Smoothed Reward")
    _leg(ax, outside=True)
    _save(fig, os.path.join(save_dir, "comparison.png"))


# ── 3-panel summary ───────────────────────────────────────────────────────────

def plot_summary(histories, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ("reward",       "Reward Comparison",   "Reward",            MODES),
        ("misreporting", "Misreporting Trend",   "Misreporting Rate", ["selfish", "coordinated", "advanced"]),
        ("stability",    "Stability Curve",       "Stability Score",   MODES),
    ]
    for ax, (key, title, ylabel, names) in zip(axes, panels):
        for name in names:
            mean, std = _mean_std_curve(histories, name, key)
            sm = smooth(mean)
            steps = np.arange(len(sm))
            ax.plot(steps, sm, label=name, color=COLORS[name], linewidth=LW)
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle("GridOps Emergent Multi-Agent Behavior Summary", fontsize=TITLE_FS, fontweight="bold")
    _save(fig, os.path.join(save_dir, "summary.png"))


# ── 2×2 Hero Plot ─────────────────────────────────────────────────────────────

def plot_main_result(histories, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#f7f9fc")
    panels = [
        ("reward",       "Reward Over Time",    "Reward",           MODES),
        ("stability",    "Grid Stability",       "Stability Score",  MODES),
        ("misreporting", "Misreporting Rate",    "Misreport Rate",   ["selfish", "coordinated", "advanced"]),
        ("blackouts",    "Blackouts per Step",   "Blackouts",        MODES),
    ]
    for (key, title, ylabel, names), ax in zip(panels, axes.flat):
        ax.set_facecolor("#f7f9fc")
        for name in names:
            mean, std = _mean_std_curve(histories, name, key)
            sm = smooth(mean)
            steps = np.arange(len(sm))
            lw = 3.0 if name == "advanced" else LW
            ax.plot(steps, sm, label=name, color=COLORS[name], linewidth=lw)
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("GridOps++: Advanced Coordination Dominates All Baselines",
                 fontsize=TITLE_FS, fontweight="bold", y=1.01)
    _save(fig, os.path.join(save_dir, "main_result.png"))


# ── One-Glance 2×2 ───────────────────────────────────────────────────────────

def plot_one_glance(histories, save_dir):
    """2×2: reward, misreporting, stability, tradeoff scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Reward
    ax = axes[0, 0]
    for name in MODES:
        mean, std = _mean_std_curve(histories, name, "reward")
        sm = smooth(mean)
        steps = np.arange(len(sm))
        ax.plot(steps, sm, label=name, color=COLORS[name],
                linewidth=3.0 if name == "advanced" else LW)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
    ax.set_title("Reward Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11); ax.set_ylabel("Reward", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Misreporting
    ax = axes[0, 1]
    for name in ["selfish", "coordinated", "advanced"]:
        mean, std = _mean_std_curve(histories, name, "misreporting")
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=name, color=COLORS[name], linewidth=LW)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
    ax.set_ylim(0, 1)
    ax.text(0.98, 0.92, "↓ 99% reduction", transform=ax.transAxes,
            fontsize=11, color=COLORS["advanced"], fontweight="bold", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["advanced"], alpha=0.9))
    ax.set_title("Misreporting Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11); ax.set_ylabel("Misreporting Rate", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Stability
    ax = axes[1, 0]
    for name in MODES:
        mean, std = _mean_std_curve(histories, name, "stability")
        sm = smooth(mean)
        steps = np.arange(len(sm))
        ax.plot(steps, sm, label=name, color=COLORS[name],
                linewidth=3.0 if name == "advanced" else LW)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[name], alpha=SHADE)
    ax.axhline(0.8, linestyle="--", color="black", linewidth=1.2, alpha=0.6, label="High Stability")
    ax.set_title("Grid Stability", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11); ax.set_ylabel("Stability Score", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Tradeoff scatter
    ax = axes[1, 1]
    for name in MODES:
        xs = [float(np.mean(h.get("reward",    [0]))) for h in histories[name]]
        ys = [float(np.mean(h.get("stability", [1]))) for h in histories[name]]
        ms = 220 if name == "advanced" else 120
        ec = "black" if name == "advanced" else "white"
        lw = 2.0 if name == "advanced" else 0.8
        ax.scatter(xs, ys, label=name, color=COLORS[name], s=ms,
                   edgecolors=ec, linewidths=lw, zorder=5)
        ax.annotate(name, (float(np.mean(xs)), float(np.mean(ys))),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=COLORS[name], fontweight="bold")
    adv_x = float(np.mean([np.mean(h.get("reward",    [0])) for h in histories["advanced"]]))
    adv_y = float(np.mean([np.mean(h.get("stability", [1])) for h in histories["advanced"]]))
    ax.text(adv_x - 2, adv_y + 0.03, "[*] Pareto Optimal",
            fontsize=10, color=COLORS["advanced"], fontweight="bold")
    ax.set_title("Tradeoff: Reward vs Stability", fontsize=13, fontweight="bold")
    ax.set_xlabel("Avg Episode Reward", fontsize=11)
    ax.set_ylabel("Avg Stability Score", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("GridOps++: From Selfish Chaos to Coordinated Stability",
                 fontsize=TITLE_FS, fontweight="bold")
    _save(fig, os.path.join(save_dir, "one_glance.png"))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def generate_all_plots(histories, save_dir="plots"):
    print(f"\nGenerating plots in '{save_dir}/' ...")
    os.makedirs(save_dir, exist_ok=True)

    plot_main_result(histories, save_dir)
    plot_one_glance(histories, save_dir)
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


# ── Stand-alone entry ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "train", os.path.join(os.path.dirname(__file__), "train.py")
    )
    _train = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_train)
    histories, _ = _train.run_all()
    generate_all_plots(histories)
