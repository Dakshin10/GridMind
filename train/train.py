import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.gridops_env import GridOpsEnv


# ----------------------------------------------------------------------
# Configurations  (name, mode, reward_mode)
# "advanced" is coordinated + global — demonstrating the full v2 stack
# ----------------------------------------------------------------------

CONFIGS = [
    ("baseline",    "baseline",    "local"),
    ("selfish",     "selfish",     "local"),
    ("coordinated", "coordinated", "global"),
    ("advanced",    "advanced",    "global"),   # rep², coalition bonus 3.0, 2× honesty pen
]

SEEDS = [0, 1, 2]


# ----------------------------------------------------------------------
# Episode runner
# ----------------------------------------------------------------------

def run_episode(env, demo_log=False):
    """
    Run a full episode using the env's internal mode policy.
    Returns (history_dict, summary_dict).
    Extra metrics introduced in v2 (misreporting_rate, coalition_rate,
    delayed_failures_triggered) are pulled from info each step and stored
    so plots.py can access them as mean curves.
    """
    obs, _ = env.reset()
    done   = False

    # Extend history with v2 keys if not present
    for key in ("misreporting", "coalition", "delayed_failures"):
        if key not in env.history:
            env.history[key] = []

    while not done:
        obs, reward, term, trunc, info = env.step(None)
        done = term or trunc

        if demo_log and env.time_step == 12:
            print(f"\n[STEP {env.time_step}]")
            print(f"Mode: {env.mode}")
            print(f"Blackouts: {info.get('blackouts', 0)}")
            print(f"Misreporting: {info.get('honesty_violations', 0)}")
            c_state = "active" if info.get('coalition_rate', 0.0) > 0 else "inactive"
            print(f"Coalition: {c_state}\n")

        # Back-fill v2 metrics (env already appends standard ones in step())
        env.history["misreporting"].append(info.get("misreporting_rate",    0.0))
        env.history["coalition"].append(   info.get("coalition_rate",       0.0))
        env.history["delayed_failures"].append(
            info.get("delayed_failures_triggered", 0)
        )

    history = env.get_history()
    summary = env.summarize_episode()
    # Augment summary with v2 averages
    summary["avg_misreporting"]    = float(np.mean(history["misreporting"]))
    summary["avg_coalition_rate"]  = float(np.mean(history["coalition"]))
    summary["avg_delayed_failures"]= float(np.sum(history["delayed_failures"]))
    return history, summary


# ----------------------------------------------------------------------
# Multi-seed evaluation
# ----------------------------------------------------------------------

def run_all(seeds=SEEDS):
    histories = {name: [] for name, *_ in CONFIGS}
    summaries = {name: [] for name, *_ in CONFIGS}

    for seed in seeds:
        for name, mode, reward_mode in CONFIGS:
            env = GridOpsEnv(num_zones=3, max_steps=50, seed=seed, mode=mode)
            env.set_reward_mode(reward_mode)

            demo = (seed == 0 and name == "coordinated")
            history, summary = run_episode(env, demo_log=demo)
            histories[name].append(history)
            summaries[name].append(summary)

    return histories, summaries


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

METRIC_KEYS = [
    ("avg_reward",          "avg_reward"),
    ("avg_blackouts",       "avg_blackouts"),
    ("avg_imbalance",       "avg_imbalance"),
    ("avg_stability",       "avg_stability"),
    ("avg_misreporting",    "misreport"),
    ("avg_coalition_rate",  "coalition"),
    ("avg_delayed_failures","delayed_fail"),
]


def aggregate(summaries):
    agg = {}
    for name, runs in summaries.items():
        agg[name] = {key: float(np.mean([r[key] for r in runs]))
                     for key, _ in METRIC_KEYS}
    return agg


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Running multi-seed evaluation ...\n")
    histories, summaries = run_all()

    agg = aggregate(summaries)

    # Print table
    col    = 13
    names  = [n for n, *_ in CONFIGS]
    header = f"{'mode':<{col}}" + "".join(f" {label:>{col}}" for _, label in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for name in names:
        row = f"{name:<{col}}"
        for key, _ in METRIC_KEYS:
            row += f" {agg[name][key]:>{col}.3f}"
        print(row)

    # Save JSON
    results_json = {name: summaries[name] for name, *_ in CONFIGS}
    with open("results.json", "w") as f:
        json.dump(results_json, f, indent=4)
    print("\nSaved results to results.json")

    # Generate plots (existing)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "plots", os.path.join(os.path.dirname(__file__), "plots.py")
    )
    _plots = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_plots)
    _plots.generate_all_plots(histories, save_dir="plots")

    # ------------------------------------------------------------------
    # Research analysis
    # ------------------------------------------------------------------
    _aspec = importlib.util.spec_from_file_location(
        "analyze", os.path.join(os.path.dirname(__file__), "analyze.py")
    )
    _analyze = importlib.util.module_from_spec(_aspec)
    _aspec.loader.exec_module(_analyze)

    # Ablation study
    print("Running ablation study ...")
    abl_histories, abl_summary = _analyze.run_ablation()
    _analyze.plot_ablation(abl_summary, save_dir="plots")
    _analyze.plot_emergence(histories, save_dir="plots")
    _analyze.plot_delay_effects(histories, save_dir="plots")
    _analyze.plot_tradeoff_curve(histories, save_dir="plots")
    _analyze.plot_cascade_delay(histories, save_dir="plots")

    # Policy comparison table
    _analyze.print_policy_table(summaries)

    # Auto insights
    insights = _analyze.generate_insights(summaries, abl_summary)
    print("Key Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")

    # Export outputs/
    _analyze.export_outputs(summaries, abl_summary, out_dir="outputs")

    adv_reward = agg["advanced"]["avg_reward"]
    sel_reward = agg["selfish"]["avg_reward"]
    reward_imp = (adv_reward - sel_reward) / sel_reward * 100

    adv_mis = agg["advanced"]["avg_misreporting"] * 100
    sel_mis = agg["selfish"]["avg_misreporting"] * 100

    adv_stab = agg["advanced"]["avg_stability"]
    sel_stab = agg["selfish"]["avg_stability"]
    stab_gain = (adv_stab - sel_stab) / sel_stab * 100

    print("\n----------------------------------------")
    print("GRIDOPS++ RESULTS")
    print("----------------------------------------")
    print("Best Mode: Advanced")
    print(f"Reward Improvement: +{reward_imp:.1f}%")
    print(f"Misreport Reduction: {sel_mis:.0f}% -> {adv_mis:.0f}%")
    print(f"Stability Gain: +{stab_gain:.1f}%")
    print("----------------------------------------\n")

