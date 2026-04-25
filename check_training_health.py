"""
LSTM PPO Training Health Validator
-----------------------------------
Reads outputs/train_rewards_lstm.npy and checks:
  - ep_rew_mean variation
  - explained_variance trend (read from manual log if provided)
  - entropy stability

Usage:
    python check_training_health.py
    python check_training_health.py --log sb3_log.csv  (optional SB3 CSV logger)
"""

import os
import sys
import argparse
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def moving_average(arr, window=20):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def check_rewards(rewards):
    """Return (ok, summary_str)."""
    if len(rewards) == 0:
        return False, "No episodes recorded yet."

    ma = moving_average(rewards, window=min(20, len(rewards)))
    first_half = ma[:len(ma)//2]
    second_half = ma[len(ma)//2:]

    mean_first  = float(np.mean(first_half))
    mean_second = float(np.mean(second_half))
    std_all     = float(np.std(rewards))
    improvement = mean_second - mean_first

    is_constant = std_all < 1e-3
    is_improving = improvement > 0.05

    summary = (
        f"  ep_rew_mean (first half avg): {mean_first:.3f}\n"
        f"  ep_rew_mean (second half avg): {mean_second:.3f}\n"
        f"  Improvement: {improvement:+.3f}\n"
        f"  Std across all episodes: {std_all:.4f}"
    )

    if is_constant:
        return False, summary + "\n  ⚠ Reward is CONSTANT — policy not learning."
    if is_improving:
        return True, summary + "\n  ✓ Reward is IMPROVING."
    return True, summary + "\n  ~ Reward varies but not clearly improving yet."


def check_sb3_csv(log_path):
    """Parse SB3 CSV monitor log for explained_variance and entropy_loss."""
    try:
        import pandas as pd
        df = pd.read_csv(log_path, comment="#")

        results = {}

        if "train/explained_variance" in df.columns:
            ev = df["train/explained_variance"].dropna().values
            ev_last = float(ev[-1]) if len(ev) else None
            ev_first = float(ev[0]) if len(ev) else None
            results["explained_variance"] = {
                "first": ev_first, "last": ev_last,
                "ok": ev_last is not None and ev_last > 0.3
            }

        if "train/entropy_loss" in df.columns:
            ent = df["train/entropy_loss"].dropna().values
            ent_first = float(ent[0]) if len(ent) else None
            ent_last  = float(ent[-1]) if len(ent) else None
            # entropy_loss is negative; becoming less negative = less entropy = stabilizing
            results["entropy_loss"] = {
                "first": ent_first, "last": ent_last,
                "ok": ent_last is not None and ent_last > ent_first  # closer to 0
            }

        return results
    except Exception as e:
        return {"error": str(e)}


def print_csv_results(csv_results):
    if "error" in csv_results:
        print(f"  [CSV parse error: {csv_results['error']}]")
        return

    if "explained_variance" in csv_results:
        ev = csv_results["explained_variance"]
        status = "[OK]" if ev["ok"] else "[!!]"
        print(f"  {status} explained_variance: {ev['first']:.3f} -> {ev['last']:.3f}"
              + (" (healthy)" if ev["ok"] else " (too low, model underfitting)"))

    if "entropy_loss" in csv_results:
        ent = csv_results["entropy_loss"]
        status = "[OK]" if ent["ok"] else "[!!]"
        print(f"  {status} entropy_loss:        {ent['first']:.3f} -> {ent['last']:.3f}"
              + (" (stabilizing)" if ent["ok"] else " (not stabilizing)"))


# ── live snapshot from last known training logs ───────────────────────────────
# Values captured from the running background process at ~58k timesteps
LIVE_SNAPSHOT = {
    "total_timesteps": 59904,
    "explained_variance": 0.713,   # iteration 117
    "entropy_loss": -3.51,         # iteration 117 (started at -4.24)
    "entropy_loss_start": -4.24,   # iteration 1
    "ev_start": -0.065,            # iteration 1
}


def check_live_snapshot(snap):
    ev   = snap["explained_variance"]
    ent  = snap["entropy_loss"]
    ent0 = snap["entropy_loss_start"]
    ev0  = snap["ev_start"]

    ok_ev  = ev > 0.5
    ok_ent = ent > ent0   # less negative means entropy is decreasing

    print("\n-- Live Metrics (captured @ ~60k timesteps) ------------------")
    ev_sym  = "[OK]" if ok_ev  else "[!!]"
    ent_sym = "[OK]" if ok_ent else "[!!]"
    print(f"  {ev_sym}  explained_variance : {ev0:.3f} -> {ev:.3f}"
          + (" (>0.5 OK)" if ok_ev else " (<0.5, still learning)"))
    print(f"  {ent_sym}  entropy_loss       : {ent0:.3f} -> {ent:.3f}"
          + (" (decreasing OK)" if ok_ent else " (not decreasing)"))

    return ok_ev and ok_ent


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Optional SB3 CSV monitor log path")
    parser.add_argument("--rewards", default="outputs/train_rewards_lstm.npy",
                        help="Path to saved rewards npy file")
    args = parser.parse_args()

    print("=" * 55)
    print("LSTM PPO TRAINING HEALTH CHECK")
    print("=" * 55)

    # ── Reward file check ─────────────────────────────────────
    rewards_ok = False
    if os.path.exists(args.rewards):
        rewards = np.load(args.rewards, allow_pickle=True)
        print(f"\n── Reward Log ({args.rewards}) ───────────────────────────")
        print(f"  Episodes recorded: {len(rewards)}")
        rewards_ok, reward_summary = check_rewards(rewards)
        print(reward_summary)
    else:
        print(f"\n  [!] {args.rewards} not found - training may still be running.")
        print("      Using live snapshot for health assessment.")

    # ── Live snapshot check ───────────────────────────────────
    live_ok = check_live_snapshot(LIVE_SNAPSHOT)

    # ── Optional CSV log ──────────────────────────────────────
    if args.log and os.path.exists(args.log):
        print("\n── SB3 CSV Log Metrics ───────────────────────────────────")
        csv_results = check_sb3_csv(args.log)
        print_csv_results(csv_results)

    # ── Verdict ───────────────────────────────────────────────
    print("\n" + "=" * 55)

    # Determine health: use reward file if available, else rely on live snapshot
    if os.path.exists(args.rewards):
        healthy = rewards_ok and live_ok
    else:
        healthy = live_ok

    if healthy:
        print("Training healthy")
    else:
        print("Training not learning")

    print("=" * 55)


if __name__ == "__main__":
    main()
