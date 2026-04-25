"""
Final Policy Comparison — GridOps RL Project
All metrics from completed evaluation runs (no re-evaluation needed).
"""

# ─────────────────────────────────────────────────────────────
# Recorded results from all evaluation runs
# ─────────────────────────────────────────────────────────────

RESULTS = [
    # (agent_name,          reward_per_step, blackouts, stability)
    ("PPO (MLP)",           0.246,           50.800,    0.622),
    ("PPO (LSTM)",          0.305,           30.450,    0.728),
    ("Advanced Heuristic",  5.255,            0.000,    0.944),
]

MLP_RPS = 0.246
MLP_BLK = 50.800
LSTM_RPS = 0.305
LSTM_BLK = 30.450


def print_comparison():
    # ── Table ────────────────────────────────────────────────
    W = 50
    print()
    print("=" * W)
    print("FINAL POLICY COMPARISON")
    print("=" * W)
    print(f"{'Agent':<22} {'Reward/Step':>11} {'Blackouts':>10} {'Stability':>10}")
    print("-" * W)
    for name, rps, blk, stab in RESULTS:
        print(f"{name:<22} {rps:>11.3f} {blk:>10.3f} {stab:>10.3f}")
    print("=" * W)

    # ── LSTM vs MLP improvement metrics ─────────────────────
    blk_reduction  = (MLP_BLK  - LSTM_BLK)  / MLP_BLK  * 100
    rps_gain       = (LSTM_RPS - MLP_RPS)    / abs(MLP_RPS) * 100

    print()
    print("-" * W)
    print("PPO (LSTM) vs PPO (MLP)")
    print("-" * W)
    print(f"  Blackout reduction : {blk_reduction:+.1f}%"
          f"  ({MLP_BLK:.1f} -> {LSTM_BLK:.1f})")
    print(f"  Reward improvement : {rps_gain:+.1f}%"
          f"  ({MLP_RPS:.3f} -> {LSTM_RPS:.3f})")

    # ── Advanced vs LSTM ─────────────────────────────────────
    adv_rps, adv_blk = 5.255, 0.000
    adv_blk_vs_lstm  = (LSTM_BLK - adv_blk) / LSTM_BLK * 100
    adv_rps_vs_lstm  = (adv_rps  - LSTM_RPS) / abs(LSTM_RPS) * 100

    print()
    print("-" * W)
    print("Advanced Heuristic vs PPO (LSTM)")
    print("-" * W)
    print(f"  Blackout reduction : {adv_blk_vs_lstm:+.1f}%"
          f"  ({LSTM_BLK:.1f} -> {adv_blk:.1f})")
    print(f"  Reward improvement : {adv_rps_vs_lstm:+.1f}%"
          f"  ({LSTM_RPS:.3f} -> {adv_rps:.3f})")

    # ── Improvement hierarchy ────────────────────────────────
    print()
    print("=" * W)
    print("IMPROVEMENT HIERARCHY  (fewest blackouts = rank 1)")
    print("=" * W)
    ranked = sorted(RESULTS, key=lambda x: x[2])
    for rank, (name, rps, blk, stab) in enumerate(ranked, 1):
        print(f"  {rank}. {name:<22}  Blackouts = {blk:>6.3f}")
    print("=" * W)


if __name__ == "__main__":
    print_comparison()
