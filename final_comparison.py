"""
Final Enhanced Policy Comparison
All metrics from completed evaluation runs — no re-evaluation needed.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# All recorded results
# ─────────────────────────────────────────────────────────────

agents = [
    # (name,              rps,    blk,     stab)
    ("Random",           -0.200,  60.000,  0.450),
    ("PPO (MLP)",         0.246,  50.800,  0.622),
    ("PPO (LSTM)",        0.305,  30.450,  0.728),
    ("Hybrid (Best)",     0.132,  40.900,  0.659),  # Risk-Aware: best balance of all hybrids
    ("Advanced",          5.255,   0.000,  0.944),
]

# All hybrid variants (for selection rationale)
hybrid_variants = [
    ("Hard Override (cumulative)",  0.122, 40.50, 0.615),
    ("Hard Override (per-step)",    0.124, 49.80, 0.615),
    ("Hard Override (overload)",   -0.297, 66.95, 0.478),
    ("Soft Blend (dynamic alpha)",  0.134, 43.20, 0.638),
    ("Risk-Aware Controller",       0.132, 40.90, 0.659),  # <- best overall balance
]

LSTM_RPS = 0.305
LSTM_BLK = 30.450
MLP_RPS  = 0.246
MLP_BLK  = 50.800


def print_table(rows, title="FINAL ENHANCED COMPARISON"):
    print("\n" + "=" * 52)
    print(title)
    print("=" * 52)
    print(f"{'Agent':<17} {'Reward/Step':<13} {'Blackouts':<11} {'Stability'}")
    print("-" * 52)
    for name, rps, blk, stab in rows:
        print(f"{name:<17} {rps:<13.3f} {blk:<11.3f} {stab:.3f}")
    print("=" * 52)


def improvement_metrics(name, rps, blk, stab):
    blk_vs_lstm  = (LSTM_BLK - blk) / LSTM_BLK  * 100
    rps_vs_lstm  = (rps - LSTM_RPS)  / abs(LSTM_RPS) * 100
    blk_vs_mlp   = (MLP_BLK - blk)  / MLP_BLK   * 100
    rps_vs_mlp   = (rps - MLP_RPS)  / abs(MLP_RPS)  * 100
    return blk_vs_lstm, rps_vs_lstm, blk_vs_mlp, rps_vs_mlp


def main():
    # ── Main comparison table ─────────────────────────────────
    # Show key agents (exclude Random from main table per request)
    main_rows = [a for a in agents if a[0] != "Random"]
    print_table(main_rows)

    # ── Improvement metrics ───────────────────────────────────
    hybrid_rps  = 0.132
    hybrid_blk  = 40.900
    hybrid_stab = 0.659

    print("\n" + "-" * 52)
    print("IMPROVEMENT METRICS")
    print("-" * 52)

    # LSTM vs MLP
    lstm_blk_vs_mlp = (MLP_BLK - LSTM_BLK) / MLP_BLK * 100
    lstm_rps_vs_mlp = (LSTM_RPS - MLP_RPS)  / abs(MLP_RPS) * 100
    print(f"\n  PPO (LSTM) vs PPO (MLP):")
    print(f"    Blackout reduction : {lstm_blk_vs_mlp:+.1f}%")
    print(f"    Reward change      : {lstm_rps_vs_mlp:+.1f}%")

    # Hybrid vs LSTM
    h_blk_vs_lstm = (LSTM_BLK - hybrid_blk) / LSTM_BLK * 100
    h_rps_vs_lstm = (hybrid_rps - LSTM_RPS)  / abs(LSTM_RPS) * 100
    print(f"\n  Hybrid (Best) vs PPO (LSTM):")
    print(f"    Blackout reduction : {h_blk_vs_lstm:+.1f}%  {'(increase)' if h_blk_vs_lstm < 0 else ''}")
    print(f"    Reward change      : {h_rps_vs_lstm:+.1f}%")

    # Hybrid vs MLP
    h_blk_vs_mlp = (MLP_BLK - hybrid_blk) / MLP_BLK * 100
    h_rps_vs_mlp = (hybrid_rps - MLP_RPS)  / abs(MLP_RPS)  * 100
    print(f"\n  Hybrid (Best) vs PPO (MLP):")
    print(f"    Blackout reduction : {h_blk_vs_mlp:+.1f}%")
    print(f"    Reward change      : {h_rps_vs_mlp:+.1f}%")

    # Advanced vs LSTM
    adv_blk_vs_lstm = (LSTM_BLK - 0.0)  / LSTM_BLK  * 100
    adv_rps_vs_lstm = (5.255 - LSTM_RPS) / abs(LSTM_RPS) * 100
    print(f"\n  Advanced vs PPO (LSTM):")
    print(f"    Blackout reduction : {adv_blk_vs_lstm:+.1f}%")
    print(f"    Reward change      : {adv_rps_vs_lstm:+.1f}%")

    # ── All hybrid variants ───────────────────────────────────
    print("\n" + "-" * 52)
    print("ALL HYBRID VARIANTS (vs pure LSTM PPO)")
    print("-" * 52)
    print(f"{'Variant':<33} {'Reward':>8} {'Blackouts':>10} {'Blk Delta':>10}")
    print("-" * 62)
    for name, rps, blk, stab in hybrid_variants:
        blk_delta = blk - LSTM_BLK
        marker = " <-- best" if name == "Risk-Aware Controller" else ""
        print(f"{name:<33} {rps:>8.3f} {blk:>10.3f} {blk_delta:>+10.3f}{marker}")

    # ── Improvement hierarchy ─────────────────────────────────
    print("\n" + "=" * 52)
    print("IMPROVEMENT HIERARCHY (Blackout reduction vs MLP)")
    print("=" * 52)

    ranked = [
        ("Advanced",      0.000,      "100.0% blackout reduction"),
        ("PPO (LSTM)",    LSTM_BLK,   f"{lstm_blk_vs_mlp:.1f}% blackout reduction"),
        ("Hybrid (Best)", hybrid_blk, f"{h_blk_vs_mlp:.1f}% blackout reduction"),
        ("PPO (MLP)",     MLP_BLK,    "Baseline"),
    ]
    ranked_sorted = sorted(ranked, key=lambda x: x[1])
    for rank, (name, blk, note) in enumerate(ranked_sorted, 1):
        print(f"  {rank}. {name:<17} Blackouts={blk:>6.3f}  ({note})")

    # ── Final verdict ─────────────────────────────────────────
    print("\n" + "=" * 52)
    print("VERDICT")
    print("=" * 52)
    print("""
  1. PPO (LSTM) is the best trainable RL policy:
       40.1% blackout reduction vs MLP, +24.0% reward gain.

  2. All hybrid overrides DEGRADE vs pure LSTM:
       The LSTM's temporal memory (EV=0.86) already encodes
       optimal risk-awareness across time steps.

  3. Risk-Aware is the best hybrid variant:
       40.9 blackouts, 0.132 reward — best balance of
       blackout control and reward among all overrides.

  4. Advanced heuristic is the safety ceiling:
       0 blackouts, reward=5.255 — unreachable by RL alone
       without domain-specific priority/reputation weighting.

  5. Key insight:
       For temporal environments, train longer with LSTM
       rather than layering post-hoc heuristics.
    """)


if __name__ == "__main__":
    main()
