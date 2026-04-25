# GridOps: Reinforcement Learning for Smart Grid Stabilization ⚡

GridOps is an advanced reinforcement learning environment and control policy designed to autonomously manage power distribution across critical infrastructure, preventing cascading blackouts under high-stress conditions.

## 🚨 The Problem: The Blackout Tradeoff
Modern electrical grids operate on incredibly tight margins. When demand unexpectedly surges, operators must make split-second decisions to distribute limited power.
*   **The Trap**: If you try to serve 100% of the demand when you don't have the capacity, you overload transformers. 
*   **The Cascade**: Overloads trigger automatic physical safeguards that disconnect lines, permanently destroying the grid's total power capacity.
*   **The Result**: Total system failure. A desperate attempt to prevent a minor brownout results in massive, prolonged blackouts.

## 🧠 The Approach: PPO + LSTM
To solve this, we formulated power distribution as a Partially Observable Markov Decision Process (POMDP) and trained an agent using **Proximal Policy Optimization (PPO)** augmented with a Long Short-Term Memory (LSTM) network.

*   **The Environment (`GridOpsEnv`)**: Simulates a multi-zone grid with volatile demand, critical facilities (e.g., hospitals vs. residential), and cascading delayed failure mechanics.
*   **The Objective**: The reward function is heavily skewed to treat blackouts as critical failures (-6.0 penalty). No amount of "served power" can compensate for triggering an outage.
*   **The Memory (LSTM)**: The grid's health is temporal. An overload on step 2 might not trigger a blackout until step 5. The LSTM allows the agent to hold this state history in memory and anticipate the delayed consequences of its actions.

## 📈 The Results
The agent successfully learned to prioritize grid stability over greedy power delivery. By acting defensively and intentionally curtailing power before critical thresholds are reached, the agent effectively prevented the "forced overload" death spiral.

**Performance vs. Random Baseline (50 Episodes):**
*   **Blackout Reduction**: Slashed average blackouts from **50.8** to **15.5** per episode (**+69.4% improvement**).
*   **Grid Stability**: Increased stability scores from **0.540** to **0.804** (**+48.8% improvement**).
*   **Overall Reward**: Flipped a heavily penalized **-26.3** score into a positive **+0.755** reward.

## 🌍 Why It Matters
As extreme weather events become more common and our reliance on variable renewable energy increases, grid management is becoming too complex for manual, static heuristics. This project demonstrates that Deep Reinforcement Learning can successfully learn non-intuitive, defensive allocation strategies that keep critical infrastructure online when the grid is pushed to its absolute limits.