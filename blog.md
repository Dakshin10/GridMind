# GridMind: Teaching AI to Prevent Power Grid Cascades

*How we used Reinforcement Learning to teach an AI "Defensive Curtailment" and save the power grid.*

---

## ⚡ The Stakes
Modern power grids are incredibly fragile. A spike in demand can overload a line, which trips a safeguard. When that line goes down, its load shifts to neighboring lines, overloading them. Within seconds, a minor localized fault becomes a city-wide blackout.

We wanted to see if an AI could learn to prevent this. 

## 🧠 Our Approach
We built a custom OpenEnv-compliant environment called `GridOpsEnv`. It simulates a 3-zone grid (Residential, Commercial, and Hospital). 

The goal was simple: Allocate power efficiently, but **never let the hospital lose power** and **never trigger a blackout**.

We trained a Proximal Policy Optimization (PPO) agent with an LSTM memory network. The memory was crucial because overloads are delayed—a bad decision on minute 1 might not cause a blackout until minute 5. 

## 📈 What Emerged
The results were stunning. After 170,000 timesteps, the AI learned a strategy that human grid operators use called **defensive curtailment**.

When demand spiked dangerously high, the AI realized that trying to serve 100% of the power would crash the grid. Instead, it *intentionally under-served* the Residential and Commercial zones, creating a safety buffer. It routed the maximum available power strictly to the Hospital. 

By sacrificing a small amount of comfort for the residential zones, it completely eliminated cascading blackouts. It learned to sacrifice the few to save the many!

## 🚀 Try It
You can try the trained agent on our Hugging Face Space, or check out our Colab training script to see the learning curves for yourself!
