import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from train.train import GridOpsEnvWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import RecurrentPPO
    DEPENDENCIES_LOADED = True
except ImportError:
    DEPENDENCIES_LOADED = False

MODEL_PATH = "models/ppo_lstm_final"
VECNORM_PATH = "models/vecnormalize_lstm_final.pkl"

class GridSimulator:
    def __init__(self):
        self.ready = False
        if not DEPENDENCIES_LOADED:
            return
            
        self.env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
        if os.path.exists(VECNORM_PATH):
            self.env = VecNormalize.load(VECNORM_PATH, self.env)
        self.env.training = False
        self.env.norm_reward = False
        
        if os.path.exists(MODEL_PATH + ".zip"):
            self.model = RecurrentPPO.load(MODEL_PATH, env=self.env)
            self.ready = True
            self.reset()
            
    def reset(self):
        self.obs = self.env.reset()
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.done = False
        self.steps = 0
        self.total_reward = 0.0
        self.blackouts = 0
        self.stability_history = [1.0]
        
        # Raw observations fallback
        self.current_demand = [0.33, 0.33, 0.33]
        self.current_supply = [0.33, 0.33, 0.33]
        self.stability = 1.0
        self.action_taken = [0.33, 0.33, 0.33]
        self.fault_status = [False, False, False]
        self.explanation = "System reset. Ready for AI allocation."
        
    def step(self, action=None, manual=False):
        if self.done:
            return self.get_ui_state()
            
        if not manual:
            action, self.lstm_states = self.model.predict(
                self.obs,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True
            )
        else:
            # Ensure action sums to 1.0
            total = sum(action)
            action = [a/total if total > 0 else 0.33 for a in action]
            action = np.array([action])
            
        self.obs, reward, done, info = self.env.step(action)
        self.episode_starts = done.copy()
        
        ep_info = info[0]
        
        self.steps += 1
        self.total_reward += reward[0]
        self.done = done[0]
        self.action_taken = action[0]
        
        # Extract metrics safely
        if "blackout_count" in ep_info:
            self.blackouts += ep_info["blackout_count"]
        elif "blackouts" in ep_info:
            self.blackouts += ep_info["blackouts"]
        elif "fault_count" in ep_info:
            self.blackouts += ep_info["fault_count"]
            
        self.stability = ep_info.get("stability_score", ep_info.get("stability", 1.0))
        self.stability_history.append(self.stability)
        
        # Extract demand/supply for visualization if available in info, otherwise estimate
        # Assuming observation space has demand in first 3 dims and supply in next 3
        # We use the raw action as supply for visual clarity
        self.current_supply = self.action_taken
        raw_obs = self.env.get_original_obs() if hasattr(self.env, 'get_original_obs') else self.obs
        self.current_demand = raw_obs[0][:3] if len(raw_obs[0]) >= 3 else [0.33, 0.33, 0.33]
        
        self.generate_explanation()
        return self.get_ui_state()
        
    def generate_explanation(self):
        z1, z2, z3 = self.action_taken
        if z3 > 0.45:
            self.explanation = "AI heavily prioritized power to the Critical Zone (Hospital) to prevent a catastrophic failure during high load."
        elif z1 > z2 and z1 > z3:
            self.explanation = "AI shifted surplus power to the Residential Zone to balance baseline load and prevent localized faults."
        elif z2 > z1 and z2 > z3:
            self.explanation = "AI allocated maximum capacity to the Commercial Zone to stabilize immediate demand spikes."
        else:
            self.explanation = "AI distributed power evenly across all zones to maintain global grid stability."
            
        if self.blackouts > 0:
            self.explanation = "⚠️ A blackout occurred! The AI is now aggressively rerouting power to prevent cascading failures."
            
        if self.done:
            self.explanation += " Episode complete."
            
    def get_ui_state(self):
        # Format KPIs
        stab_str = f"{self.stability:.2f} " + ("🟢" if self.stability > 0.8 else "🟡" if self.stability > 0.5 else "🔴")
        blk_str = f"{int(self.blackouts)} " + ("🟢" if self.blackouts == 0 else "🔴")
        rew_str = f"{self.total_reward:.2f}"
        step_str = str(self.steps)
        
        # Generate Trend Plot
        fig_trend, ax_trend = plt.subplots(figsize=(6, 3))
        ax_trend.plot(self.stability_history, color="#10b981", linewidth=2)
        ax_trend.set_title("Grid Stability Over Time", fontsize=10, pad=10)
        ax_trend.set_ylim(0, 1.1)
        ax_trend.grid(True, linestyle="--", alpha=0.3)
        ax_trend.spines['top'].set_visible(False)
        ax_trend.spines['right'].set_visible(False)
        plt.tight_layout()
        
        # Generate Bar Plot
        fig_grid, ax_grid = plt.subplots(figsize=(6, 3))
        zones = ["Residential", "Commercial", "Critical"]
        x = np.arange(len(zones))
        width = 0.35
        
        ax_grid.bar(x - width/2, self.current_demand, width, label='Demand', color="#ef4444")
        ax_grid.bar(x + width/2, self.current_supply, width, label='Supply (AI)', color="#3b82f6")
        
        ax_grid.set_title("Demand vs. Supply Allocation", fontsize=10, pad=10)
        ax_grid.set_xticks(x)
        ax_grid.set_xticklabels(zones)
        ax_grid.legend(loc="upper right", fontsize=8)
        ax_grid.grid(axis="y", linestyle="--", alpha=0.3)
        ax_grid.spines['top'].set_visible(False)
        ax_grid.spines['right'].set_visible(False)
        plt.tight_layout()
        
        return stab_str, blk_str, rew_str, step_str, self.explanation, fig_trend, fig_grid

# Global Simulator Instance
sim = GridSimulator()

def ui_reset():
    if not sim.ready: return ("Error", "Error", "Error", "Error", "Model not loaded.", None, None)
    sim.reset()
    return sim.get_ui_state()

def ui_ai_step():
    if not sim.ready: return ("Error", "Error", "Error", "Error", "Model not loaded.", None, None)
    return sim.step()

def ui_auto_run():
    if not sim.ready: 
        yield ("Error", "Error", "Error", "Error", "Model not loaded.", None, None)
        return
        
    while not sim.done and sim.steps < 100:
        yield sim.step()
        time.sleep(0.1)  # Smooth animation

def ui_manual_step(z1, z2, z3):
    if not sim.ready: return ("Error", "Error", "Error", "Error", "Model not loaded.", None, None)
    return sim.step(action=[z1, z2, z3], manual=True)

# --- UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Base()) as demo:
    
    # 1. HEADER
    with gr.Row():
        gr.Markdown(
            """
            # ⚡ GridMind: AI for Power Grid Stability
            ### AI dynamically allocates power to prevent cascading blackouts in real-time
            **Goal:** Stability ↑ | Blackouts ↓
            """
        )
        
    # 2. PRIMARY CONTROLS
    with gr.Row():
        btn_ai_step = gr.Button("🤖 AI Step", variant="primary", size="lg")
        btn_auto = gr.Button("▶️ Auto Run", variant="primary", size="lg")
        btn_reset = gr.Button("🔄 Reset", variant="secondary", size="lg")
        
    # 3. KPI DASHBOARD
    with gr.Row():
        kpi_stability = gr.Textbox(label="⚡ Grid Stability", value="1.00 🟢", interactive=False)
        kpi_blackouts = gr.Textbox(label="🚨 Blackouts", value="0 🟢", interactive=False)
        kpi_reward = gr.Textbox(label="🏆 Total Reward", value="0.00", interactive=False)
        kpi_steps = gr.Textbox(label="⏱️ Steps", value="0", interactive=False)
        
    # 6. AI DECISION EXPLANATION
    with gr.Row():
        explanation = gr.Textbox(
            label="🤖 AI Decision Explanation", 
            value="System ready. Click 'AI Step' or 'Auto Run' to begin.",
            interactive=False,
            lines=2
        )
        
    # 4 & 5. VISUALIZATIONS
    with gr.Row():
        plot_trend = gr.Plot(label="Live Trend Graph")
        plot_grid = gr.Plot(label="Grid Visualization")
        
    # 8. MANUAL CONTROL
    with gr.Accordion("🎮 Manual Control (Compare against AI)", open=False):
        gr.Markdown("Take control of the grid. Can you prevent blackouts better than the AI?")
        with gr.Row():
            slider_z1 = gr.Slider(0, 1, value=0.33, label="Zone 1 (Residential)")
            slider_z2 = gr.Slider(0, 1, value=0.33, label="Zone 2 (Commercial)")
            slider_z3 = gr.Slider(0, 1, value=0.33, label="Zone 3 (Critical)")
        btn_manual_step = gr.Button("🎮 Manual Step", variant="secondary")
        
    # Wiring Events
    outputs = [kpi_stability, kpi_blackouts, kpi_reward, kpi_steps, explanation, plot_trend, plot_grid]
    
    btn_reset.click(fn=ui_reset, outputs=outputs)
    btn_ai_step.click(fn=ui_ai_step, outputs=outputs)
    btn_auto.click(fn=ui_auto_run, outputs=outputs)
    btn_manual_step.click(fn=ui_manual_step, inputs=[slider_z1, slider_z2, slider_z3], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
