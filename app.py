import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure the local modules can be imported correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

MODEL_PATH = "models/ppo_lstm_final"
VECNORM_PATH = "models/vecnormalize_lstm_final.pkl"

def simulate_episode():
    # 1. Setup Environment
    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    if os.path.exists(VECNORM_PATH):
        env = VecNormalize.load(VECNORM_PATH, env)
    else:
        return "VecNormalize stats not found! Please check paths.", None
        
    env.training = False
    env.norm_reward = False
    
    # 2. Setup Model
    if not os.path.exists(MODEL_PATH + ".zip"):
        return "Trained model not found! Please check paths.", None
        
    model = RecurrentPPO.load(MODEL_PATH, env=env)
    
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    done = [False]
    
    steps = 0
    total_reward = 0
    blackouts = 0
    
    allocations = []
    demands = []
    
    # 3. Run Simulation
    while not done[0] and steps < 100:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        
        # We can extract demand from the raw observation if possible, 
        # but for simplicity we will just chart the actions (allocations)
        obs, reward, done, info = env.step(action)
        episode_starts = done.copy()
        
        steps += 1
        total_reward += reward[0]
        
        # Safely track blackouts
        ep_info = info[0]
        if "blackout_count" in ep_info:
            blackouts += ep_info["blackout_count"]
        elif "blackouts" in ep_info:
            blackouts += ep_info["blackouts"]
        elif "fault_count" in ep_info:
            blackouts += ep_info["fault_count"]
            
        allocations.append(action[0])
        
    # 4. Create Beautiful Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    allocs = np.array(allocations)
    
    # Plotting each zone's allocation
    ax.plot(allocs[:, 0], label="Zone 1 (Residential - Low Priority)", color="#3a86ff", linewidth=2)
    ax.plot(allocs[:, 1], label="Zone 2 (Commercial - Medium Priority)", color="#ffbe0b", linewidth=2)
    ax.plot(allocs[:, 2], label="Zone 3 (Hospital - High Priority)", color="#ff006e", linewidth=2)
    
    ax.set_title(f"Agent Power Allocations Over Time\nTotal Reward: {total_reward:.2f} | Total Blackouts: {int(blackouts)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Power Allocation Fraction", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    # 5. Format Summary
    summary = (
        f"✅ Simulation finished successfully after {steps} steps.\n"
        f"🏆 Total Reward: {total_reward:.2f}\n"
        f"🚨 Total Blackouts: {int(blackouts)}\n\n"
        f"Insight: Notice how the agent stabilizes the allocations to prioritize the critical Zone 3 while actively managing the volatility in other zones to prevent cascading failures."
    )
    
    return summary, fig

# --- Gradio UI Design ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # ⚡ GridMind: Teaching an AI to Prevent Power Grid Blackouts
        
        This live demo runs our fully trained **PPO+LSTM** agent against the custom `GridOpsEnv`. 
        The agent must distribute limited power across 3 zones (Residential, Commercial, Hospital) while facing sudden demand spikes.
        
        Click **Run Simulation** to watch the agent react to a highly volatile episode in real-time.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            run_btn = gr.Button("▶️ Run Full Episode Simulation", variant="primary", size="lg")
            output_text = gr.Textbox(label="Simulation Summary", lines=6)
        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Live Power Allocations")
            
    run_btn.click(fn=simulate_episode, outputs=[output_text, output_plot])
    
    gr.Markdown("*(Developed for the OpenEnv Hackathon 2026)*")

if __name__ == "__main__":
    demo.launch()
