import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from train.train import GridOpsEnvWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import RecurrentPPO
    DEPENDENCIES_LOADED = True
except ImportError:
    DEPENDENCIES_LOADED = False

MODEL_PATH  = "models/ppo_lstm_final"
VECNORM_PATH = "models/vecnormalize_lstm_final.pkl"
MAX_STEPS   = 50

ZONE_LABELS = ["Zone 1 (Residential)", "Zone 2 (Commercial)", "Zone 3 (Hospital)"]
ZONE_TYPES  = ["Commercial (medium)", "Residential (low)", "Commercial (medium)"]

_DARK_BG  = "#1e1e1e"
_DARK_AX  = "#181818"
_GRID_CLR = "#2a2a2a"


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_array_obs(obs, length=3):
    try:
        if isinstance(obs, dict):
            for key in ("observation", "obs", "state"):
                if key in obs:
                    arr = np.asarray(obs[key]).flatten()
                    return arr[:length].tolist() if len(arr) >= length else [0.33] * length
            if obs:
                arr = np.asarray(list(obs.values())[0]).flatten()
                return arr[:length].tolist() if len(arr) >= length else [0.33] * length
        arr = np.asarray(obs).flatten()
        return arr[:length].tolist() if len(arr) >= length else [0.33] * length
    except Exception:
        return [0.33] * length


def _heuristic_action(demand):
    weights = [demand[0], demand[1], demand[2] * 1.2]
    total = sum(weights)
    if total <= 0:
        return np.array([[0.25, 0.35, 0.40]])
    return np.array([[w / total for w in weights]])


def _make_grid_chart(demand_vals, supply_vals, fault_status=None):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_DARK_BG)
    ax.set_facecolor(_DARK_AX)

    x = np.arange(3)
    width = 0.35

    demand_vals = [max(float(v), 0) for v in demand_vals]
    supply_vals = [max(float(v), 0) for v in supply_vals]

    bars_d = ax.bar(x - width / 2, demand_vals, width,
                    label="Demand", color="#ef4444", alpha=0.90, zorder=3)
    bars_s = ax.bar(x + width / 2, supply_vals, width,
                    label="Supply", color="#22c55e", alpha=0.90, zorder=3)

    # FAULT annotations
    if fault_status is None:
        fault_status = [
            demand_vals[i] > 0.01 and supply_vals[i] < demand_vals[i] * 0.6
            for i in range(3)
        ]
    for i, fault in enumerate(fault_status):
        if fault:
            top = max(demand_vals[i], supply_vals[i])
            ax.text(x[i], top + max(top * 0.06, 0.04),
                    "⚠ FAULT", ha="center", va="bottom",
                    color="#ef4444", fontsize=9, fontweight="bold", zorder=5)

    y_top = max(max(demand_vals), max(supply_vals), 0.5) * 1.4
    ax.set_ylim(0, y_top)
    ax.set_title("Grid Status per Zone", color="white", fontsize=12,
                 pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Zone 1", "Zone 2", "Zone 3"], color="#cccccc", fontsize=10)
    ax.set_ylabel("Power Units", color="#888888", fontsize=9)
    ax.tick_params(axis="y", colors="#888888", labelsize=8)
    ax.tick_params(axis="x", colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a2a", edgecolor="#444444",
              labelcolor="white", fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.12, color="white", zorder=0)
    plt.tight_layout(pad=1.5)
    return fig


# ── simulator ─────────────────────────────────────────────────────────────────

class GridSimulator:
    def __init__(self):
        self.ready = False
        self.obs   = None
        if not DEPENDENCIES_LOADED:
            return

        self.env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
        if os.path.exists(VECNORM_PATH):
            self.env = VecNormalize.load(VECNORM_PATH, self.env)
        self.env.training  = False
        self.env.norm_reward = False

        if os.path.exists(MODEL_PATH + ".zip"):
            self.model = RecurrentPPO.load(MODEL_PATH, env=self.env)
        else:
            self.model = None          # heuristic-only demo mode

        self.ready = True
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        if DEPENDENCIES_LOADED and hasattr(self, "env"):
            self.obs = self.env.reset()
        else:
            self.obs = np.array([[0.33, 0.33, 0.33, 0.33, 0.33, 0.33]])

        self.lstm_states    = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.done           = False
        self.steps          = 0
        self.total_reward   = 0.0
        self.last_reward    = 0.0
        self.blackouts      = 0
        self.unmet_demand   = 0.0
        self.stability      = 1.0
        self.stability_history = []
        self.current_demand = [0.80, 0.60, 0.40]
        self.current_supply = [0.33, 0.33, 0.33]
        self.action_taken   = [0.33, 0.33, 0.33]
        self.fault_status   = [False, False, False]

    # ------------------------------------------------------------------
    def step(self, action=None, manual=False):
        if self.obs is None:
            return self._safe_state("⚠️ Please click Reset Env before running")
        if self.done:
            return self.get_ui_state("Episode done — click Reset Env to restart.")

        if not manual:
            if self.model is not None:
                action, self.lstm_states = self.model.predict(
                    self.obs,
                    state=self.lstm_states,
                    episode_start=self.episode_starts,
                    deterministic=True,
                )
            else:
                raw = self.obs[0] if hasattr(self.obs, '__len__') else self.obs
                action = _heuristic_action(_extract_array_obs(raw))
        else:
            total = sum(action)
            action = np.array([[a / total if total > 0 else 0.33 for a in action]])

        # Gym / Gymnasium compatibility
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated | truncated
        else:
            obs, reward, done, info = result

        self.obs = obs
        self.episode_starts = done.copy()
        ep_info = info[0] if isinstance(info, (list, tuple)) else info

        self.last_reward   = float(reward[0]) if hasattr(reward, '__len__') else float(reward)
        self.steps        += 1
        self.total_reward += self.last_reward
        self.done          = bool(done[0]) if hasattr(done, '__len__') else bool(done)
        self.action_taken  = (action[0].tolist()
                              if hasattr(action[0], 'tolist') else list(action[0]))

        for key in ("blackout_count", "blackouts", "fault_count"):
            if ep_info.get(key):
                self.blackouts += int(ep_info[key])
                break

        self.stability = float(ep_info.get("stability_score",
                                            ep_info.get("stability", 1.0)))
        self.stability_history.append(self.stability)
        self.unmet_demand += float(ep_info.get("unmet_demand", 0.0))

        try:
            raw_obs = (self.env.get_original_obs()
                       if hasattr(self.env, 'get_original_obs') else self.obs)
            raw = raw_obs[0] if hasattr(raw_obs, '__len__') else raw_obs
            self.current_demand = _extract_array_obs(raw, 3)
        except Exception:
            pass

        self.current_supply = self.action_taken[:3]
        self.fault_status   = [
            self.current_demand[i] > 0.01 and
            self.current_supply[i] < self.current_demand[i] * 0.6
            for i in range(3)
        ]
        return self.get_ui_state("Step completed.")

    # ------------------------------------------------------------------
    def get_ui_state(self, status_msg="Step completed."):
        reward_str = f"{self.last_reward:.4f}"
        done_str   = "✅ Yes — Episode Complete" if self.done else ""
        env_desc   = self._build_env_description()
        chart      = _make_grid_chart(self.current_demand,
                                      self.current_supply,
                                      self.fault_status)
        return reward_str, done_str, status_msg, env_desc, chart

    # ------------------------------------------------------------------
    def _build_env_description(self):
        pct = int(self.steps / MAX_STEPS * 100)
        lines = [
            f"--- POWER GRID STATE (Step {self.steps}/{MAX_STEPS} - {pct}% complete) ---",
            "",
            "Grid Zones:",
        ]
        for i in range(3):
            d = self.current_demand[i] if i < len(self.current_demand) else 0.33
            s = self.current_supply[i] if i < len(self.current_supply) else 0.33
            fault  = self.fault_status[i] if i < len(self.fault_status) else False
            status = "⚠ FAULT DETECTED" if fault else "🟢 Healthy"
            lines.append(
                f"  Zone {i+1} [{ZONE_TYPES[i]}]: "
                f"demand={d:.3f}, supply={s:.3f}, status={status}"
            )

        stab_icon = "🟢" if self.stability >= 0.75 else ("🟡" if self.stability >= 0.45 else "🔴")
        lines += [
            "",
            "Episode so far:",
            f"  Blackouts:          {int(self.blackouts)}",
            f"  Stability:          {self.stability:.3f} {stab_icon}",
            f"  Total unmet demand: {self.unmet_demand:.3f}",
            f"  Total reward:       {self.total_reward:.2f}",
            "",
            "Task: Allocate power to 3 zones as fractions summing to 1.0.",
            "Priority: Serve Zone 3 (Hospital) first. "
            "Avoid overloads – they cascade into blackouts.",
            "Reply with exactly 3 space-separated floats. Example: 0.20 0.30 0.50",
        ]
        if self.done:
            lines.append("\n✅ Episode complete. Click 'Reset Env' to start a new episode.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _safe_state(self, msg):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=_DARK_BG)
        ax.set_facecolor(_DARK_AX)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="white")
        ax.axis("off")
        plt.tight_layout()
        return ("—", "", msg, "", fig)


# ── global instance ───────────────────────────────────────────────────────────
sim  = GridSimulator()
_ERR = ("Error", "", "⚠ Dependencies not loaded.", "Install required packages.", None)


def ui_reset():
    if not sim.ready:
        return _ERR
    sim.reset()
    return sim.get_ui_state("Environment reset. Ready.")


def ui_ai_step():
    if not sim.ready:
        return _ERR
    if sim.obs is None:
        return sim._safe_state("⚠️ Please click Reset Env first")
    return sim.step()


def ui_auto_run():
    if not sim.ready:
        yield _ERR
        return
    if sim.obs is None:
        yield sim._safe_state("⚠️ Please click Reset Env first")
        return
    while not sim.done and sim.steps < MAX_STEPS:
        yield sim.step()
        time.sleep(0.15)


def ui_take_step(z1, z2, z3):
    if not sim.ready:
        return _ERR
    if sim.obs is None:
        return sim._safe_state("⚠️ Please click Reset Env first")
    return sim.step(action=[z1, z2, z3], manual=True)


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
/* ── page background ── */
body, .gradio-container {
    background-color: #111111 !important;
    color: #e5e5e5 !important;
}
/* ── panel cards ── */
.gr-box, .gr-form, .gr-panel {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
}
/* ── section label colour ── */
.label-wrap span { color: #aaaaaa !important; font-size: 0.78rem; }
/* ── inputs / textareas ── */
textarea, input[type="text"], input[type="number"] {
    background-color: #1e1e1e !important;
    color: #e5e5e5 !important;
    border-color: #333333 !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 0.82rem !important;
}
/* ── sliders ── */
input[type="range"]::-webkit-slider-thumb { background: #f97316 !important; }
input[type="range"]::-webkit-slider-runnable-track { background: #333333 !important; }
/* ── hide Gradio footer ── */
footer { display: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(primary_hue=gr.themes.colors.orange,
                         neutral_hue=gr.themes.colors.slate),
    css=css,
) as demo:

    # ── HEADER ────────────────────────────────────────────────────────────────
    gr.Markdown(
        "## ⚡ GridMind: Power Grid Coordination AI\n"
        "_Interactive demo of the **GridOpsEnv** environment. "
        "Allocate power to prevent cascading blackouts._"
    )

    with gr.Row(equal_height=False):

        # ── LEFT PANEL — Controls ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🎮 Controls")

            slider_z1 = gr.Slider(0, 1, value=0.33, step=0.01,
                                  label="Zone 1 Allocation (Residential)")
            slider_z2 = gr.Slider(0, 1, value=0.33, step=0.01,
                                  label="Zone 2 Allocation (Commercial)")
            slider_z3 = gr.Slider(0, 1, value=0.34, step=0.01,
                                  label="Zone 3 Allocation (Hospital)")

            with gr.Row():
                btn_take = gr.Button("🎮 Take Step",  variant="primary",   size="lg")
                btn_reset = gr.Button("🔄 Reset Env", variant="secondary", size="lg")

            with gr.Row():
                btn_ai   = gr.Button("🤖 AI Step",   variant="primary",   size="sm")
                btn_auto = gr.Button("▶ Auto Run",   variant="primary",   size="sm")

            gr.Markdown("---")

            last_reward = gr.Textbox(
                label="Last Step Reward", value="0.0000", interactive=False)
            done_box = gr.Textbox(
                label="Episode Done", value="", interactive=False)
            status_box = gr.Textbox(
                label="Status",
                value="Ready. Click Reset Env to begin.",
                interactive=False)

        # ── RIGHT PANEL — Visualisation ───────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Grid State")

            with gr.Tabs():
                with gr.Tab("Demand vs Supply"):
                    plot_grid = gr.Plot(show_label=False)

            env_desc = gr.Textbox(
                label="Environment Description",
                value=(
                    "Reset the environment to begin.\n\n"
                    "Task: Allocate power to 3 zones as fractions summing to 1.0.\n"
                    "Priority: Serve Zone 3 (Hospital) first. "
                    "Avoid overloads – they cascade into blackouts.\n"
                    "Reply with exactly 3 space-separated floats. Example: 0.20 0.30 0.50"
                ),
                interactive=False,
                lines=12,
                max_lines=15,
            )

    # ── wiring ────────────────────────────────────────────────────────────────
    _outputs = [last_reward, done_box, status_box, env_desc, plot_grid]

    btn_reset.click(fn=ui_reset,    outputs=_outputs)
    btn_ai.click   (fn=ui_ai_step,  outputs=_outputs)
    btn_auto.click (fn=ui_auto_run, outputs=_outputs)
    btn_take.click (fn=ui_take_step,
                    inputs=[slider_z1, slider_z2, slider_z3],
                    outputs=_outputs)

if __name__ == "__main__":
    demo.launch()