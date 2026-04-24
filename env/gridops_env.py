import numpy as np


class GridOpsEnv:
    """
    Multi-zone power grid environment with bidding-based allocation.
    Gym-style API (no external RL dependencies).

    v2 additions (non-breaking):
      - Negotiation layer with extreme-bid dampening
      - Strategic misreporting detection
      - Delayed cascading failures (failure queue)
      - Long-horizon memory (rolling 10-step window)
      - Coalition signal / bonus
      - Extended info/history metrics
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, num_zones=3, max_time=50, seed=None):
        self.num_zones    = num_zones
        self.max_time     = max_time
        self.mode         = "baseline"   # "baseline" | "selfish" | "coordinated"
        self.reward_mode  = "local"      # "local"    | "global"

        # Reputation hyperparameters
        self.reputation   = np.ones(self.num_zones, dtype=float)
        self.rep_decay    = 0.1
        self.rep_recover  = 0.02
        self.rep_min      = 0.2
        self.rep_max      = 2.0

        # Ethical prioritisation weight (for priority-3 zones in global mode)
        self.ethical_weight = 1.5

        # Coalition bonus threshold and magnitude
        self.coalition_var_threshold = 0.05
        self.coalition_bonus_value   = 2.0

        self.seed(seed)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        self.mode = mode

    def set_reward_mode(self, mode: str):
        self.reward_mode = mode

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.time = 0
        self.reputation = np.ones(self.num_zones, dtype=float)
        self._init_state()
        return self._get_obs(), {}

    def step(self, action=None):
        # 1) Resolve action from mode policy if none provided
        if action is None:
            if self.mode == "selfish":
                action = self.selfish_policy()
            elif self.mode == "advanced":
                action = self.advanced_policy()
            elif self.mode == "coordinated":
                action = self.coordinated_policy()
            else:
                action = self.rng.integers(1, 10, size=self.num_zones)

        action = np.asarray(action, dtype=float)
        if action.shape != (self.num_zones,):
            raise ValueError(
                f"Expected action shape ({self.num_zones},), got {action.shape}"
            )

        bids = np.array(action, dtype=float)

        # 2) Honesty / misreporting check
        misreport_ratio = bids / (self.demand + 1e-8)
        self._misreport_ratio = np.clip(misreport_ratio, 0, 10)
        self._misreport_mask  = misreport_ratio > 1.2

        overbid    = bids > (1.1 * self.demand)
        honesty_pen = (overbid.astype(float) + self._misreport_mask.astype(float)) * 2.0

        self.reputation = np.clip(
            self.reputation
            - self.rep_decay * overbid.astype(float)
            - self.rep_decay * self._misreport_mask.astype(float)
            + self.rep_recover,
            self.rep_min, self.rep_max,
        )
        self._honesty_pen  = honesty_pen
        self._overbid_mask = overbid

        # 3) Coalition detection (before allocation)
        bid_norm = bids / (np.max(bids) + 1e-8)
        self._coalition_active = bool(np.var(bid_norm) < self.coalition_var_threshold)
        self._coalition_bonus  = self.coalition_bonus_value if self._coalition_active else 0.0

        # 4) Negotiation round → dampened allocation weights
        negotiated_weights = self._negotiation_round(bids)

        self.time += 1
        self._apply_action_negotiated(negotiated_weights)
        self._dynamics()
        self._local_rewards = self._compute_local_rewards()

        # 5) Process delayed failure queue
        delayed_triggered = self._process_failure_queue()
        self._delayed_failures_triggered = delayed_triggered

        # 6) Long-horizon memory update
        self._update_memory()

        reward     = self._compute_reward()
        obs        = self._get_obs()
        terminated = False
        truncated  = self.time >= self.max_time
        info       = self._get_info()

        # 7) Append to history
        self.history["reward"].append(reward)
        self.history["blackouts"].append(info["blackouts"])
        self.history["overloads"].append(info["overloads"])
        self.history["imbalance"].append(info["imbalance"])
        self.history["efficiency"].append(info["efficiency"])
        self.history["stability"].append(info["stability_score"])
        self.history["avg_reputation"].append(info["avg_reputation"])
        self.history["honesty_violations"].append(info["honesty_violations"])
        self.history["misreporting_rate"].append(info["misreporting_rate"])
        self.history["coalition_rate"].append(float(self._coalition_active))
        self.history["delayed_failures"].append(info["delayed_failures_triggered"])

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # State Initialisation
    # ------------------------------------------------------------------

    def _init_state(self):
        self.demand      = self.rng.integers(8, 17, size=self.num_zones)   # [8,16]
        self.allocated   = np.zeros(self.num_zones, dtype=float)
        self.priority    = self.rng.integers(1, 4,  size=self.num_zones)   # {1,2,3}
        self.total_power = int(self.rng.integers(25, 41))                  # [25,40]
        self.failed      = np.zeros(self.num_zones, dtype=bool)

        # Delayed failure queue: list of {"delay": int, "power_loss": float}
        self.failure_queue = []

        # Long-horizon memory
        self.memory = {
            "recent_rewards":  [],
            "recent_blackouts": [],
            "recent_imbalance": [],
            "summary": {},
        }

        self.history = {
            "reward": [], "blackouts": [], "overloads": [],
            "imbalance": [], "efficiency": [], "stability": [],
            "avg_reputation": [], "honesty_violations": [],
            "misreporting_rate": [], "coalition_rate": [],
            "delayed_failures": [],
        }

        # Pre-initialise masks so reward/info calls are safe before first step
        self._overload_mask  = np.zeros(self.num_zones, dtype=bool)
        self._blackout_mask  = np.zeros(self.num_zones, dtype=bool)
        self._local_rewards  = np.zeros(self.num_zones, dtype=float)
        self._honesty_pen    = np.zeros(self.num_zones, dtype=float)
        self._overbid_mask   = np.zeros(self.num_zones, dtype=bool)
        self._misreport_mask = np.zeros(self.num_zones, dtype=bool)
        self._misreport_ratio = np.ones(self.num_zones, dtype=float)
        self._coalition_active  = False
        self._coalition_bonus   = 0.0
        self._delayed_failures_triggered = 0

    # ------------------------------------------------------------------
    # Negotiation Layer
    # ------------------------------------------------------------------

    def _negotiation_round(self, bids):
        """
        Dampen extreme bids before allocation.
        Returns normalised weights in [0, 1] shaped (num_zones,).
        """
        bids_safe = np.maximum(bids, 0.0)
        # Normalise by max to squash outlier overbidding
        weights = bids_safe / (np.max(bids_safe) + 1e-8)
        return weights  # used as proportional weights, not absolute bids

    # ------------------------------------------------------------------
    # Action → Allocation (negotiation-aware)
    # ------------------------------------------------------------------

    def _apply_action_negotiated(self, negotiated_weights):
        """Allocate using negotiation weights × reputation (or rep² for advanced)."""
        rep_power = 2 if self.mode == "advanced" else 1
        weights = np.maximum(negotiated_weights, 0.0) * (self.reputation ** rep_power)
        total_w = weights.sum() + 1e-8
        self.allocated = (weights / total_w) * self.total_power

    def _apply_action(self, action):
        """Legacy path kept for backward compatibility."""
        bids    = np.maximum(np.array(action, dtype=float), 0.0)
        weights = bids * self.reputation
        total_w = weights.sum() + 1e-8
        self.allocated = (weights / total_w) * self.total_power

    # ------------------------------------------------------------------
    # Stochastic Dynamics + Failure Queue
    # ------------------------------------------------------------------

    def _dynamics(self):
        # Demand drift
        noise = self.rng.integers(-2, 3, size=self.num_zones)
        self.demand = np.maximum(self.demand + noise, 1)

        # Fault masks
        self._overload_mask = self.allocated > 1.3 * self.demand
        self._blackout_mask = self.allocated < 0.4 * self.demand

        self.failed = np.logical_or(self.failed, self._overload_mask)

        # Queue delayed failures for overloaded zones
        for i in np.where(self._overload_mask)[0]:
            delay      = int(self.rng.integers(1, 4))  # 1–3 steps
            power_loss = float(self.total_power * 0.05)  # 5 % per queued failure
            self.failure_queue.append({"delay": delay, "power_loss": power_loss})

        # Cascade effect / recovery (immediate)
        if self._overload_mask.any():
            self.total_power = max(int(self.total_power * 0.9), 10)
        else:
            self.total_power = min(int(self.total_power * 1.02), 50)

    def _process_failure_queue(self):
        """Decrement delays, apply losses when delay hits 0. Returns count triggered."""
        remaining = []
        triggered = 0
        for event in self.failure_queue:
            event["delay"] -= 1
            if event["delay"] <= 0:
                self.total_power = max(
                    int(self.total_power - event["power_loss"]), 10
                )
                triggered += 1
            else:
                remaining.append(event)
        self.failure_queue = remaining
        return triggered

    # ------------------------------------------------------------------
    # Long-Horizon Memory
    # ------------------------------------------------------------------

    def _update_memory(self):
        reward_now   = self.history["reward"][-1]   if self.history["reward"]   else 0.0
        blackout_now = self.history["blackouts"][-1] if self.history["blackouts"] else 0.0
        imbal_now    = self.history["imbalance"][-1] if self.history["imbalance"] else 0.0

        self.memory["recent_rewards"].append(reward_now)
        self.memory["recent_blackouts"].append(blackout_now)
        self.memory["recent_imbalance"].append(imbal_now)

        # Keep only the last 10 entries
        for key in ("recent_rewards", "recent_blackouts", "recent_imbalance"):
            self.memory[key] = self.memory[key][-10:]

        # Summarise every 10 steps
        if self.time % 10 == 0:
            self.memory["summary"] = {
                "avg_reward":      float(np.mean(self.memory["recent_rewards"])),
                "blackout_rate":   float(np.mean(self.memory["recent_blackouts"])),
                "imbalance_trend": float(np.mean(self.memory["recent_imbalance"])),
            }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        served       = np.minimum(self.allocated, self.demand)
        overload_pen = 5.0 * int(self._overload_mask.sum())

        if self.reward_mode == "local":
            # Selfish agents get amplified served signal but weakened blackout signal
            # → encourages short-term hoarding while system health silently degrades
            if self.mode == "selfish":
                blackout_pen = 2.0 * int(self._blackout_mask.sum())  # reduced from 5
                return float(served.sum() * 1.3 - blackout_pen)      # amplified local gain
            blackout_pen = 5.0 * int(self._blackout_mask.sum())
            return float(served.sum())

        # --- Global reward ---
        blackout_pen = 5.0 * int(self._blackout_mask.sum())

        priority_weights = self.priority.astype(float)
        priority_weights[self.priority == 3] *= self.ethical_weight
        ethical_served = float((served * priority_weights).sum())

        share        = self.allocated / (self.allocated.sum() + 1e-8)
        fairness_pen = 3.0 * float(np.var(share))

        honesty_pen_total = float(self._honesty_pen.sum())

        if self.mode == "advanced":
            # Doubled honesty penalty + stronger coalition bonus (3.0)
            honesty_pen_total *= 2.0
            coalition_bonus = 3.0 if self._coalition_active else 0.0
        else:
            coalition_bonus = self._coalition_bonus

        reward = (
            ethical_served
            - overload_pen
            - blackout_pen
            - fairness_pen
            - honesty_pen_total
            + coalition_bonus
        )
        return float(np.clip(reward, -200, 500))

    def _compute_local_rewards(self):
        return np.minimum(self.allocated, self.demand).astype(float)

    # ------------------------------------------------------------------
    # Observation & Info
    # ------------------------------------------------------------------

    def _get_obs(self):
        obs = {
            "time":        self.time,
            "demand":      self.demand.copy(),
            "allocated":   self.allocated.copy(),
            "priority":    self.priority.copy(),
            "total_power": self.total_power,
        }
        if self.memory["summary"]:
            obs["memory_summary"] = dict(self.memory["summary"])
        return obs

    def _get_info(self):
        alloc    = self.allocated
        share    = alloc / (alloc.sum() + 1e-8)
        imbalance = float(np.var(share))

        served    = np.minimum(alloc, self.demand)
        overloads = int(self._overload_mask.sum())
        blackouts = int(self._blackout_mask.sum())

        return {
            "served":                    float(served.sum()),
            "weighted_served":           float(np.sum(served * self.priority)),
            "overloads":                 overloads,
            "blackouts":                 blackouts,
            "local_rewards":             list(self._local_rewards),
            "imbalance":                 imbalance,
            "efficiency":                float(served.sum() / (self.demand.sum() + 1e-8)),
            "fairness_penalty":          float(3.0 * imbalance),
            "stability_score":           float(1.0 / (1 + blackouts + overloads)),
            "avg_reputation":            float(self.reputation.mean()),
            "min_reputation":            float(self.reputation.min()),
            "honesty_violations":        int(self._overbid_mask.sum()),
            "misreporting_rate":         float(self._misreport_mask.mean()),
            "coalition_rate":            float(self._coalition_active),
            "delayed_failures_triggered": int(self._delayed_failures_triggered),
        }

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------

    def selfish_policy(self):
        alpha = 1.2
        noise = self.rng.integers(0, 3, size=self.num_zones)
        bid   = self.demand * self.priority * alpha + noise
        return np.clip(bid, 0, None).astype(float)

    def coordinated_policy(self):
        weights = self.priority * self.demand * self.reputation
        total   = weights.sum() + 1e-8
        alloc   = (weights / total) * self.total_power
        return alloc.astype(float)

    def advanced_policy(self):
        """Coordinated policy using reputation² — amplifies trust differential."""
        weights = self.priority * self.demand * (self.reputation ** 2)
        total   = weights.sum() + 1e-8
        alloc   = (weights / total) * self.total_power
        return alloc.astype(float)

    # ------------------------------------------------------------------
    # History / Summary helpers
    # ------------------------------------------------------------------

    def get_history(self) -> dict:
        return self.history

    def summarize_episode(self):
        return {
            "avg_reward":          float(np.mean(self.history["reward"])),
            "avg_blackouts":       float(np.mean(self.history["blackouts"])),
            "avg_imbalance":       float(np.mean(self.history["imbalance"])),
            "avg_stability":       float(np.mean(self.history["stability"])),
            "avg_reputation":      float(np.mean(self.history["avg_reputation"])),
            "honesty_violations":  int(np.sum(self.history["honesty_violations"])),
            "misreporting_rate":   float(np.mean(self.history["misreporting_rate"])),
            "coalition_rate":      float(np.mean(self.history["coalition_rate"])),
            "delayed_failures":    int(np.sum(self.history["delayed_failures"])),
        }


# ----------------------------------------------------------------------
# Behavioural validation
# ----------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        ("baseline",    "local"),
        ("selfish",     "global"),
        ("coordinated", "global"),
    ]

    results = []
    for mode, reward_mode in configs:
        env = GridOpsEnv(num_zones=3, max_time=50, seed=42)
        env.set_mode(mode)
        env.set_reward_mode(reward_mode)
        env.reset()

        truncated = False
        while not truncated:
            _, _, _, truncated, _ = env.step()

        summary = env.summarize_episode()
        results.append((mode, reward_mode, summary))

    keys = [
        ("avg_reward",        "avg_reward"),
        ("avg_blackouts",     "blackouts"),
        ("avg_stability",     "stability"),
        ("avg_reputation",    "reputation"),
        ("misreporting_rate", "misreport"),
        ("coalition_rate",    "coalition"),
        ("delayed_failures",  "delayed_fail"),
    ]
    col = 14
    header = f"{'mode':<{col}} {'reward_mode':<{col}}" + "".join(
        f" {label:>{col}}" for _, label in keys
    )
    print(header)
    print("-" * len(header))
    for mode, reward_mode, s in results:
        row = f"{mode:<{col}} {reward_mode:<{col}}"
        for key, _ in keys:
            val = s[key]
            row += f" {val:>{col}.2f}" if isinstance(val, float) else f" {val:>{col}d}"
        print(row)
