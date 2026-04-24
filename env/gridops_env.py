import numpy as np


class GridOpsEnv:
    """
    Multi-zone power grid environment with bidding-based allocation.
    Gym-style API (no external RL dependencies).
    """

    def __init__(self, num_zones=3, max_time=50, seed=None):
        self.num_zones = num_zones
        self.max_time = max_time
        self.seed(seed)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.time = 0
        self._init_state()
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.num_zones,):
            raise ValueError(
                f"Expected action shape ({self.num_zones},), got {action.shape}"
            )

        self.time += 1
        self._apply_action(action)
        self._dynamics()

        reward = self._compute_reward()
        obs = self._get_obs()
        terminated = False
        truncated = self.time >= self.max_time
        info = self._get_info()

        # Append metrics to history
        self.history["reward"].append(reward)
        self.history["blackouts"].append(info["blackouts"])
        self.history["overloads"].append(info["overloads"])

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # State Initialization
    # ------------------------------------------------------------------

    def _init_state(self):
        self.demand = self.rng.integers(8, 17, size=self.num_zones)       # [8, 16]
        self.allocated = np.zeros(self.num_zones, dtype=float)
        self.priority = self.rng.integers(1, 4, size=self.num_zones)      # {1, 2, 3}
        self.total_power = int(self.rng.integers(25, 41))                 # [25, 40]
        self.failed = np.zeros(self.num_zones, dtype=bool)
        self.history = {"reward": [], "blackouts": [], "overloads": []}

        # Initialize masks so _compute_reward/_get_info are safe pre-step
        self._overload_mask = np.zeros(self.num_zones, dtype=bool)
        self._blackout_mask = np.zeros(self.num_zones, dtype=bool)

    # ------------------------------------------------------------------
    # Action → Allocation
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        action = np.clip(action, 0, None)
        total_bids = action.sum() + 1e-8
        self.allocated = (action / total_bids) * self.total_power

    # ------------------------------------------------------------------
    # Stochastic Dynamics
    # ------------------------------------------------------------------

    def _dynamics(self):
        # Demand drift: uniform noise in [-2, +2], floor at 1
        noise = self.rng.integers(-2, 3, size=self.num_zones)
        self.demand = np.maximum(self.demand + noise, 1)

        # Fault conditions
        self._overload_mask = self.allocated > 1.5 * self.demand
        self._blackout_mask = self.allocated < 0.3 * self.demand

        # Zones that overloaded are permanently marked failed
        self.failed = np.logical_or(self.failed, self._overload_mask)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        served = np.minimum(self.allocated, self.demand)
        weighted_served = float(np.sum(served * self.priority))

        overload_pen = 5.0 * int(self._overload_mask.sum())
        blackout_pen = 5.0 * int(self._blackout_mask.sum())

        return weighted_served - overload_pen - blackout_pen

    # ------------------------------------------------------------------
    # Observation & Info
    # ------------------------------------------------------------------

    def _get_obs(self):
        return {
            "time": self.time,
            "demand": self.demand.copy(),
            "allocated": self.allocated.copy(),
            "priority": self.priority.copy(),
            "total_power": self.total_power,
        }

    def _get_info(self):
        served = np.minimum(self.allocated, self.demand)
        return {
            "served": float(served.sum()),
            "weighted_served": float(np.sum(served * self.priority)),
            "overloads": int(self._overload_mask.sum()),
            "blackouts": int(self._blackout_mask.sum()),
        }


# ----------------------------------------------------------------------
# Quick sanity test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    env = GridOpsEnv(num_zones=3, max_time=50, seed=42)
    obs, info = env.reset()
    print(f"Reset  | demand={obs['demand']}  priority={obs['priority']}  "
          f"total_power={obs['total_power']}")
    print("-" * 65)

    for i in range(10):
        action = env.rng.integers(1, 11, size=env.num_zones).astype(float)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i+1:2d} | action={action}  reward={reward:7.2f}  "
            f"overloads={info['overloads']}  blackouts={info['blackouts']}  "
            f"served={info['served']:.2f}"
        )
        if terminated or truncated:
            break
