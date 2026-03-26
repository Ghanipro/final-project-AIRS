from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Action types (semantic)
ACTION_NOOP = 0
ACTION_BLOCK_IP = 1
ACTION_ISOLATE_NODE = 2
ACTION_QUARANTINE_FILE = 3
ACTION_INCREASE_MONITORING = 4
ACTION_PATCH_VULNERABILITY = 5
ACTION_RATE_LIMIT = 6

ACTION_NAMES = {
    ACTION_NOOP: "NO_OP",
    ACTION_BLOCK_IP: "BLOCK_IP",
    ACTION_ISOLATE_NODE: "ISOLATE_NODE",
    ACTION_QUARANTINE_FILE: "QUARANTINE_FILE",
    ACTION_INCREASE_MONITORING: "INCREASE_MONITORING",
    ACTION_PATCH_VULNERABILITY: "PATCH_VULNERABILITY",
    ACTION_RATE_LIMIT: "RATE_LIMIT_TRAFFIC",
}


@dataclass
class AIRSConfig:
    n_nodes: int = 5
    max_steps: int = 150

    # attacker stage transition base probs for c=0..4
    p_adv_base: Tuple[float, float, float, float, float] = (0.05, 0.12, 0.10, 0.08, 0.0)

    # defense multipliers (reduce success)
    m_isolated: float = 0.70
    m_patched: float = 0.40
    m_quarantined: float = 0.55
    m_rate_limited_stage3plus: float = 0.60
    m_block_ip_step: float = 0.60
    m_monitoring: Tuple[float, float, float] = (1.0, 0.85, 0.70)  # mon=0..2

    # regress (stage drop)
    p_regress_base: float = 0.01
    p_regress_bonus_quarantine: float = 0.03
    p_regress_bonus_mon2: float = 0.02

    # patching
    patch_delay_steps: int = 2

    # quarantine
    p_quarantine_success: float = 0.85
    p_quarantine_roll_back_stage: float = 0.60

    # attacker intensity b in [0,1]
    intensity_drift: float = 0.03
    intensity_control_k: float = 0.08

    # termination
    breach_on_stage4: bool = True
    contain_hold_steps: int = 5
    intensity_contained_threshold: float = 0.10

    # costs (normalized)
    cost_noop: float = 0.0
    cost_block_ip: float = 0.05
    cost_isolate: float = 0.20
    cost_quarantine: float = 0.08
    cost_monitoring: float = 0.03
    cost_patch: float = 0.12
    cost_rate_limit: float = 0.10

    # availability penalties (normalized per node)
    avail_isolate: float = 0.50
    avail_rate_limit: float = 0.25
    avail_patching: float = 0.10

    # reward weights
    alpha_security: float = 1.0
    beta_cost: float = 0.5
    gamma_avail: float = 0.7

    # security weights per stage (w[4]=1)
    stage_weight: Tuple[float, float, float, float, float] = (0.0, 0.25, 0.5, 0.75, 1.0)

    breach_penalty: float = 5.0
    contain_bonus: float = 1.0


class AIRSEnv(gym.Env):
    """
    Simulated stage-based MDP for intrusion response.

    Per-node internal state:
      c_i stage in {0..4}
      iso_i in {0,1}
      mon_i in {0,1,2}
      patched_i in {0,1}
      quarantined_i in {0,1}
      rl_i in {0,1}
      patch_timer_i in {0..patch_delay}
      b_i intensity in [0,1]

    Observation:
      per node: [c/4, iso, mon/2, patched, quarantined, rl, patch_timer/patch_delay, b]
      + [t/max_steps]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[AIRSConfig] = None):
        super().__init__()
        self.cfg = config or AIRSConfig()
        if self.cfg.n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")

        self.n = self.cfg.n_nodes
        self.max_steps = self.cfg.max_steps

        # Discrete(1 + 6N): targeted actions per node
        self.action_space = spaces.Discrete(1 + 6 * self.n)

        self.obs_dim = self.n * 8 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(0)
        self._t = 0
        self._contain_counter = 0

        # node arrays
        self.c = np.zeros(self.n, dtype=np.int32)
        self.iso = np.zeros(self.n, dtype=np.int32)
        self.mon = np.zeros(self.n, dtype=np.int32)
        self.patched = np.zeros(self.n, dtype=np.int32)
        self.quarantined = np.zeros(self.n, dtype=np.int32)
        self.rl = np.zeros(self.n, dtype=np.int32)
        self.patch_timer = np.zeros(self.n, dtype=np.int32)
        self.b = np.zeros(self.n, dtype=np.float32)

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _decode_action(self, a: int) -> Tuple[int, Optional[int]]:
        if a == 0:
            return ACTION_NOOP, None
        a -= 1
        block = a // self.n  # 0..5
        idx = a % self.n
        action_type = {
            0: ACTION_BLOCK_IP,
            1: ACTION_ISOLATE_NODE,
            2: ACTION_QUARANTINE_FILE,
            3: ACTION_INCREASE_MONITORING,
            4: ACTION_PATCH_VULNERABILITY,
            5: ACTION_RATE_LIMIT,
        }[block]
        return action_type, int(idx)

    def _get_obs(self) -> np.ndarray:
        patch_div = max(1, self.cfg.patch_delay_steps)
        parts: List[float] = []
        for i in range(self.n):
            parts.extend(
                [
                    float(self.c[i]) / 4.0,
                    float(self.iso[i]),
                    float(self.mon[i]) / 2.0,
                    float(self.patched[i]),
                    float(self.quarantined[i]),
                    float(self.rl[i]),
                    float(self.patch_timer[i]) / float(patch_div),
                    float(self.b[i]),
                ]
            )
        parts.append(float(self._t) / float(self.max_steps))
        return np.array(parts, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        self._t = 0
        self._contain_counter = 0

        self.c[:] = 0
        self.iso[:] = 0
        self.mon[:] = 0
        self.patched[:] = 0
        self.quarantined[:] = 0
        self.rl[:] = 0
        self.patch_timer[:] = 0

        self.b = self._rng.uniform(0.05, 0.25, size=self.n).astype(np.float32)
        if self._rng.random() < 0.35:
            j = int(self._rng.integers(0, self.n))
            self.c[j] = 1

        return self._get_obs(), {"config": asdict(self.cfg)}

    def _action_cost(self, action_type: int) -> float:
        c = self.cfg
        return {
            ACTION_NOOP: c.cost_noop,
            ACTION_BLOCK_IP: c.cost_block_ip,
            ACTION_ISOLATE_NODE: c.cost_isolate,
            ACTION_QUARANTINE_FILE: c.cost_quarantine,
            ACTION_INCREASE_MONITORING: c.cost_monitoring,
            ACTION_PATCH_VULNERABILITY: c.cost_patch,
            ACTION_RATE_LIMIT: c.cost_rate_limit,
        }[action_type]

    def _availability_loss(self) -> float:
        c = self.cfg
        loss = 0.0
        for i in range(self.n):
            loss += c.avail_isolate * float(self.iso[i])
            loss += c.avail_rate_limit * float(self.rl[i])
            loss += c.avail_patching * float(self.patch_timer[i] > 0)
        denom = self.n * max(1e-6, (c.avail_isolate + c.avail_rate_limit + c.avail_patching))
        return float(np.clip(loss / denom, 0.0, 1.0))

    def _security_loss(self) -> float:
        w = self.cfg.stage_weight
        return float(np.clip(np.mean([w[int(x)] for x in self.c]), 0.0, 1.0))

    def _apply_defender_action(self, action_type: int, idx: Optional[int]) -> None:
        c = self.cfg
        if action_type == ACTION_NOOP:
            return
        if idx is None:
            raise ValueError("Targeted action without node index")

        i = int(idx)

        if action_type == ACTION_BLOCK_IP:
            self.b[i] = float(max(0.0, self.b[i] - 0.20))

        elif action_type == ACTION_ISOLATE_NODE:
            self.iso[i] = 1

        elif action_type == ACTION_QUARANTINE_FILE:
            if self._rng.random() < c.p_quarantine_success:
                self.quarantined[i] = 1
                if self._rng.random() < c.p_quarantine_roll_back_stage:
                    self.c[i] = max(0, int(self.c[i]) - 1)

        elif action_type == ACTION_INCREASE_MONITORING:
            self.mon[i] = min(2, int(self.mon[i]) + 1)

        elif action_type == ACTION_PATCH_VULNERABILITY:
            if self.patched[i] == 0 and self.patch_timer[i] == 0:
                self.patch_timer[i] = c.patch_delay_steps

        elif action_type == ACTION_RATE_LIMIT:
            self.rl[i] = 1

    def _advance_patch_timers(self) -> None:
        for i in range(self.n):
            if self.patch_timer[i] > 0:
                self.patch_timer[i] -= 1
                if self.patch_timer[i] == 0:
                    self.patched[i] = 1

    def _attacker_step(self, blocked_idx: Optional[int]) -> None:
        c = self.cfg
        for i in range(self.n):
            stage = int(self.c[i])
            p = float(c.p_adv_base[stage])

            p *= c.m_monitoring[int(self.mon[i])]
            if self.iso[i] == 1:
                p *= c.m_isolated
            if self.patched[i] == 1:
                p *= c.m_patched
            if self.quarantined[i] == 1:
                p *= c.m_quarantined
            if self.rl[i] == 1 and stage >= 3:
                p *= c.m_rate_limited_stage3plus
            if blocked_idx is not None and i == blocked_idx:
                p *= c.m_block_ip_step

            p = float(np.clip(p, 0.0, 1.0))
            if stage < 4 and self._rng.random() < p:
                self.c[i] = stage + 1

            pr = c.p_regress_base
            if self.quarantined[i] == 1:
                pr += c.p_regress_bonus_quarantine
            if self.mon[i] == 2:
                pr += c.p_regress_bonus_mon2
            pr = float(np.clip(pr, 0.0, 1.0))
            if int(self.c[i]) > 0 and self._rng.random() < pr:
                self.c[i] = max(0, int(self.c[i]) - 1)

            controls = (
                0.9 * float(self.iso[i])
                + 0.5 * float(self.quarantined[i])
                + 0.4 * float(self.rl[i])
                + 0.2 * float(self.mon[i])
                + (0.3 if (blocked_idx is not None and i == blocked_idx) else 0.0)
            )
            noise = float(self._rng.normal(0.0, 0.02))
            self.b[i] = float(
                np.clip(
                    self.b[i] + c.intensity_drift + noise - c.intensity_control_k * controls,
                    0.0,
                    1.0,
                )
            )

    def step(self, action: int):
        action_type, idx = self._decode_action(int(action))
        self._t += 1

        action_cost = self._action_cost(action_type)
        self._apply_defender_action(action_type, idx)
        self._advance_patch_timers()

        blocked_idx = idx if action_type == ACTION_BLOCK_IP else None
        self._attacker_step(blocked_idx=blocked_idx)

        breach = bool(self.cfg.breach_on_stage4 and np.any(self.c >= 4))
        contained_now = bool(np.all(self.c == 0) and np.all(self.b < self.cfg.intensity_contained_threshold))

        if contained_now:
            self._contain_counter += 1
        else:
            self._contain_counter = 0

        contained = self._contain_counter >= self.cfg.contain_hold_steps
        terminated = breach or contained
        truncated = self._t >= self.max_steps

        security_loss = self._security_loss()
        availability_loss = self._availability_loss()

        reward = (
            -self.cfg.alpha_security * security_loss
            -self.cfg.beta_cost * action_cost
            -self.cfg.gamma_avail * availability_loss
        )
        if breach:
            reward -= self.cfg.breach_penalty
        if contained:
            reward += self.cfg.contain_bonus

        info: Dict[str, Any] = {
            "action_type": action_type,
            "action_name": ACTION_NAMES[action_type],
            "target_node": idx,
            "breach": breach,
            "contained": contained,
            "security_loss": security_loss,
            "availability_loss": availability_loss,
            "action_cost": action_cost,
            "t": self._t,
            "max_stage": int(np.max(self.c)),
        }
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info
