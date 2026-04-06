from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# Action types
ACTION_NOOP = 0
ACTION_BLOCK_IP = 1
ACTION_ISOLATE_NODE = 2
ACTION_QUARANTINE_FILE = 3
ACTION_INCREASE_MONITORING = 4
ACTION_PATCH_VULNERABILITY = 5
ACTION_RATE_LIMIT = 6


@dataclass
class AIRSConfig:
    # Enterprise defaults
    n_nodes: int = 10
    max_steps: int = 150
    patch_delay_steps: int = 2

    # attacker base probabilities
    p_adv_base: List[float] = None  # per stage
    p_regress_base: float = 0.01

    # defense multipliers
    m_isolated: float = 0.7
    m_patched: float = 0.4
    m_quarantined: float = 0.55
    m_rate_limited_stage3plus: float = 0.6
    m_block_ip_step: float = 0.6
    m_monitoring: List[float] = None  # per monitoring level

    # quarantine + patch
    p_quarantine_success: float = 0.85
    p_quarantine_roll_back_stage: float = 0.6
    p_patch_heal: float = 0.2

    # intensity dynamics
    intensity_drift: float = 0.03
    intensity_control_k: float = 0.08

    # termination
    breach_on_stage4: bool = True
    contain_hold_steps: int = 5
    intensity_contained_threshold: float = 0.1

    # reward
    cost_noop: float = 0.0
    cost_block_ip: float = 0.05
    cost_isolate: float = 0.2
    cost_quarantine: float = 0.08
    cost_monitoring: float = 0.03
    cost_patch: float = 0.12
    cost_rate_limit: float = 0.1

    avail_isolate: float = 0.5
    avail_rate_limit: float = 0.25
    avail_patching: float = 0.1

    alpha_security: float = 1.0
    beta_cost: float = 0.5
    gamma_avail: float = 0.7

    stage_weight: List[float] = None
    breach_penalty: float = 5.0
    contain_bonus: float = 1.0

    # enterprise topology
    edges: List[Tuple[int, int]] = None
    zones: List[int] = None
    node_types: List[int] = None
    criticality: List[float] = None

    # zone multipliers (attack progression)
    zone_adv_mul: Dict[int, float] = None

    def __post_init__(self):
        if self.p_adv_base is None:
            self.p_adv_base = [0.05, 0.12, 0.10, 0.08, 0.0]
        if self.m_monitoring is None:
            self.m_monitoring = [1.0, 0.85, 0.7]
        if self.stage_weight is None:
            self.stage_weight = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Enterprise zones
        # 0=DMZ, 1=User, 2=Data, 3=Admin, 4=Security
        if self.zones is None:
            self.zones = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]

        # Node types (0=web,1=app,2=workstation,3=db,4=file,5=dc,6=adminpc,7=siem)
        if self.node_types is None:
            self.node_types = [0, 1, 2, 2, 2, 3, 4, 5, 6, 7]

        if self.criticality is None:
            self.criticality = [0.6, 0.6, 0.4, 0.4, 0.4, 1.0, 0.8, 1.0, 0.7, 0.5]

        # Zone-specific attack multiplier
        if self.zone_adv_mul is None:
            self.zone_adv_mul = {
                0: 1.2,  # DMZ
                1: 1.0,  # User LAN
                2: 0.9,  # Data zone
                3: 0.8,  # Admin zone
                4: 0.6,  # Security
            }

        if self.edges is None:
            self.edges = [
                (0, 1), (0, 2), (1, 2),
                (2, 3), (3, 4),
                (3, 5), (4, 6),
                (5, 6), (5, 7),
                (7, 8),
                (9, 0), (9, 2), (9, 5),
            ]


class AIRSEnv(Env):
    def __init__(self, config: AIRSConfig):
        super().__init__()
        self.cfg = config
        self.n = self.cfg.n_nodes

        # action space: 1 + 6*N
        self.action_space = Discrete(1 + 6 * self.n)

        # observation space: per node 10 features + 1 global time
        # [stage/4, iso, mon/2, patched, quarantined, rate_limit, patch_timer/patch_delay, intensity, zone/4, criticality]
        self.obs_dim = self.n * 10 + 1
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # graph adjacency
        self.adj = [[] for _ in range(self.n)]
        for u, v in self.cfg.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        self._reset_state()
        self.np_random = None

    def _reset_state(self):
        self.c = np.zeros(self.n, dtype=np.int32)   # compromise stage
        self.iso = np.zeros(self.n, dtype=np.int32)
        self.mon = np.zeros(self.n, dtype=np.int32)
        self.patched = np.zeros(self.n, dtype=np.int32)
        self.quar = np.zeros(self.n, dtype=np.int32)
        self.rl = np.zeros(self.n, dtype=np.int32)
        self.patch_timer = np.zeros(self.n, dtype=np.int32)
        self.b = np.zeros(self.n, dtype=np.float32)
        self._t = 0
        self._contained_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()

        # initialize one DMZ/User node with attacker intensity
        starters = self.np_random.choice([0, 1, 2], size=1, replace=False)
        for i in starters:
            self.b[i] = self.np_random.uniform(0.2, 0.4)
            self.c[i] = 1

        return self._get_obs(), {}

    def _decode_action(self, action: int):
        if action == 0:
            return ACTION_NOOP, None
        action -= 1
        block = action // self.n
        node = action % self.n
        action_type = block + 1
        return action_type, node

    def _apply_action(self, action_type: int, node: int):
        if action_type == ACTION_NOOP or node is None:
            return 0.0, 0.0

        cost = 0.0
        avail_loss = 0.0

        if action_type == ACTION_BLOCK_IP:
            cost = self.cfg.cost_block_ip
            self.b[node] *= self.cfg.m_block_ip_step

        elif action_type == ACTION_ISOLATE_NODE:
            cost = self.cfg.cost_isolate
            avail_loss = self.cfg.avail_isolate
            self.iso[node] = 1

        elif action_type == ACTION_QUARANTINE_FILE:
            cost = self.cfg.cost_quarantine
            self.quar[node] = 1

        elif action_type == ACTION_INCREASE_MONITORING:
            cost = self.cfg.cost_monitoring
            self.mon[node] = min(2, self.mon[node] + 1)

        elif action_type == ACTION_PATCH_VULNERABILITY:
            cost = self.cfg.cost_patch
            avail_loss = self.cfg.avail_patching
            self.patch_timer[node] = self.cfg.patch_delay_steps

        elif action_type == ACTION_RATE_LIMIT:
            cost = self.cfg.cost_rate_limit
            avail_loss = self.cfg.avail_rate_limit
            self.rl[node] = 1

        return cost, avail_loss

    def _advance_patch_timers(self):
        for i in range(self.n):
            if self.patch_timer[i] > 0:
                self.patch_timer[i] -= 1
                if self.patch_timer[i] == 0:
                    self.patched[i] = 1

    def _attack_step(self):
        rng = self.np_random

        new_c = self.c.copy()
        new_b = self.b.copy()

        for i in range(self.n):
            if self.iso[i] == 1:
                continue

            stage = self.c[i]
            if stage >= 4:
                continue

            # base advance prob
            p_adv = self.cfg.p_adv_base[stage]

            # zone multiplier
            z = self.cfg.zones[i]
            p_adv *= self.cfg.zone_adv_mul.get(z, 1.0)

            # defense multipliers
            if self.patched[i]:
                p_adv *= self.cfg.m_patched
            if self.quar[i]:
                p_adv *= self.cfg.m_quarantined
            if self.rl[i] and stage >= 3:
                p_adv *= self.cfg.m_rate_limited_stage3plus
            p_adv *= self.cfg.m_monitoring[self.mon[i]]

            # reduced lateral pressure
            neighbor_pressure = 0.0
            for j in self.adj[i]:
                neighbor_pressure += 0.02 * self.c[j]
            p_adv = min(1.0, p_adv + neighbor_pressure)

            # progression
            if rng.random() < p_adv:
                new_c[i] = min(4, stage + 1)

            # quarantine success rollback
            if self.quar[i] and rng.random() < self.cfg.p_quarantine_success:
                if rng.random() < self.cfg.p_quarantine_roll_back_stage:
                    new_c[i] = max(0, new_c[i] - 1)
                new_b[i] = max(0.0, new_b[i] - 0.2)

            # regression bonus from monitoring/quarantine
            p_regress = self.cfg.p_regress_base
            if self.quar[i]:
                p_regress += 0.03
            if self.mon[i] == 2:
                p_regress += 0.02
            if rng.random() < p_regress and new_c[i] > 0:
                new_c[i] -= 1

            # patch can heal a stage
            if self.patched[i] and new_c[i] > 0 and rng.random() < self.cfg.p_patch_heal:
                new_c[i] -= 1

            # intensity dynamics
            delta = self.cfg.intensity_drift - self.cfg.intensity_control_k * (
                self.iso[i] + self.quar[i] + self.rl[i] + self.patched[i]
            )
            new_b[i] = np.clip(new_b[i] + delta, 0.0, 1.0)

        self.c = new_c
        self.b = new_b

    def _get_obs(self):
        feats = []
        for i in range(self.n):
            stage = self.c[i] / 4.0
            iso = self.iso[i]
            mon = self.mon[i] / 2.0
            patched = self.patched[i]
            quar = self.quar[i]
            rl = self.rl[i]
            pt = self.patch_timer[i] / max(1, self.cfg.patch_delay_steps)
            intensity = self.b[i]
            zone = self.cfg.zones[i] / 4.0
            critical = self.cfg.criticality[i]

            feats.extend([stage, iso, mon, patched, quar, rl, pt, intensity, zone, critical])

        feats.append(self._t / max(1, self.cfg.max_steps))
        return np.array(feats, dtype=np.float32)

    def step(self, action: int):
        self._t += 1

        action_type, node = self._decode_action(action)
        action_cost, avail_loss = self._apply_action(action_type, node)
        self._advance_patch_timers()

        # attacker dynamics
        self._attack_step()

        # security loss weighted by criticality
        sec_loss = 0.0
        for i in range(self.n):
            sec_loss += self.cfg.stage_weight[self.c[i]] * self.cfg.criticality[i]

        reward = -self.cfg.alpha_security * sec_loss \
                 - self.cfg.beta_cost * action_cost \
                 - self.cfg.gamma_avail * avail_loss

        breach = False
        if self.cfg.breach_on_stage4:
            breach = np.any(self.c >= 4)

        contained = np.all(self.c == 0) and np.max(self.b) < self.cfg.intensity_contained_threshold
        if contained:
            self._contained_steps += 1
        else:
            self._contained_steps = 0

        terminated = breach or (self._contained_steps >= self.cfg.contain_hold_steps)
        truncated = self._t >= self.cfg.max_steps

        if breach:
            reward -= self.cfg.breach_penalty
        if self._contained_steps >= self.cfg.contain_hold_steps:
            reward += self.cfg.contain_bonus

        info = {
            "breach": breach,
            "contained": self._contained_steps >= self.cfg.contain_hold_steps,
            "max_stage": int(self.c.max()),
            "security_loss": sec_loss,
            "availability_loss": avail_loss,
            "action_cost": action_cost,
        }

        return self._get_obs(), reward, terminated, truncated, info