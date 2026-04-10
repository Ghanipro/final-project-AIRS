from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from src.environment.airs_env import AIRSConfig, AIRSEnv


BluePolicyFn = Callable[[np.ndarray, "CyberBattleState"], int]
RedPolicyFn = Callable[[np.ndarray, "CyberBattleState"], int]


RED_ACTION_NOOP = 0
RED_ACTION_SCAN = 1
RED_ACTION_EXPLOIT = 2
RED_ACTION_PIVOT = 3
RED_ACTION_PERSIST = 4
RED_ACTION_EXFILTRATE = 5


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class CyberBattleConfig(AIRSConfig):
    # Red-side dynamics
    red_scan_gain: float = 0.08
    red_exploit_gain: float = 0.28
    red_pivot_gain: float = 0.18
    red_persist_gain: float = 0.10
    red_exfil_gain: float = 0.55
    red_stealth_decay: float = 0.02
    vulnerability_drift: float = 0.012
    detection_base: float = 0.04
    detection_gain: float = 0.16
    detect_stealth_penalty: float = 0.10
    red_pressure_decay: float = 0.05
    red_attack_cost: float = 0.02

    # Reward balance
    red_breach_bonus: float = 4.0
    red_damage_weight: float = 1.0
    red_stealth_weight: float = 0.25
    red_persistence_weight: float = 0.15
    red_detection_penalty: float = 0.6

    blue_breach_penalty: float = 25.0
    blue_containment_reward: float = 2.0
    blue_detection_reward: float = 0.20

    def __post_init__(self):
        super().__post_init__()


class CyberBattleState:
    def __init__(self, cfg: CyberBattleConfig):
        self.cfg = cfg
        self.n = cfg.n_nodes
        self.adj = [[] for _ in range(self.n)]
        for u, v in self.cfg.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        self.rng: np.random.Generator = np.random.default_rng()
        self._reset_state()

    def _reset_state(self):
        self.c = np.zeros(self.n, dtype=np.int32)
        self.iso = np.zeros(self.n, dtype=np.int32)
        self.mon = np.zeros(self.n, dtype=np.int32)
        self.patched = np.zeros(self.n, dtype=np.int32)
        self.quar = np.zeros(self.n, dtype=np.int32)
        self.rl = np.zeros(self.n, dtype=np.int32)
        self.patch_timer = np.zeros(self.n, dtype=np.int32)

        self.vuln = np.zeros(self.n, dtype=np.float32)
        self.red_pressure = np.zeros(self.n, dtype=np.float32)
        self.red_stealth = np.zeros(self.n, dtype=np.float32)
        self.detection = np.zeros(self.n, dtype=np.float32)
        self.availability = np.ones(self.n, dtype=np.float32)

        self.b = np.zeros(self.n, dtype=np.float32)
        self._t = 0
        self._contained_steps = 0
        self._breach = False
        self._breach_node: Optional[int] = None

    def reset(self, seed: Optional[int] = None):
        self.rng, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()

        self.vuln = self.rng.uniform(0.35, 0.65, size=self.n).astype(np.float32)
        starters = self.rng.choice(np.arange(min(3, self.n)), size=1, replace=False)
        for i in starters:
            self.c[i] = 1
            self.b[i] = float(self.rng.uniform(0.25, 0.45))
            self.red_pressure[i] = 0.25
            self.red_stealth[i] = 0.10

    def _decode_blue_action(self, action: int) -> Tuple[int, Optional[int]]:
        if action == 0:
            return 0, None
        action -= 1
        block = action // self.n
        node = action % self.n
        return block + 1, node

    def _decode_red_action(self, action: int) -> Tuple[int, Optional[int]]:
        if action == 0:
            return RED_ACTION_NOOP, None
        action -= 1
        block = action // self.n
        node = action % self.n
        return block + 1, node

    def _apply_blue_action(self, action: int) -> Dict[str, float]:
        action_type, node = self._decode_blue_action(action)
        cost = 0.0
        avail_loss = 0.0

        if action_type == 0 or node is None:
            return {"blue_cost": 0.0, "blue_avail_loss": 0.0, "blue_action_type": 0}

        if action_type == 1:
            cost = self.cfg.cost_block_ip
            self.b[node] *= self.cfg.m_block_ip_step

        elif action_type == 2:
            cost = self.cfg.cost_isolate
            avail_loss = self.cfg.avail_isolate
            self.iso[node] = 1

        elif action_type == 3:
            cost = self.cfg.cost_quarantine
            self.quar[node] = 1

        elif action_type == 4:
            cost = self.cfg.cost_monitoring
            self.mon[node] = min(2, self.mon[node] + 1)

        elif action_type == 5:
            cost = self.cfg.cost_patch
            avail_loss = self.cfg.avail_patching
            self.patch_timer[node] = self.cfg.patch_delay_steps

        elif action_type == 6:
            cost = self.cfg.cost_rate_limit
            avail_loss = self.cfg.avail_rate_limit
            self.rl[node] = 1

        return {"blue_cost": cost, "blue_avail_loss": avail_loss, "blue_action_type": action_type}

    def _apply_red_action(self, action: int) -> Dict[str, float]:
        action_type, node = self._decode_red_action(action)
        cost = 0.0
        breach_delta = 0.0

        if action_type == RED_ACTION_NOOP or node is None:
            return {"red_cost": 0.0, "red_breach_delta": 0.0, "red_action_type": 0}

        if action_type == RED_ACTION_SCAN:
            self.red_pressure[node] = _clip01(self.red_pressure[node] + self.cfg.red_scan_gain)

        elif action_type == RED_ACTION_EXPLOIT:
            if self.iso[node] == 0:
                exploit_prob = _sigmoid(2.0 * self.vuln[node] + 1.2 * self.red_pressure[node] - 0.8 * self.detection[node])
                if self.rng.random() < exploit_prob:
                    self.c[node] = max(self.c[node], 1)
                    self.red_pressure[node] = _clip01(self.red_pressure[node] + self.cfg.red_exploit_gain)
                    self.red_stealth[node] = _clip01(self.red_stealth[node] + 0.04)

        elif action_type == RED_ACTION_PIVOT:
            if self.c[node] > 0:
                for neighbor in self.adj[node]:
                    self.red_pressure[neighbor] = _clip01(self.red_pressure[neighbor] + self.cfg.red_pivot_gain)

        elif action_type == RED_ACTION_PERSIST:
            if self.c[node] > 0:
                self.red_stealth[node] = _clip01(self.red_stealth[node] + self.cfg.red_persist_gain)
                self.red_pressure[node] = _clip01(self.red_pressure[node] + 0.04)

        elif action_type == RED_ACTION_EXFILTRATE:
            if self.c[node] >= 3 and self.iso[node] == 0:
                critical = float(self.cfg.criticality[node])
                breach_prob = _sigmoid(1.4 * self.c[node] + 1.0 * self.red_pressure[node] + 0.8 * critical - 1.2 * self.detection[node])
                if self.rng.random() < breach_prob:
                    self._breach = True
                    self._breach_node = node
                    breach_delta = 1.0

        cost = self.cfg.red_attack_cost
        return {"red_cost": cost, "red_breach_delta": breach_delta, "red_action_type": action_type}

    def _advance_patch_timers(self):
        for i in range(self.n):
            if self.patch_timer[i] > 0:
                self.patch_timer[i] -= 1
                if self.patch_timer[i] == 0:
                    self.patched[i] = 1
                    self.vuln[i] = max(0.0, self.vuln[i] - 0.25)

    def _advance_dynamics(self):
        new_c = self.c.copy()
        new_b = self.b.copy()

        for i in range(self.n):
            if self.iso[i] == 1:
                self.availability[i] = max(0.0, self.availability[i] - 0.02)
                continue

            stage = int(self.c[i])
            if stage >= 4:
                continue

            neighborhood_pressure = float(np.sum(self.red_pressure[self.adj[i]]) * 0.10)
            p_adv = self.cfg.p_adv_base[stage]
            p_adv *= self.cfg.zone_adv_mul.get(self.cfg.zones[i], 1.0)
            p_adv *= 0.65 + 0.35 * float(self.vuln[i])
            p_adv *= 1.0 + float(self.red_pressure[i]) + neighborhood_pressure

            if self.patched[i]:
                p_adv *= self.cfg.m_patched
            if self.quar[i]:
                p_adv *= self.cfg.m_quarantined
            if self.rl[i] and stage >= 3:
                p_adv *= self.cfg.m_rate_limited_stage3plus
            p_adv *= self.cfg.m_monitoring[self.mon[i]]
            p_adv = min(1.0, p_adv)

            if self.rng.random() < p_adv:
                new_c[i] = min(4, stage + 1)

            if self.quar[i] and self.rng.random() < self.cfg.p_quarantine_success:
                if self.rng.random() < self.cfg.p_quarantine_roll_back_stage:
                    new_c[i] = max(0, new_c[i] - 1)
                new_b[i] = max(0.0, new_b[i] - 0.15)

            p_regress = self.cfg.p_regress_base
            p_regress += 0.03 if self.quar[i] else 0.0
            p_regress += 0.02 if self.mon[i] == 2 else 0.0
            if self.rng.random() < p_regress and new_c[i] > 0:
                new_c[i] -= 1

            if self.patched[i] and new_c[i] > 0 and self.rng.random() < self.cfg.p_patch_heal:
                new_c[i] -= 1

            delta = self.cfg.intensity_drift - self.cfg.intensity_control_k * (
                self.iso[i] + self.quar[i] + self.rl[i] + self.patched[i]
            )
            new_b[i] = np.clip(new_b[i] + delta, 0.0, 1.0)

            # Vulnerability evolves slowly and can be reduced by patching.
            vul_drift = self.cfg.vulnerability_drift - 0.06 * self.patched[i] - 0.03 * self.quar[i]
            self.vuln[i] = np.clip(self.vuln[i] + vul_drift, 0.0, 1.0)

            # Detection improves with monitoring and degrades with stealth.
            p_detect = self.cfg.detection_base + self.cfg.detection_gain * (0.15 * self.mon[i] + 0.25 * new_c[i] + 0.20 * self.red_pressure[i])
            p_detect -= self.cfg.detect_stealth_penalty * self.red_stealth[i]
            p_detect = float(np.clip(p_detect, 0.0, 1.0))
            if self.rng.random() < p_detect:
                self.detection[i] = np.clip(self.detection[i] + 0.20, 0.0, 1.0)
            else:
                self.detection[i] = max(0.0, self.detection[i] - 0.03)

            self.red_stealth[i] = max(0.0, self.red_stealth[i] - self.cfg.red_stealth_decay)
            self.red_pressure[i] = max(0.0, self.red_pressure[i] - self.cfg.red_pressure_decay)

            # Availability is affected by compromise and defensive interventions.
            self.availability[i] = np.clip(
                1.0 - 0.15 * new_c[i] - 0.10 * self.iso[i] - 0.08 * self.quar[i] - 0.05 * self.rl[i],
                0.0,
                1.0,
            )

        self.c = new_c
        self.b = new_b

    def _security_loss(self) -> float:
        sec_loss = 0.0
        for i in range(self.n):
            sec_loss += self.cfg.stage_weight[self.c[i]] * self.cfg.criticality[i]
        return float(sec_loss)

    def _blue_reward(self, prev_sec_loss: float, blue_cost: float, avail_loss: float, blue_action_type: int) -> float:
        sec_loss = self._security_loss()
        sec_delta = prev_sec_loss - sec_loss
        critical_mask = np.array(self.cfg.criticality) >= 0.9
        critical_stage3plus = int(np.sum((self.c >= 3) & critical_mask))
        contained = np.all(self.c == 0) and np.max(self.b) < self.cfg.intensity_contained_threshold and np.max(self.red_pressure) < 0.10

        reward = (
            self.cfg.deescalation_reward_scale * sec_delta
            - self.cfg.alpha_security * self.cfg.step_security_weight * sec_loss
            - self.cfg.beta_cost * blue_cost
            - self.cfg.gamma_avail * avail_loss
            - self.cfg.critical_stage3_penalty * critical_stage3plus
            + self.cfg.survival_bonus
        )

        if self._breach:
            reward -= self.cfg.blue_breach_penalty
        if contained:
            reward += self.cfg.blue_containment_reward
        if np.any(self.detection > 0.3):
            reward += self.cfg.blue_detection_reward
        return float(reward)

    def _red_reward(self, prev_sec_loss: float, red_cost: float, breach_delta: float) -> float:
        sec_loss = self._security_loss()
        damage_gain = sec_loss - prev_sec_loss
        stealth_bonus = float(np.mean(self.red_stealth))
        persistence_bonus = float(np.mean(self.red_pressure))
        detection_penalty = float(np.mean(self.detection))

        reward = (
            self.cfg.red_damage_weight * damage_gain
            + self.cfg.red_breach_bonus * breach_delta
            + self.cfg.red_stealth_weight * stealth_bonus
            + self.cfg.red_persistence_weight * persistence_bonus
            - self.cfg.red_detection_penalty * detection_penalty
            - red_cost
        )
        return float(reward)

    def _blue_obs(self) -> np.ndarray:
        feats = []
        for i in range(self.n):
            stage = self.c[i] / 4.0
            vuln = self.vuln[i]
            intensity = self.b[i]
            red_pressure = self.red_pressure[i]
            stealth = self.red_stealth[i]
            iso = self.iso[i]
            mon = self.mon[i] / 2.0
            patched = self.patched[i]
            quar = self.quar[i]
            rl = self.rl[i]
            pt = self.patch_timer[i] / max(1, self.cfg.patch_delay_steps)
            detection = self.detection[i]
            avail = self.availability[i]
            critical = self.cfg.criticality[i]

            feats.extend([
                stage, vuln, intensity, red_pressure, stealth,
                iso, mon, patched, quar, rl, pt, detection, avail, critical,
            ])

        feats.append(self._t / max(1, self.cfg.max_steps))
        return np.array(feats, dtype=np.float32)

    def _red_obs(self) -> np.ndarray:
        feats = []
        for i in range(self.n):
            stage = self.c[i] / 4.0
            vuln = self.vuln[i]
            intensity = self.b[i]
            red_pressure = self.red_pressure[i]
            stealth = self.red_stealth[i]
            iso = self.iso[i]
            mon = self.mon[i] / 2.0
            detection = self.detection[i]
            critical = self.cfg.criticality[i]
            zone = self.cfg.zones[i] / 4.0

            feats.extend([
                stage, vuln, intensity, red_pressure, stealth,
                iso, mon, detection, critical, zone,
            ])

        feats.append(self._t / max(1, self.cfg.max_steps))
        return np.array(feats, dtype=np.float32)

    def step(self, blue_action: int, red_action: int):
        self._t += 1
        prev_sec_loss = self._security_loss()

        red_info = self._apply_red_action(red_action)
        blue_info = self._apply_blue_action(blue_action)
        self._advance_patch_timers()
        self._advance_dynamics()

        sec_loss = self._security_loss()
        critical_mask = np.array(self.cfg.criticality) >= 0.9
        breach = bool(self._breach or np.any((self.c >= 4) & critical_mask))

        contained = np.all(self.c == 0) and np.max(self.b) < self.cfg.intensity_contained_threshold
        if contained:
            self._contained_steps += 1
        else:
            self._contained_steps = 0

        blue_reward = self._blue_reward(prev_sec_loss, blue_info["blue_cost"], blue_info["blue_avail_loss"], blue_info["blue_action_type"])
        red_reward = self._red_reward(prev_sec_loss, red_info["red_cost"], red_info["red_breach_delta"])

        terminated = breach or (self._contained_steps >= self.cfg.contain_hold_steps)
        truncated = self._t >= self.cfg.max_steps

        info = {
            "breach": breach,
            "contained": self._contained_steps >= self.cfg.contain_hold_steps,
            "max_stage": int(self.c.max()),
            "security_loss": sec_loss,
            "availability_loss": float(np.mean(1.0 - self.availability)),
            "blue_cost": blue_info["blue_cost"],
            "blue_avail_loss": blue_info["blue_avail_loss"],
            "red_cost": red_info["red_cost"],
            "red_action_type": red_info["red_action_type"],
            "blue_action_type": blue_info["blue_action_type"],
            "breach_node": self._breach_node,
        }

        return self._blue_obs(), self._red_obs(), blue_reward, red_reward, terminated, truncated, info


def heuristic_blue_policy(state: CyberBattleState) -> int:
    critical = np.array(state.cfg.criticality) >= 0.9

    if np.any((state.c >= 3) & critical):
        i = int(np.argmax(state.c * state.cfg.criticality))
        return 1 + 1 * state.n + i

    if np.any(state.c >= 2):
        i = int(np.argmax(state.c))
        if state.quar[i] == 0:
            return 1 + 2 * state.n + i
        return 1 + 4 * state.n + i

    if np.any(state.c == 1):
        i = int(np.argmax(state.b))
        if state.b[i] >= 0.35:
            return 1 + 0 * state.n + i
        return 1 + 3 * state.n + i

    i = int(np.argmax(state.red_pressure + state.b))
    return 1 + 3 * state.n + i


def heuristic_red_policy(state: CyberBattleState) -> int:
    critical = np.array(state.cfg.criticality) >= 0.9
    compromised_critical = np.where((state.c >= 2) & critical)[0]
    if len(compromised_critical) > 0:
        i = int(compromised_critical[np.argmax(state.c[compromised_critical])])
        return 1 + RED_ACTION_EXFILTRATE * state.n + i

    if np.any(state.c >= 1):
        i = int(np.argmax(state.vuln + state.red_pressure))
        if state.c[i] == 0:
            return 1 + RED_ACTION_EXPLOIT * state.n + i
        if state.red_pressure[i] > 0.35:
            return 1 + RED_ACTION_PIVOT * state.n + i
        return 1 + RED_ACTION_PERSIST * state.n + i

    i = int(np.argmax(state.vuln))
    return 1 + RED_ACTION_SCAN * state.n + i


class BlueDefenseEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[CyberBattleConfig] = None, red_policy: Optional[RedPolicyFn] = None):
        super().__init__()
        self.cfg = config or CyberBattleConfig()
        self.state = CyberBattleState(self.cfg)
        self.red_policy = red_policy or (lambda _obs, s: heuristic_red_policy(s))

        self.action_space = Discrete(1 + 6 * self.cfg.n_nodes)
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.cfg.n_nodes * 14 + 1,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state.reset(seed=seed)
        return self.state._blue_obs(), {"red_obs": self.state._red_obs()}

    def step(self, action: int):
        red_obs = self.state._red_obs()
        red_action = int(self.red_policy(red_obs, self.state))
        blue_obs, _, blue_reward, _, terminated, truncated, info = self.state.step(int(action), red_action)
        info = dict(info)
        info["red_action"] = red_action
        return blue_obs, blue_reward, terminated, truncated, info


class RedAttackEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[CyberBattleConfig] = None, blue_policy: Optional[BluePolicyFn] = None):
        super().__init__()
        self.cfg = config or CyberBattleConfig()
        self.state = CyberBattleState(self.cfg)
        self.blue_policy = blue_policy or (lambda _obs, s: heuristic_blue_policy(s))

        self.action_space = Discrete(1 + 5 * self.cfg.n_nodes)
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.cfg.n_nodes * 10 + 1,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state.reset(seed=seed)
        return self.state._red_obs(), {"blue_obs": self.state._blue_obs()}

    def step(self, action: int):
        blue_obs = self.state._blue_obs()
        blue_action = int(self.blue_policy(blue_obs, self.state))
        _, red_obs, _, red_reward, terminated, truncated, info = self.state.step(blue_action, int(action))
        info = dict(info)
        info["blue_action"] = blue_action
        return red_obs, red_reward, terminated, truncated, info


class SB3PolicyAdapter:
    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, obs: np.ndarray, _state: CyberBattleState) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)


def make_blue_rule_based_env() -> BlueDefenseEnv:
    return BlueDefenseEnv(red_policy=lambda _obs, s: heuristic_red_policy(s))


def make_red_rule_based_env() -> RedAttackEnv:
    return RedAttackEnv(blue_policy=lambda _obs, s: heuristic_blue_policy(s))