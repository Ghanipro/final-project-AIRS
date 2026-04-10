from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import A2C, DQN, PPO

from src.environment import (
    BlueDefenseEnv,
    CyberBattleConfig,
    RedAttackEnv,
    SB3PolicyAdapter,
    heuristic_blue_policy,
    heuristic_red_policy,
)

BLUE_ALGOS: Dict[str, Type] = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}


def _find_latest_self_play_blue_model(data_dir: str, blue_algo: Optional[str] = None) -> Optional[str]:
    base_root = Path(data_dir) / "marl"
    search_roots = []

    if blue_algo:
        algo_root = base_root / blue_algo
        search_roots.append(algo_root)
    search_roots.append(base_root)

    candidates = []
    for marl_root in search_roots:
        if not marl_root.exists():
            continue
        for p in marl_root.glob("round_*"):
            if not p.is_dir():
                continue
            model_path = p / "blue_model.zip"
            if model_path.exists():
                candidates.append(model_path)

    if not candidates:
        return None

    latest = max(candidates, key=lambda fp: fp.stat().st_mtime)
    return str(latest)


def _action_id(cfg: CyberBattleConfig, action_type: int, node: int) -> int:
    if action_type <= 0:
        return 0
    return int(1 + (action_type - 1) * cfg.n_nodes + node)


def guarded_blue_action(state, rl_action: int) -> int:
    """Safety filter for blue policy in high-risk states.

    Uses learned action in low-risk regimes, falls back to strong heuristic when
    critical assets or propagation risk is high.
    """
    critical = np.array(state.cfg.criticality) >= 0.9
    critical_stage2plus = bool(np.any((state.c >= 2) & critical))
    max_stage = int(np.max(state.c))
    max_pressure = float(np.max(state.red_pressure))
    max_intensity = float(np.max(state.b))
    mean_detection = float(np.mean(state.detection))

    high_risk = (
        critical_stage2plus
        or max_stage >= 3
        or max_pressure >= 0.35
        or max_intensity >= 0.45
        or mean_detection >= 0.50
    )

    # Reject no-op in unsafe regimes.
    if rl_action == 0 and (max_stage >= 1 or max_pressure >= 0.20):
        return int(heuristic_blue_policy(state))

    if high_risk:
        return int(heuristic_blue_policy(state))

    return int(rl_action)


def strategic_blue_action(state) -> int:
    """Risk-aware defense policy that is stronger than basic rule heuristics.

    Prioritizes protecting critical nodes, then forcing de-escalation/containment,
    while maintaining monitoring coverage for early detection.
    """
    cfg = state.cfg
    n = cfg.n_nodes

    c = np.asarray(state.c)
    vuln = np.asarray(state.vuln)
    pressure = np.asarray(state.red_pressure)
    intensity = np.asarray(state.b)
    detection = np.asarray(state.detection)
    criticality = np.asarray(cfg.criticality)
    iso = np.asarray(state.iso)
    quar = np.asarray(state.quar)
    mon = np.asarray(state.mon)
    patched = np.asarray(state.patched)
    patch_timer = np.asarray(state.patch_timer)
    rl = np.asarray(state.rl)

    risk = (
        0.95 * (c / 4.0)
        + 0.55 * vuln
        + 0.65 * pressure
        + 0.45 * intensity
        + 0.20 * (1.0 - detection)
    ) * (0.70 + 0.60 * criticality)

    critical = criticality >= 0.9
    stage3_critical = np.where((c >= 3) & critical)[0]
    if stage3_critical.size > 0:
        i = int(stage3_critical[np.argmax(risk[stage3_critical])])
        if iso[i] == 0:
            return _action_id(cfg, 2, i)  # isolate
        if quar[i] == 0:
            return _action_id(cfg, 3, i)  # quarantine
        if rl[i] == 0:
            return _action_id(cfg, 6, i)  # rate limit
        return _action_id(cfg, 4, i)      # monitor

    stage2plus = np.where(c >= 2)[0]
    if stage2plus.size > 0:
        i = int(stage2plus[np.argmax(risk[stage2plus])])
        if quar[i] == 0 and criticality[i] >= 0.85:
            return _action_id(cfg, 3, i)
        if mon[i] < 2:
            return _action_id(cfg, 4, i)
        if rl[i] == 0:
            return _action_id(cfg, 6, i)
        if iso[i] == 0 and criticality[i] >= 0.8:
            return _action_id(cfg, 2, i)

    stage1 = np.where(c == 1)[0]
    if stage1.size > 0:
        i = int(stage1[np.argmax(risk[stage1])])
        if mon[i] < 2 and (pressure[i] >= 0.25 or criticality[i] >= 0.90):
            return _action_id(cfg, 4, i)
        if patch_timer[i] == 0 and patched[i] == 0 and vuln[i] >= 0.50 and criticality[i] >= 0.80:
            return _action_id(cfg, 5, i)
        if intensity[i] >= 0.40:
            return _action_id(cfg, 1, i)

    # In clearly safe states, prefer no-op to avoid unnecessary defense cost.
    if int(np.max(c)) == 0 and float(np.max(pressure)) < 0.22 and float(np.max(intensity)) < 0.25:
        return 0

    # Proactive hardening over highest-risk nodes only.
    order = np.argsort(-risk)
    for idx in order[: max(2, n // 3)]:
        i = int(idx)
        if pressure[i] >= 0.35 and mon[i] < 2:
            return _action_id(cfg, 4, i)
        if pressure[i] >= 0.30 and patch_timer[i] == 0 and patched[i] == 0 and vuln[i] >= 0.62:
            return _action_id(cfg, 5, i)
        if pressure[i] >= 0.45 and rl[i] == 0:
            return _action_id(cfg, 6, i)

    return 0


def hybrid_blue_action(state, rl_action: int) -> int:
    """Fuse RL with strategic + heuristic safety controllers.

    Chooses the lowest-risk action from a small candidate set to avoid policy
    collapse while still allowing RL to contribute in safer regions.
    """
    n = state.cfg.n_nodes

    def decode(action: int) -> tuple[int, int]:
        if action <= 0:
            return 0, -1
        raw = action - 1
        return (raw // n) + 1, raw % n

    def predicted_risk(action: int) -> float:
        action_type, node = decode(int(action))
        c = np.asarray(state.c, dtype=np.float32)
        pressure = np.asarray(state.red_pressure, dtype=np.float32)
        intensity = np.asarray(state.b, dtype=np.float32)
        vuln = np.asarray(state.vuln, dtype=np.float32)
        criticality = np.asarray(state.cfg.criticality, dtype=np.float32)

        # Baseline threat estimate.
        threat = np.sum((0.90 * (c / 4.0) + 0.60 * pressure + 0.40 * intensity + 0.35 * vuln) * (0.65 + 0.70 * criticality))

        # Penalize expensive actions to prevent action spam.
        action_cost_penalty = {
            0: 0.0,
            1: 0.010,  # block ip
            2: 0.060,  # isolate
            3: 0.040,  # quarantine
            4: 0.015,  # monitoring
            5: 0.070,  # patch
            6: 0.030,  # rate limit
        }

        if action_type == 0 or node < 0:
            return float(threat + 0.06)

        node_weight = 1.0 + 0.8 * criticality[node]
        if action_type == 1:  # block ip
            threat -= node_weight * (0.10 + 0.20 * pressure[node] + 0.15 * intensity[node])
        elif action_type == 2:  # isolate
            threat -= node_weight * (0.20 + 0.35 * (c[node] / 4.0) + 0.15 * pressure[node])
            threat += 0.03
        elif action_type == 3:  # quarantine
            threat -= node_weight * (0.16 + 0.30 * (c[node] / 4.0) + 0.10 * intensity[node])
        elif action_type == 4:  # monitoring
            threat -= node_weight * (0.08 + 0.25 * pressure[node] + 0.10 * vuln[node])
        elif action_type == 5:  # patch
            threat -= node_weight * (0.10 + 0.28 * vuln[node] + 0.10 * (c[node] / 4.0))
            threat += 0.02
        elif action_type == 6:  # rate limit
            threat -= node_weight * (0.09 + 0.25 * pressure[node] + 0.18 * (c[node] >= 3))
            threat += 0.01

        threat += action_cost_penalty.get(action_type, 0.02)
        return float(threat)

    heur = int(heuristic_blue_policy(state))
    strat = int(strategic_blue_action(state))
    # In low-risk regions, no-op is often optimal due cost penalties and survival bonus.
    low_risk = int(np.max(state.c)) == 0 and float(np.max(state.red_pressure)) < 0.22 and float(np.max(state.b)) < 0.25
    candidates = [int(rl_action), heur, strat, 0]
    if low_risk:
        candidates = [0, int(rl_action), heur, strat]
    unique_candidates = list(dict.fromkeys([c for c in candidates if c >= 0]))

    best = min(unique_candidates, key=predicted_risk)
    return int(best)


class LegacyBlueObsWrapper(gym.ObservationWrapper):
    """Projects MARL blue observation (14 features/node + t) to legacy AIRS format (10 features/node + t).

    Legacy per-node layout:
      [stage, iso, mon, patched, quar, rl, pt, intensity, zone, critical]
    New per-node layout:
      [stage, vuln, intensity, red_pressure, stealth, iso, mon, patched, quar, rl, pt, detection, avail, critical]
    """

    def __init__(self, env: BlueDefenseEnv, cfg: CyberBattleConfig):
        super().__init__(env)
        self._n = cfg.n_nodes
        self._zones = np.array(cfg.zones, dtype=np.float32) / 4.0
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._n * 10 + 1,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        out = np.zeros((self._n * 10 + 1,), dtype=np.float32)

        for i in range(self._n):
            src = i * 14
            dst = i * 10
            stage = obs[src + 0]
            intensity = obs[src + 2]
            iso = obs[src + 5]
            mon = obs[src + 6]
            patched = obs[src + 7]
            quar = obs[src + 8]
            rl = obs[src + 9]
            pt = obs[src + 10]
            critical = obs[src + 13]
            zone = self._zones[i]

            out[dst:dst + 10] = np.array(
                [stage, iso, mon, patched, quar, rl, pt, intensity, zone, critical],
                dtype=np.float32,
            )

        out[-1] = obs[-1]
        return out


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(algo: str, model_path: str, env):
    return BLUE_ALGOS[algo].load(model_path, env=env)


def evaluate_blue(
    blue_algo: Optional[str],
    blue_model_path: Optional[str],
    red_model_path: Optional[str],
    config: CyberBattleConfig,
    episodes: int,
    seed: int,
    data_dir: str,
    use_guard: bool = True,
) -> Dict[str, Any]:
    selected_blue_model_path: Optional[str] = None

    if blue_algo is None:
        blue_env = BlueDefenseEnv(config, red_policy=None)
        blue_policy_fn = lambda _obs, state: heuristic_blue_policy(state)
    else:
        blue_env_base = BlueDefenseEnv(config, red_policy=None)
        candidates: List[str] = []
        if blue_model_path is not None:
            candidates.append(blue_model_path)
        else:
            latest_self_play = _find_latest_self_play_blue_model(data_dir, blue_algo=blue_algo)
            if latest_self_play:
                candidates.append(latest_self_play)
            candidates.append(os.path.join(data_dir, "models", blue_algo, f"seed_{seed}", "model.zip"))

        model_cls = BLUE_ALGOS[blue_algo]
        tmp_model = None
        load_errors: List[str] = []
        for candidate in candidates:
            if not os.path.exists(candidate):
                load_errors.append(f"missing:{candidate}")
                continue
            try:
                tmp_model = model_cls.load(candidate)
                selected_blue_model_path = candidate
                break
            except Exception as ex:
                load_errors.append(f"{candidate} -> {type(ex).__name__}: {ex}")

        if tmp_model is None or selected_blue_model_path is None:
            raise RuntimeError(
                "Could not load any blue model checkpoint. Tried: " + " | ".join(load_errors)
            )

        expected_dim = int(tmp_model.observation_space.shape[0])
        current_dim = int(blue_env_base.observation_space.shape[0])

        if expected_dim == current_dim:
            blue_env = blue_env_base
        elif expected_dim == config.n_nodes * 10 + 1:
            blue_env = LegacyBlueObsWrapper(blue_env_base, config)
        else:
            raise ValueError(
                f"Unsupported blue model observation dimension {expected_dim}. "
                f"Expected either {current_dim} (MARL) or {config.n_nodes * 10 + 1} (legacy)."
            )

        blue_model = load_model(blue_algo, selected_blue_model_path, blue_env)
        if use_guard:
            blue_policy_fn = lambda obs, state: hybrid_blue_action(state, int(blue_model.predict(obs, deterministic=True)[0]))
        else:
            blue_policy_fn = lambda obs, _state: int(blue_model.predict(obs, deterministic=True)[0])

    if red_model_path is None:
        red_policy_fn = lambda _obs, state: heuristic_red_policy(state)
    else:
        if not os.path.exists(red_model_path):
            raise FileNotFoundError(f"Red model not found: {red_model_path}")
        red_env = RedAttackEnv(config, blue_policy=None)
        red_model = PPO.load(red_model_path, env=red_env)
        red_policy_fn = SB3PolicyAdapter(red_model)

    env = BlueDefenseEnv(config, red_policy=red_policy_fn)

    rows: List[Dict[str, Any]] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed * 10_000 + ep)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0
        breached = 0
        contained = 0
        max_stage = 0
        time_to_breach = None
        time_to_containment = None
        sec_losses = []
        avail_losses = []
        costs = []

        while not (done or truncated):
            action = int(blue_policy_fn(obs, env.state))
            obs, reward, done, truncated, step_info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            breached = int(step_info.get("breach", False)) or breached
            contained = int(step_info.get("contained", False)) or contained
            max_stage = max(max_stage, int(step_info.get("max_stage", 0)))

            if step_info.get("breach", False) and time_to_breach is None:
                time_to_breach = ep_len
            if step_info.get("contained", False) and time_to_containment is None:
                time_to_containment = ep_len

            sec_losses.append(float(step_info.get("security_loss", 0.0)))
            avail_losses.append(float(step_info.get("availability_loss", 0.0)))
            costs.append(float(step_info.get("blue_cost", 0.0)))

        rows.append(
            {
                "episode": ep,
                "return": ep_ret,
                "ep_len": ep_len,
                "breach": breached,
                "contained": contained,
                "max_stage": max_stage,
                "time_to_breach": float(time_to_breach) if time_to_breach is not None else np.nan,
                "time_to_containment": float(time_to_containment) if time_to_containment is not None else np.nan,
                "avg_security_loss": float(np.mean(sec_losses)) if sec_losses else 0.0,
                "avg_availability_loss": float(np.mean(avail_losses)) if avail_losses else 0.0,
                "avg_blue_cost": float(np.mean(costs)) if costs else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "blue_algo": blue_algo or "RuleBasedBlue",
        "blue_model_path": selected_blue_model_path or "RuleBasedBlue",
        "red_model_path": red_model_path or "RuleBasedRed",
        "episodes": episodes,
        "breach_rate": float(df["breach"].mean()),
        "contain_rate": float(df["contained"].mean()),
        "avg_return": float(df["return"].mean()),
        "avg_ep_len": float(df["ep_len"].mean()),
        "avg_time_to_breach": float(df["time_to_breach"].mean(skipna=True)) if not df["time_to_breach"].dropna().empty else None,
        "avg_time_to_containment": float(df["time_to_containment"].mean(skipna=True)) if not df["time_to_containment"].dropna().empty else None,
        "avg_security_loss": float(df["avg_security_loss"].mean()),
        "avg_availability_loss": float(df["avg_availability_loss"].mean()),
        "avg_blue_cost": float(df["avg_blue_cost"].mean()),
    }
    return summary, df
