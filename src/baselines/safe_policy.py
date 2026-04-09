from __future__ import annotations

from typing import Optional

import numpy as np

from src.environment.airs_env import (
    ACTION_BLOCK_IP,
    ACTION_INCREASE_MONITORING,
    ACTION_ISOLATE_NODE,
    ACTION_PATCH_VULNERABILITY,
    ACTION_QUARANTINE_FILE,
    ACTION_RATE_LIMIT,
)


def _encode(env, action_type: int, node: int) -> int:
    block_idx = {
        ACTION_BLOCK_IP: 0,
        ACTION_ISOLATE_NODE: 1,
        ACTION_QUARANTINE_FILE: 2,
        ACTION_INCREASE_MONITORING: 3,
        ACTION_PATCH_VULNERABILITY: 4,
        ACTION_RATE_LIMIT: 5,
    }[action_type]
    return 1 + block_idx * env.n + int(node)


def _risk_scores(env) -> np.ndarray:
    stage_norm = env.c.astype(np.float32) / 4.0
    critical = np.array(env.cfg.criticality, dtype=np.float32)
    intensity = env.b.astype(np.float32)
    return 1.6 * stage_norm * critical + 0.8 * intensity


def shield_action(env) -> int:
    """Deterministic emergency controller used only when risk is elevated."""
    scores = _risk_scores(env)
    target = int(np.argmax(scores))

    c = env.c
    b = env.b
    critical = np.array(env.cfg.criticality) >= 0.9

    # Hard emergency: isolate critical assets early.
    crit_idx = np.where((c >= 2) & critical)[0]
    if len(crit_idx) > 0:
        i = int(crit_idx[np.argmax(scores[crit_idx])])
        return _encode(env, ACTION_ISOLATE_NODE, i)

    # Stop lateral movement in late stages.
    stage3_idx = np.where(c >= 3)[0]
    if len(stage3_idx) > 0:
        i = int(stage3_idx[np.argmax(scores[stage3_idx])])
        if env.rl[i] == 0:
            return _encode(env, ACTION_RATE_LIMIT, i)
        return _encode(env, ACTION_ISOLATE_NODE, i)

    # Mid-stage compromise: quarantine first, then patch.
    stage2_idx = np.where(c == 2)[0]
    if len(stage2_idx) > 0:
        i = int(stage2_idx[np.argmax(scores[stage2_idx])])
        if env.quar[i] == 0:
            return _encode(env, ACTION_QUARANTINE_FILE, i)
        return _encode(env, ACTION_PATCH_VULNERABILITY, i)

    # Early compromise with high intensity: block + monitor.
    stage1_idx = np.where(c == 1)[0]
    if len(stage1_idx) > 0:
        i = int(stage1_idx[np.argmax(scores[stage1_idx])])
        if b[i] >= 0.35:
            return _encode(env, ACTION_BLOCK_IP, i)
        return _encode(env, ACTION_INCREASE_MONITORING, i)

    return _encode(env, ACTION_INCREASE_MONITORING, target)


def guarded_action(env, rl_action: Optional[int]) -> int:
    """
    Adaptive safety shield for RL inference/evaluation.
    Uses RL in low-risk regimes and a stronger deterministic shield in high-risk regimes.
    """
    critical = np.array(env.cfg.criticality) >= 0.9
    critical_stage1plus = np.any((env.c >= 1) & critical)
    avg_stage = float(np.mean(env.c))
    max_intensity = float(np.max(env.b))
    max_stage = int(np.max(env.c))

    high_risk = (
        critical_stage1plus
        or max_stage >= 3
        or avg_stage >= 0.9
        or max_intensity >= 0.35
    )

    if rl_action is None:
        return int(shield_action(env))

    # Reject passivity when the environment is heating up.
    action_type, _ = env._decode_action(int(rl_action))
    if action_type == 0 and (avg_stage >= 0.5 or max_intensity >= 0.28):
        return int(shield_action(env))

    if high_risk:
        return int(shield_action(env))
    return int(rl_action)
