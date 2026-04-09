from __future__ import annotations

from typing import Optional

import numpy as np

from src.baselines.rule_based import rule_based_action


def guarded_action(env, rl_action: Optional[int]) -> int:
    """
    Safety guardrail for demo/evaluation:
    - If a critical node is at stage >= 2, use the rule-based responder.
    - If aggregate risk is high, use the rule-based responder.
    - Otherwise, allow the RL action.
    """
    critical = np.array(env.cfg.criticality) >= 0.9
    critical_stage2plus = np.any((env.c >= 2) & critical)
    avg_stage = float(np.mean(env.c))
    max_intensity = float(np.max(env.b))

    high_risk = critical_stage2plus or avg_stage >= 1.2 or max_intensity >= 0.45

    if high_risk or rl_action is None:
        return int(rule_based_action(env))
    return int(rl_action)
