from __future__ import annotations
from typing import Optional

import numpy as np

from src.environment.airs_env import (
    ACTION_NOOP,
    ACTION_BLOCK_IP,
    ACTION_ISOLATE_NODE,
    ACTION_QUARANTINE_FILE,
    ACTION_INCREASE_MONITORING,
    ACTION_PATCH_VULNERABILITY,
    ACTION_RATE_LIMIT,
)


def _encode(env, action_type: int, node: Optional[int]) -> int:
    if action_type == ACTION_NOOP or node is None:
        return 0

    n = env.n
    block_idx = {
        ACTION_BLOCK_IP: 0,
        ACTION_ISOLATE_NODE: 1,
        ACTION_QUARANTINE_FILE: 2,
        ACTION_INCREASE_MONITORING: 3,
        ACTION_PATCH_VULNERABILITY: 4,
        ACTION_RATE_LIMIT: 5,
    }[action_type]
    return 1 + block_idx * n + int(node)


def rule_based_action(env) -> int:
    """
    Simple baseline policy:
      - if any node stage >= 3 => ISOLATE highest-stage node
      - elif any node stage == 2 => QUARANTINE it
      - elif any node stage == 1 => BLOCK_IP it
      - else => increase monitoring on node with highest attacker intensity b
    """
    c = env.c
    b = env.b

    if np.any(c >= 3):
        i = int(np.argmax(c))
        return _encode(env, ACTION_ISOLATE_NODE, i)

    if np.any(c == 2):
        i = int(np.where(c == 2)[0][0])
        return _encode(env, ACTION_QUARANTINE_FILE, i)

    if np.any(c == 1):
        i = int(np.where(c == 1)[0][0])
        return _encode(env, ACTION_BLOCK_IP, i)

    i = int(np.argmax(b))
    return _encode(env, ACTION_INCREASE_MONITORING, i)
