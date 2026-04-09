from typing import Optional, Dict, Any
import numpy as np

from stable_baselines3 import PPO, DQN, A2C
try:
    from sb3_contrib import RecurrentPPO  # type: ignore
except Exception:
    RecurrentPPO = None

try:
    from sb3_contrib import QRDQN  # type: ignore
except Exception:
    QRDQN = None

from src.environment.airs_env import AIRSEnv, AIRSConfig
from src.baselines.rule_based import rule_based_action
from src.baselines.safe_policy import guarded_action

ALGOS = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}

if RecurrentPPO is not None:
    ALGOS["RecurrentPPO"] = RecurrentPPO
if QRDQN is not None:
    ALGOS["QRDQN"] = QRDQN

ALGOS["RuleBased"] = None


def load_model(algo: str, model_path: str, env: AIRSEnv):
    if algo == "RuleBased":
        return None
    model_cls = ALGOS[algo]
    return model_cls.load(model_path, env=env)


def step_demo(env: AIRSEnv, obs: np.ndarray, algo: str, model=None, manual_action: Optional[int] = None):
    use_safe_guard = algo.endswith("_Safe")
    base_algo = algo.replace("_Safe", "")

    if manual_action is not None:
        action = manual_action
    elif base_algo == "RuleBased":
        action = rule_based_action(env)
    else:
        rl_action, _ = model.predict(obs, deterministic=True)
        action = guarded_action(env, int(rl_action)) if use_safe_guard else int(rl_action)

    obs, reward, terminated, truncated, info = env.step(int(action))
    return obs, reward, terminated, truncated, info, int(action)