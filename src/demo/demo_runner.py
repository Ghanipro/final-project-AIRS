from typing import Optional, Dict, Any
import numpy as np

from stable_baselines3 import PPO, DQN, A2C

from src.environment.airs_env import AIRSEnv, AIRSConfig
from src.baselines.rule_based import rule_based_action
from src.baselines.safe_policy import guarded_action

ALGOS = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}

ALGOS["RuleBased"] = None


def load_model(algo: str, model_path: str, env: AIRSEnv):
    if algo == "RuleBased":
        return None
    model_cls = ALGOS[algo]
    return model_cls.load(model_path, env=env)


def step_demo(env: AIRSEnv, obs: np.ndarray, algo: str, model=None, manual_action: Optional[int] = None):
    if manual_action is not None:
        action = manual_action
    elif algo == "RuleBased":
        action = rule_based_action(env)
    else:
        rl_action, _ = model.predict(obs, deterministic=True)
        action = guarded_action(env, int(rl_action))

    obs, reward, terminated, truncated, info = env.step(int(action))
    return obs, reward, terminated, truncated, info, int(action)