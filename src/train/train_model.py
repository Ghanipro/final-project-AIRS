from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Dict, Any

import yaml

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import RecurrentPPO, QRDQN

from src.environment.airs_env import AIRSEnv, AIRSConfig


ALGOS = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "RecurrentPPO": RecurrentPPO,
    "QRDQN": QRDQN,
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_cfg: Dict[str, Any], seed: int) -> AIRSEnv:
    cfg = AIRSConfig(**env_cfg)
    env = AIRSEnv(cfg)
    env.reset(seed=seed)
    return env


def train_one(algo: str, config_path: str, seed: int, out_dir: str = "data") -> str:
    cfg = load_config(config_path)
    env_cfg = cfg["env"]
    train_cfg = cfg["train"]

    env = make_env(env_cfg, seed=seed)

    model_cls = ALGOS[algo]
    policy = train_cfg.get("policy", "MlpPolicy")
    model_kwargs = dict(train_cfg.get("model_kwargs", {}))

    model = model_cls(policy, env, verbose=1, seed=seed, **model_kwargs)

    timesteps = int(train_cfg.get("timesteps", 200_000))
    model.learn(total_timesteps=timesteps)

    save_root = os.path.join(out_dir, "models", algo, f"seed_{seed}")
    os.makedirs(save_root, exist_ok=True)

    model_path = os.path.join(save_root, "model.zip")
    model.save(model_path)

    meta = {
        "algo": algo,
        "seed": seed,
        "timesteps": timesteps,
        "env_config": env_cfg,
        "train_config": train_cfg,
        "airs_config_dataclass": asdict(AIRSConfig(**env_cfg)),
    }
    with open(os.path.join(save_root, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=list(ALGOS.keys()))
    parser.add_argument("--config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    path = train_one(args.algo, args.config, seed=args.seed, out_dir=args.out)
    print(f"Saved model to: {path}")
