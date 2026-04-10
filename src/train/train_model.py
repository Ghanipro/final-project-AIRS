from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Dict, Any, Optional

import yaml

from stable_baselines3 import PPO, DQN, A2C

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from src.environment.airs_env import AIRSEnv, AIRSConfig


ALGOS = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_cfg: Dict[str, Any], seed: int) -> AIRSEnv:
    cfg = AIRSConfig(**env_cfg)
    env = AIRSEnv(cfg)
    env.reset(seed=seed)
    return env


DEFAULT_POLICY_BY_ALGO = {
    "PPO": "MlpPolicy",
    "A2C": "MlpPolicy",
    "DQN": "MlpPolicy",
}


def train_one(
    algo: str,
    config_path: str,
    seed: int,
    out_dir: str = "data",
    warm_start_model: Optional[str] = None,
) -> str:
    if algo not in ALGOS:
        available = ", ".join(sorted(ALGOS.keys()))
        raise ValueError(f"Algorithm '{algo}' is not available. Available: {available}")

    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})

    timesteps = int(train_cfg.get("timesteps", 200000))
    policy = train_cfg.get("policy_by_algo", {}).get(algo, DEFAULT_POLICY_BY_ALGO[algo])
    model_cls = ALGOS[algo]

    # Logging setup
    log_cfg = cfg.get("logging", {})
    log_root = log_cfg.get("log_root", "data/logs")
    tb_name = log_cfg.get("tb_name", "AIRS")
    os.makedirs(log_root, exist_ok=True)

    run_name = f"{algo}_seed_{seed}"
    run_dir = os.path.join(log_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Create env + monitor
    env = make_env(env_cfg, seed)
    env = Monitor(env, filename=os.path.join(run_dir, "monitor.csv"))

    # Build model kwargs (global + per‑algo)
    model_kwargs = dict(train_cfg.get("model_kwargs", {}))
    per_algo = train_cfg.get("model_kwargs_by_algo", {}).get(algo, {})
    model_kwargs.update(per_algo)

    if warm_start_model and os.path.exists(warm_start_model):
        model = model_cls.load(warm_start_model, env=env)
        model.set_random_seed(seed)
    else:
        model = model_cls(
            policy,
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=log_root,
            **model_kwargs
        )

    # Eval + Early stopping
    eval_cfg = cfg.get("eval", {})
    callback = None
    if eval_cfg.get("enabled", True):
        eval_env = make_env(env_cfg, seed + 999)
        eval_env = Monitor(eval_env, filename=os.path.join(run_dir, "eval_monitor.csv"))

        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=eval_cfg.get("early_stop_patience", 4),
            min_evals=1,
            verbose=1
        )

        callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=os.path.join(run_dir, "eval_logs"),
            eval_freq=eval_cfg.get("freq", 20000),
            n_eval_episodes=eval_cfg.get("n_episodes", 20),
            deterministic=True,
            callback_after_eval=stop_callback,
            verbose=1
        )

    model.learn(total_timesteps=timesteps, callback=callback, tb_log_name=tb_name)

    save_root = os.path.join(out_dir, "models", algo, f"seed_{seed}")
    os.makedirs(save_root, exist_ok=True)

    model_path = os.path.join(save_root, "model.zip")
    model.save(model_path)

    meta = {
        "algo": algo,
        "seed": seed,
        "timesteps": timesteps,
        "warm_start_model": warm_start_model,
        "env_config": env_cfg,
        "train_config": train_cfg,
        "airs_config_dataclass": asdict(AIRSConfig(**env_cfg)),
    }
    with open(os.path.join(save_root, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    env.close()
    if callback is not None and hasattr(callback, "eval_env") and callback.eval_env is not None:
        callback.eval_env.close()

    return model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=list(ALGOS.keys()))
    parser.add_argument("--config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--warm-start", type=str, default=None)
    args = parser.parse_args()

    path = train_one(args.algo, args.config, seed=args.seed, out_dir=args.out, warm_start_model=args.warm_start)
    print(f"Saved model to: {path}")