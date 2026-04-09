from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import yaml

from stable_baselines3 import PPO, DQN, A2C

from src.environment.airs_env import AIRSEnv, AIRSConfig
from src.baselines.rule_based import rule_based_action
from src.baselines.safe_policy import guarded_action


ALGOS = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}

RL_ALGOS = list(ALGOS.keys())
ALL_METHODS = RL_ALGOS + ["RuleBased"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_cfg: Dict[str, Any], seed: int) -> AIRSEnv:
    cfg = AIRSConfig(**env_cfg)
    env = AIRSEnv(cfg)
    env.reset(seed=seed)
    return env


def _load_model(algo: str, model_path: str, env: AIRSEnv):
    model_cls = ALGOS[algo]
    # SB3 loads without needing env, but passing env keeps predict consistent if needed
    return model_cls.load(model_path, env=env)


def evaluate_one(
    method: str,
    config_path: str,
    seed: int,
    episodes: Optional[int] = None,
    data_dir: str = "data",
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    env_cfg = cfg["env"]
    eval_cfg = cfg.get("eval", {})
    n_episodes = int(episodes or eval_cfg.get("episodes", eval_cfg.get("n_episodes", 200)))

    env = make_env(env_cfg, seed=seed)

    model = None
    if method in RL_ALGOS:
        model_path = os.path.join(data_dir, "models", method, f"seed_{seed}", "model.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = _load_model(method, model_path, env)

    rows: List[Dict[str, Any]] = []
    action_counts = np.zeros(env.action_space.n, dtype=np.int64)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed * 10_000 + ep)
        done = False
        truncated = False

        ep_return = 0.0
        ep_len = 0
        breached = 0
        contained = 0
        max_stage = 0

        sec_losses = []
        avail_losses = []
        costs = []

        while not (done or truncated):
            if method == "RuleBased":
                action = rule_based_action(env)
            else:
                rl_action, _state = model.predict(obs, deterministic=True)
                action = guarded_action(env, int(rl_action))

            action = int(action)
            action_counts[action] += 1

            obs, reward, done, truncated, step_info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

            breached = int(step_info.get("breach", False)) or breached
            contained = int(step_info.get("contained", False)) or contained
            max_stage = max(max_stage, int(step_info.get("max_stage", 0)))

            sec_losses.append(float(step_info.get("security_loss", 0.0)))
            avail_losses.append(float(step_info.get("availability_loss", 0.0)))
            costs.append(float(step_info.get("action_cost", 0.0)))

        rows.append(
            {
                "method": method,
                "seed": seed,
                "episode": ep,
                "return": ep_return,
                "ep_len": ep_len,
                "breach": breached,
                "contained": contained,
                "max_stage": max_stage,
                "avg_security_loss": float(np.mean(sec_losses)) if sec_losses else 0.0,
                "avg_availability_loss": float(np.mean(avail_losses)) if avail_losses else 0.0,
                "sum_action_cost": float(np.sum(costs)) if costs else 0.0,
            }
        )

    df = pd.DataFrame(rows)

    out_root = os.path.join(data_dir, "results", method, f"seed_{seed}")
    os.makedirs(out_root, exist_ok=True)

    episodes_csv = os.path.join(out_root, "episodes.csv")
    df.to_csv(episodes_csv, index=False)

    summary = {
        "method": method,
        "seed": seed,
        "episodes": n_episodes,
        "breach_rate": float(df["breach"].mean()),
        "contain_rate": float(df["contained"].mean()),
        "avg_return": float(df["return"].mean()),
        "std_return": float(df["return"].std(ddof=1)) if len(df) > 1 else 0.0,
        "avg_ep_len": float(df["ep_len"].mean()),
        "avg_max_stage": float(df["max_stage"].mean()),
        "avg_security_loss": float(df["avg_security_loss"].mean()),
        "avg_availability_loss": float(df["avg_availability_loss"].mean()),
        "avg_action_cost": float(df["sum_action_cost"].mean()),
        "action_counts": action_counts.tolist(),
        "action_counts_total": int(action_counts.sum()),
        "env_config": env_cfg,
        "airs_config": asdict(AIRSConfig(**env_cfg)),
        "paths": {"episodes_csv": episodes_csv},
    }

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_leaderboard(data_dir: str = "data") -> str:
    results_root = os.path.join(data_dir, "results")
    rows = []
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"No results directory found at {results_root}")

    for method in os.listdir(results_root):
        mdir = os.path.join(results_root, method)
        if not os.path.isdir(mdir):
            continue
        for seed_dir in os.listdir(mdir):
            sdir = os.path.join(mdir, seed_dir)
            if not os.path.isdir(sdir):
                continue
            summary_path = os.path.join(sdir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    rows.append(json.load(f))

    df = pd.DataFrame(rows)
    leaderboard_path = os.path.join(results_root, "leaderboard.csv")
    df.to_csv(leaderboard_path, index=False)
    return leaderboard_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS)
    parser.add_argument("--config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--data", type=str, default="data")
    args = parser.parse_args()

    evaluate_one(args.method, args.config, seed=args.seed, episodes=args.episodes, data_dir=args.data)
    lb = build_leaderboard(data_dir=args.data)
    print(f"Leaderboard: {lb}")
