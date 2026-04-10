from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Type

import yaml
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from src.environment import CyberBattleConfig, BlueDefenseEnv, RedAttackEnv, SB3PolicyAdapter, heuristic_red_policy, heuristic_blue_policy
from src.eval.marl_eval import evaluate_blue


BLUE_ALGOS: Dict[str, Type] = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
}

BLUE_DEFAULT_KWARGS: Dict[str, Dict] = {
    "PPO": {
        "learning_rate": 2e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ent_coef": 0.005,
    },
    "DQN": {
        "learning_rate": 2e-4,
        "buffer_size": 100000,
        "learning_starts": 5000,
        "batch_size": 256,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.30,
        "exploration_final_eps": 0.05,
    },
    "A2C": {
        "learning_rate": 3e-4,
        "n_steps": 64,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ent_coef": 0.005,
    },
}

RED_PPO_KWARGS: Dict = {
    "learning_rate": 2e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
}


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_blue_env(cfg: CyberBattleConfig, red_policy=None) -> BlueDefenseEnv:
    env = BlueDefenseEnv(cfg, red_policy=red_policy)
    return Monitor(env)


def make_red_env(cfg: CyberBattleConfig, blue_policy=None) -> RedAttackEnv:
    env = RedAttackEnv(cfg, blue_policy=blue_policy)
    return Monitor(env)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def make_blue_snapshot_metrics(algo_name: str, model_path: Path, cfg: CyberBattleConfig, seed: int, episodes: int, data_dir: str) -> Dict:
    summary, _ = evaluate_blue(
        blue_algo=algo_name,
        blue_model_path=str(model_path),
        red_model_path=None,
        config=cfg,
        episodes=episodes,
        seed=seed,
        data_dir=data_dir,
        use_guard=True,
    )
    return summary


def train_blue(algo_name: str, cfg: CyberBattleConfig, timesteps: int, seed: int, out_path: Path, red_model=None, warm_start: Optional[Path] = None):
    algo = BLUE_ALGOS[algo_name]
    env = make_blue_env(cfg, red_policy=SB3PolicyAdapter(red_model) if red_model is not None else None)

    if warm_start is not None and warm_start.exists():
        model = algo.load(str(warm_start), env=env)
        model.set_random_seed(seed)
    else:
        model = algo(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(out_path.parent / "tb"),
            **BLUE_DEFAULT_KWARGS.get(algo_name, {}),
        )

    model.learn(total_timesteps=timesteps)
    save_model(model, out_path)
    env.close()
    return out_path


def train_red(cfg: CyberBattleConfig, timesteps: int, seed: int, out_path: Path, blue_model=None, warm_start: Optional[Path] = None):
    env = make_red_env(cfg, blue_policy=SB3PolicyAdapter(blue_model) if blue_model is not None else None)

    if warm_start is not None and warm_start.exists():
        model = PPO.load(str(warm_start), env=env)
        model.set_random_seed(seed)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(out_path.parent / "tb"),
            **RED_PPO_KWARGS,
        )

    model.learn(total_timesteps=timesteps)
    save_model(model, out_path)
    env.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play training for AIRS Red-vs-Blue cyber battle")
    parser.add_argument("--config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--algo", type=str, default="PPO", choices=list(BLUE_ALGOS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blue-pretrain-timesteps", type=int, default=100000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--blue-timesteps", type=int, default=100000)
    parser.add_argument("--red-timesteps", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--out", type=str, default="data/marl")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.config))
    env_cfg = CyberBattleConfig(**base_cfg.get("env", {}))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    blue_model = None
    red_model = None
    history = []

    if args.blue_pretrain_timesteps > 0:
        pre_dir = out_root / "pretrain"
        pre_blue_path = pre_dir / "blue_model.zip"
        print("[pretrain] training blue against heuristic red")
        pre_blue_path = train_blue(
            args.algo,
            env_cfg,
            timesteps=args.blue_pretrain_timesteps,
            seed=args.seed,
            out_path=pre_blue_path,
            red_model=None,
            warm_start=None,
        )
        blue_model = BLUE_ALGOS[args.algo].load(str(pre_blue_path))

        pre_eval = make_blue_snapshot_metrics(
            args.algo,
            pre_blue_path,
            env_cfg,
            seed=args.seed + 77,
            episodes=args.eval_episodes,
            data_dir=args.data_dir,
        )
        history.append(
            {
                "round": "pretrain",
                "blue_path": str(pre_blue_path),
                "red_path": "RuleBasedRed",
                "blue_eval": pre_eval,
            }
        )

    for round_idx in range(args.rounds):
        round_dir = out_root / f"round_{round_idx}"
        blue_path = round_dir / "blue_model.zip"
        red_path = round_dir / "red_model.zip"

        if round_idx > 0:
            blue_warm = out_root / f"round_{round_idx - 1}" / "blue_model.zip"
        elif blue_model is not None:
            blue_warm = out_root / "pretrain" / "blue_model.zip"
        else:
            blue_warm = None
        red_warm = (out_root / f"round_{round_idx - 1}" / "red_model.zip") if round_idx > 0 else None

        print(f"[round {round_idx}] training blue against red snapshot")
        blue_path = train_blue(
            args.algo,
            env_cfg,
            timesteps=args.blue_timesteps,
            seed=args.seed + round_idx,
            out_path=blue_path,
            red_model=red_model,
            warm_start=blue_warm,
        )
        blue_model = BLUE_ALGOS[args.algo].load(str(blue_path))

        blue_eval = make_blue_snapshot_metrics(
            args.algo,
            blue_path,
            env_cfg,
            seed=args.seed + 100 * round_idx,
            episodes=args.eval_episodes,
            data_dir=args.data_dir,
        )

        print(f"[round {round_idx}] training red against blue snapshot")
        red_path = train_red(
            env_cfg,
            timesteps=args.red_timesteps,
            seed=args.seed + 10_000 + round_idx,
            out_path=red_path,
            blue_model=blue_model,
            warm_start=red_warm,
        )
        red_model = PPO.load(str(red_path))

        history.append(
            {
                "round": round_idx,
                "blue_path": str(blue_path),
                "red_path": str(red_path),
                "blue_eval": blue_eval,
            }
        )

    with (out_root / "self_play_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved self-play history to {out_root / 'self_play_history.json'}")
    print(f"Blue evaluation uses {args.eval_episodes} episodes per round")


if __name__ == "__main__":
    main()
