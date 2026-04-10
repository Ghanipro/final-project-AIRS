from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ALGORITHMS = ["PPO", "DQN", "A2C"]


def run_cmd(args: list[str]) -> None:
    print("[run]", " ".join(args))
    result = subprocess.run(args)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train self-play for all blue algorithms")
    parser.add_argument("--config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--blue-pretrain-timesteps", type=int, default=120000)
    parser.add_argument("--blue-timesteps", type=int, default=120000)
    parser.add_argument("--red-timesteps", type=int, default=80000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/marl")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for algo in ALGORITHMS:
        algo_out = out_root / algo
        cmd = [
            args.python,
            "scripts/train_self_play.py",
            "--config",
            args.config,
            "--algo",
            algo,
            "--seed",
            str(args.seed),
            "--rounds",
            str(args.rounds),
            "--blue-pretrain-timesteps",
            str(args.blue_pretrain_timesteps),
            "--blue-timesteps",
            str(args.blue_timesteps),
            "--red-timesteps",
            str(args.red_timesteps),
            "--eval-episodes",
            str(args.eval_episodes),
            "--out",
            str(algo_out),
        ]
        run_cmd(cmd)

    print(f"Completed self-play training for: {', '.join(ALGORITHMS)}")


if __name__ == "__main__":
    main()
