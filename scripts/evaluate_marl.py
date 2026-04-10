from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.environment import CyberBattleConfig
from src.eval.marl_eval import evaluate_blue, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Red-vs-Blue AIRS policies")
    parser.add_argument("--config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--blue-algo", type=str, default="", choices=["", "PPO", "DQN", "A2C"])
    parser.add_argument("--blue-model", type=str, default=None)
    parser.add_argument("--red-model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="data/marl_eval")
    parser.add_argument("--disable-guard", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    battle_cfg = CyberBattleConfig(**cfg.get("env", {}))
    blue_algo = args.blue_algo or None

    summary, df = evaluate_blue(
        blue_algo=blue_algo,
        blue_model_path=args.blue_model,
        red_model_path=args.red_model,
        config=battle_cfg,
        episodes=args.episodes,
        seed=args.seed,
        data_dir=args.data_dir,
        use_guard=not args.disable_guard,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    df_path = out_root / "episodes.csv"
    sum_path = out_root / "summary.json"
    df.to_csv(df_path, index=False)
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved episodes to {df_path}")
    print(f"Saved summary to {sum_path}")


if __name__ == "__main__":
    main()
