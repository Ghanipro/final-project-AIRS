from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from src.train.train_model import ALGOS, train_one


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum performance training for AIRS")
    parser.add_argument("--base-config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--scenario-config", type=str, default="configs/research_scenarios.yaml")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--phases", type=str, default="train_easy,train_base,train_hard")
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.base_config))
    scenario_cfg = load_yaml(Path(args.scenario_config))
    scenarios = scenario_cfg.get("scenarios", {})

    curriculum = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    missing = [name for name in curriculum if name not in scenarios]
    if missing:
        raise ValueError(f"Missing scenarios in {args.scenario_config}: {missing}")

    seeds = base_cfg.get("seeds", [0, 1, 2, 3, 4])
    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    out_root = Path(args.out)
    config_root = out_root / "performance" / "configs"
    model_root = out_root / "performance" / "models"
    config_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    history = []
    for algo in ALGOS.keys():
        for seed in seeds:
            warm_start = None
            for phase in curriculum:
                merged = deep_merge(base_cfg, scenarios[phase])
                merged["seeds"] = [seed]
                merged["train"]["curriculum_phase"] = phase
                cfg_path = config_root / f"{algo}_seed_{seed}_{phase}.yaml"
                save_yaml(cfg_path, merged)

                phase_out = model_root / phase
                model_path = train_one(
                    algo=algo,
                    config_path=str(cfg_path),
                    seed=seed,
                    out_dir=str(phase_out),
                    warm_start_model=warm_start,
                )
                warm_start = model_path

            final_target = out_root / "models" / algo / f"seed_{seed}" / "model.zip"
            final_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(warm_start, final_target)
            history.append({"algo": algo, "seed": seed, "final_model": str(final_target)})
            print(f"[curriculum] promoted {algo} seed={seed} -> {final_target}")

    history_path = out_root / "performance" / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved curriculum history to {history_path}")


if __name__ == "__main__":
    main()
