import os
import yaml

from src.eval.evaluate import evaluate_one, build_leaderboard

from src.eval.evaluate import RL_ALGOS

METHODS = RL_ALGOS + ["RuleBased"]

if __name__ == "__main__":
    with open("configs/airs_train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    episodes = int(cfg.get("eval", {}).get("episodes", 200))

    os.makedirs("data/results", exist_ok=True)

    for method in METHODS:
        for seed in seeds:
            print(f"=== Evaluating {method} seed={seed} episodes={episodes} ===")
            evaluate_one(method=method, config_path="configs/airs_train.yaml", seed=seed, episodes=episodes, data_dir="data")

    lb = build_leaderboard(data_dir="data")
    print(f"Saved leaderboard to: {lb}")
