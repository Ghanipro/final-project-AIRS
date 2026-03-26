import os
import yaml

from src.train.train_model import train_one

ALGOS = ["PPO", "DQN", "A2C", "RecurrentPPO", "QRDQN"]

if __name__ == "__main__":
    with open("configs/airs_train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    os.makedirs("data/models", exist_ok=True)

    for algo in ALGOS:
        for seed in seeds:
            print(f"=== Training {algo} seed={seed} ===")
            train_one(algo=algo, config_path="configs/airs_train.yaml", seed=seed, out_dir="data")
