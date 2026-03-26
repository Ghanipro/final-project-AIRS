# ... keep imports and ALGOS ...

DEFAULT_POLICY_BY_ALGO = {
    "PPO": "MlpPolicy",
    "A2C": "MlpPolicy",
    "DQN": "MlpPolicy",
    "QRDQN": "MlpPolicy",
    "RecurrentPPO": "MlpLstmPolicy",
}

def train_one(algo: str, config_path: str, seed: int, out_dir: str = "data") -> str:
    cfg = load_config(config_path)
    env_cfg = cfg["env"]
    train_cfg = cfg["train"]

    env = make_env(env_cfg, seed=seed)

    model_cls = ALGOS[algo]

    # Use per-algo default, allow override via config
    policy = train_cfg.get("policy_by_algo", {}).get(algo) or DEFAULT_POLICY_BY_ALGO[algo]

    model_kwargs = dict(train_cfg.get("model_kwargs", {}))
    model = model_cls(policy, env, verbose=1, seed=seed, **model_kwargs)
    # ...
