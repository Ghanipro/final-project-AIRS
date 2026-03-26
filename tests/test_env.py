from src.environment.airs_env import AIRSEnv, AIRSConfig

def test_env_runs():
    env = AIRSEnv(AIRSConfig(n_nodes=5, max_steps=25))
    obs, info = env.reset(seed=123)
    assert obs.shape == (env.obs_dim,)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
