import gym
from gym import spaces

class GymnasiumMDP(gym.Env):
    def __init__(self):
        super(GymnasiumMDP, self).__init__()
        self.action_space = spaces.Discrete(2)  # Example actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0, False, {}