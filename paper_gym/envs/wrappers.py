import numpy as np
import gym
from gym import spaces

from gym.envs.registration import load

class NormalizedActionWrapper(gym.ActionWrapper):
    """Environment wrapper to normalize the action space to [-1, 1]. This
    wrapper is adapted from rllab's [1] wrapper `NormalizedEnv`
    https://github.com/rll/rllab/blob/b3a28992eca103cab3cb58363dd7a4bb07f250a0/rllab/envs/normalized_env.py
    [1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel,
        "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016
        (https://arxiv.org/abs/1604.06778)
    """
    def __init__(self, env):
        super(NormalizedActionWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
            shape=self.env.action_space.shape)

    def action(self, action):
        # Clip the action in [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        # Map the normalized action to original action space
        lb, ub = self.env.action_space.low, self.env.action_space.high
        action = lb + 0.5 * (action + 1.0) * (ub - lb)
        return action

    def reverse_action(self, action):
        # Map the original action to normalized action space
        lb, ub = self.env.action_space.low, self.env.action_space.high
        action = 2.0 * (action - lb) / (ub - lb) - 1.0
        # Clip the action in [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        return action



def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    # Normalization wrapper
    env = NormalizedActionWrapper(env)
    return env