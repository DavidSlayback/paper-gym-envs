import gym

class TerminationCriticMDP(gym.Env):
    """
    3-state MDP, no actions

    [0,1,2]: green, red, blue

    :param p is probability of attraction to green
    """
    def __init__(self, p: float = 0.5):
        pass

    def step(self, action):
        pass
