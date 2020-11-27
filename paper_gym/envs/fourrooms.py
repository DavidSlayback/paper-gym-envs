#Environment File for Classic Fourrooms Grid World
# From Option Critic (jeanharb repo https://github.com/jeanharb/option_critic/blob/master/fourrooms/fourrooms.py)
import numpy as np
import gym
from gym import core, spaces
from random import uniform

#class Fourrooms(gym.Env):
class Fourrooms(gym.Env):
    def __init__(self, initstate_seed=1234, rand_action_prob=(1/3.), goal_reward=1., step_reward=0.):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""


        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        self.goal_reward = goal_reward
        self.step_reward = step_reward

        # Action Space: from any state the agent can perform one of the four actions; Up, Down, Left and Right
        self.action_space = spaces.Discrete(4)
        self.rand_action_prob = rand_action_prob

        # Observation Space
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

        self.rng = np.random.RandomState(initstate_seed)

        self.initstate_seed = initstate_seed

        self.tostate = {}

        self.occ_dict = dict(zip(range(self.observation_space.n),
                                 np.argwhere(self.occupancy.flatten() == 0).squeeze()))


        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1

        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)


    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return np.array(state)

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect.
        We consider a case in which rewards are zero on all state transitions
        except the goal state which has a reward of +1.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            self.currentcell = nextcell
            if self.rng.uniform() < self.rand_action_prob:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = self.goal_reward if done else self.step_reward
        return state, float(done), done, {}

        """
        if self.rng.uniform() < 1/3:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[self.rng.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]

        done = state == self.goal
        return np.array(state), float(done), done, {}
        """
