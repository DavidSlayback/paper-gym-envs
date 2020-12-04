#Environment File for Classic Fourrooms Grid World
import numpy as np
import gym
from gym import core, spaces

class FourroomsMonitor(gym.Wrapper):
    pass
"""
Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning

As a simple illustration of planning with options, consider the rooms example, a
gridworld environment of four rooms as shown in Fig. 2. The cells of the grid correspond to
the states of the environment. From any state the agent can perform one of four actions, up,
down, left or right, which have a stochastic effect. With probability 2/3, the actions
cause the agent to move one cell in the corresponding direction, and with probability 1/3,
the agent moves instead in one of the other three directions, each with probability 1/9. In
either case, if the movement would take the agent into a wall then the agent remains in the
same cell. For now we consider a case in which rewards are zero on all state transitions.
"""
class Fourrooms(gym.Env):
    def __init__(self, seed=1234, rand_action_prob=(1/3.), goal_reward=1., step_reward=0.):
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
        self.base_grid = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])  # Wall or no wall
        self.x, self.y = self.base_grid.shape
        # self.agent_grid = np.zeros_like(self.base_grid)  # Agent/no-agent
        self.goal_reward = goal_reward
        self.step_reward = step_reward

        self.action_space = spaces.Discrete(4)
        self.rand_action_prob = rand_action_prob
        self.observation_space = spaces.Discrete(np.sum(self.base_grid == 0))
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

        # Identify which squares are which discrete states
        tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.base_grid[i, j] == 0:
                    tostate[(i, j)] = statenum
                    statenum += 1
        self.state_grid = np.full_like(self.base_grid, -1)
        for k, v in tostate.items(): self.state_grid[k] = v

        # Define transition matrix
        self.transition_function = np.zeros((self.observation_space.n, self.action_space.n), dtype=int)  # Which state each action transitions to
        for i in range(self.x):
            for j in range(self.y):
                if self.state_grid[i,j] == -1: continue  # No transitions for wall spaces
                for a in range(self.action_space.n):
                    new_space = self.directions[a] + np.array([i,j])
                    if new_space.max() >= self.x or new_space.min() < 0 or self.state_grid[tuple(new_space)] == -1: self.transition_function[self.state_grid[i,j], a] = self.state_grid[i,j]  # Wall/edge, no movement
                    else: self.transition_function[self.state_grid[i,j], a] = self.state_grid[tuple(new_space)]  # Move to new square

        # Hallways (if considering blocking)
        self.hallways = [25, 51, 62, 88]
        self.hallways.sort()
        self.blocked_hallways = []
        self.blocked_hallways_T_idx = []
        self.blocked_hallways_S_idx = []
        self.seed(seed)
        self.set_goal(62)
        self.reset()

    def reset(self):
        self.state = self.rng.choice(self.init_states)
        return self.state

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def set_goal(self, goal):
        self.goal = goal
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)

    # Block selected (or random) hallway
    def block_hallway(self, hallway=None):
        # Select hallway to block
        hallway = self.rng.choice(self.hallways) if hallway is None else self.hallways[hallway]
        # Remove from initial states
        self.init_states.remove(hallway)
        # Remove from state grid
        i, j = np.nonzero(self.state_grid == hallway)
        self.state_grid[i,j] = -1
        self.blocked_hallways_S_idx.append((i,j))
        # Remove from transition function
        i, j = np.nonzero(self.transition_function == hallway)
        self.transition_function[i,j] = i  # Moves redirect to same square
        # Add to list of blocked hallways
        self.blocked_hallways.append(hallway)
        self.blocked_hallways_T_idx.append((i,j))
        self.hallways.remove(hallway)

    # Unblock selected hallway (index of first blocked hallway)
    def unblock_hallway(self, hallwayIndex=0):
        if not self.blocked_hallways: return
        hallway = self.blocked_hallways[hallwayIndex]
        self.init_states.append(hallway)
        self.init_states.sort()
        i, j = self.blocked_hallways_S_idx[hallwayIndex]
        self.blocked_hallways_S_idx.pop(hallwayIndex)
        self.state_grid[i, j] = hallway
        i, j = self.blocked_hallways_T_idx[hallwayIndex]
        self.blocked_hallways_T_idx.remove((i,j))
        self.transition_function[i, j] = hallway
        self.blocked_hallways.pop(hallwayIndex)
        self.hallways.append(hallway)
        self.hallways.sort()

    def step(self, action):
        if self.rng.uniform() < self.rand_action_prob:  #
            action = self.rng.choice(np.setdiff1d(np.arange(self.action_space.n), action, True))
        self.state = self.transition_function[self.state, action]  # Transition
        done = self.state == self.goal
        reward = self.goal_reward if done else self.step_reward
        return self.state, reward, done, {}

# class FourroomsTorch(gym.Env):


if __name__ == "__main__":
    testEnv = Fourrooms()
    for t in range(20000):
        s, r, d, _ = testEnv.step(np.random.randint(4))
        if d:
            print("D")
            print(testEnv.goal)
            testEnv.set_goal(np.random.randint(104))
            s = testEnv.reset()

