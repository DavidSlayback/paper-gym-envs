#Environment File for Classic Fourrooms Grid World
import numpy as np
import gym
from gym import core, spaces
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
    def __init__(self, seed=1234, rand_action_prob=(1/3.), rand_goal=False, goal_reward=1., step_reward=0.):
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
        directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

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
                    new_space = directions[a] + np.array([i,j])
                    if new_space.max() >= self.x or new_space.min() < 0 or self.state_grid[tuple(new_space)] == -1: self.transition_function[self.state_grid[i,j], a] = self.state_grid[i,j]  # Wall/edge, no movement
                    else: self.transition_function[self.state_grid[i,j], a] = self.state_grid[tuple(new_space)]  # Move to new square

        # Hallways (if considering blocking)
        self.hallways = [25, 51, 62, 88]
        self.hallways.sort()
        self.possible_goals = list(
            set(range(self.observation_space.n)) - set(self.hallways))  # Remove hallways from list of possible goals
        self.blocked_hallways = []
        self.blocked_hallways_T_idx = []
        self.blocked_hallways_S_idx = []
        self.seed(seed)
        self.set_goal(62) if not rand_goal else self.set_goal(self.rng.choice(self.observation_space.n))
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
        return goal

    # Block selected (or random) hallway
    def block_hallway(self, hallway=None):
        # Select hallway to block
        hallway = self.rng.choice(self.hallways) if hallway is None else self.hallways[hallway]
        # Ensure goal isn't part of list of hallways
        if hallway == self.goal: hallway = self.rng.choice(list(set(self.hallways) - {self.goal}))
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
        return hallway

    # Unblock selected hallway (index of first blocked hallway)
    def unblock_hallway(self, hallwayIndex=0):
        if not self.blocked_hallways: return None
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
        return hallway

    def step(self, action):
        if self.rng.uniform() < self.rand_action_prob:  #
            action = self.rng.choice(np.setdiff1d(np.arange(self.action_space.n), action, True))
        self.state = self.transition_function[self.state, action]  # Transition
        done = self.state == self.goal
        reward = self.goal_reward if done else self.step_reward
        return self.state, reward, done, {}

import torch
class FourRoomsTorch(gym.Env):
    def __init__(self, seed=1234, rand_action_prob=(1/3.), rand_goal=False, goal_reward=1., step_reward=0., numenvs=1, device="cuda", max_episode_steps=2000):
        self.envcount = numenvs
        self.device = device
        self.goal_reward = goal_reward
        self.step_reward = step_reward
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
        self.base_grid = torch.tensor([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()]).long().to(device)  # Wall or no wall
        self.x, self.y = self.base_grid.size()
        self.action_space = spaces.Discrete(4)
        self.rand_action_prob = rand_action_prob
        self.observation_space = spaces.Discrete(torch.sum(self.base_grid == 0))
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # Identify which squares are which discrete states
        tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.base_grid[i, j] == 0:
                    tostate[(i, j)] = statenum
                    statenum += 1
        self.state_grid = torch.full_like(self.base_grid, -1)
        for k, v in tostate.items(): self.state_grid[k] = v
        tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.base_grid[i, j] == 0:
                    tostate[(i, j)] = statenum
                    statenum += 1
        self.state_grid = torch.full_like(self.base_grid, -1)
        for k, v in tostate.items(): self.state_grid[k] = v

        # Define transition matrix
        self.transition_function = torch.zeros((self.observation_space.n, self.action_space.n)).long().to(device)  # Which state each action transitions to
        for i in range(self.x):
            for j in range(self.y):
                if self.state_grid[i, j] == -1: continue  # No transitions for wall spaces
                for a in range(self.action_space.n):
                    new_space = directions[a] + torch.tensor([i, j])
                    if new_space.max() >= self.x or new_space.min() < 0 or self.state_grid[tuple(new_space)] == -1:
                        self.transition_function[self.state_grid[i, j], a] = self.state_grid[i, j]  # Wall/edge, no movement
                    else:
                        self.transition_function[self.state_grid[i, j], a] = self.state_grid[tuple(new_space)]  # Move to new square

        # Hallways (if considering blocking)
        self.hallways = [25, 51, 62, 88]
        self.hallways.sort()
        self.blocked_hallways = []
        self.blocked_hallways_T_idx = []
        self.blocked_hallways_S_idx = []
        self.max_episode_steps = max_episode_steps
        self.step_count = torch.zeros(numenvs).long().to(device)
        self.seed(seed)
        self.set_goal(62) if not rand_goal else self.set_goal(self.rng_np.choice(self.observation_space.n))
        self.reset()

    # Reset all environments
    def reset(self):
        self.states = torch.from_numpy(self.rng_np.choice(self.init_states, size=self.envcount)).long().to(self.device)  # Reset states
        return self.states

    def seed(self, seed=None):
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.rng_np = np.random.RandomState(seed)

    # Change the goal of all environments to a single different goal
    def set_goal(self, goal):
        self.goal = goal  # Change goal
        self.init_states = list(range(self.observation_space.n))  # Change possible initial states
        self.init_states.remove(self.goal)
        return goal

    # Block selected (or random) hallway
    def block_hallway(self, hallway=None):
        # Select hallway to block
        hallway = self.rng_np.choice(self.hallways) if hallway is None else self.hallways[hallway]
        # Ensure goal isn't part of list of hallways
        if hallway == self.goal: hallway = self.rng_np.choice(list(set(self.hallways) - {self.goal}))
        # Remove from initial states
        self.init_states.remove(hallway)
        # Remove from state grid
        i, j = torch.nonzero(self.state_grid == hallway, as_tuple=True)
        self.state_grid[i,j] = -1
        self.blocked_hallways_S_idx.append((i,j))
        # Remove from transition function
        i, j = torch.nonzero(self.transition_function == hallway, as_tuple=True)
        self.transition_function[i,j] = i  # Moves redirect to same square
        # Add to list of blocked hallways
        self.blocked_hallways.append(hallway)
        self.blocked_hallways_T_idx.append((i,j))
        self.hallways.remove(hallway)
        return hallway

    # Unblock selected hallway (index of first blocked hallway)
    def unblock_hallway(self, hallwayIndex=0):
        if not self.blocked_hallways: return None
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
        return hallway

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect.
        We consider a case in which rewards are zero on all state transitions
        except the goal state which has a reward of +1.
        """
        rand_action = torch.rand(action.size(), generator=self.rng, device=self.device) < self.rand_action_prob  # Check which ones get random
        action[rand_action] = torch.randint(0, 4, action[rand_action].size(), generator=self.rng, device=self.device)  # Assign random
        self.states = self.transition_function[self.states, action]  # Transition
        self.step_count = self.step_count + 1  # Increment step count
        done = self.states == self.goal  # Which envs reached goal?
        reward = done.float() * self.goal_reward + (~done).float() * self.step_reward  # Reward for goal or for step
        done = torch.logical_or(done, self.step_count > self.max_episode_steps)  # Envs that timed out as well (no reward for them)
        self.states[done] = torch.from_numpy(self.rng_np.choice(self.init_states, size=done.sum().cpu().item())).to(self.device)  # Reset done environments
        self.step_count[done] = 0  # Reset step counts for done environments
        return self.states, reward, done, {}

from collections import deque
import time
class FourRoomsEpisodeRecorderTorch(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = np.full(env.envcount, time.time())  # TODO: use perf_counter when gym removes Python 2 support
        self.episode_return = torch.zeros(env.envcount).to(env.device)
        self.episode_length = torch.zeros(env.envcount).long().to(env.device)
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_return = torch.zeros(self.env.envcount).to(self.env.device)
        self.episode_length = torch.zeros(self.env.envcount).long().to(self.env.device)
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_return += reward
        self.episode_length += 1
        if torch.any(done):
            didx = torch.nonzero(done)  # Indices of done episodes
            r = self.episode_return[didx].cpu().numpy().flatten()
            l = self.episode_length[didx].cpu().numpy().flatten()
            t = np.round(time.time() - self.t0[didx.cpu().numpy()], 6)
            info['episode'] = {
                'r': r,
                'l': l,
                't': t
            }
            self.episode_length[didx] = 0
            self.episode_return[didx] = 0.0
            self.return_queue.append(i for i in r)
            self.length_queue.append(i for i in l)
        return observation, reward, done, info

if __name__ == "__main__":
    n = 8
    testTorchEnv = FourRoomsTorch(numenvs=n)
    for t in range(20000):
        s, r, d, info = testTorchEnv.step(torch.randint(4, (n,), device='cuda'))
        if torch.any(d):
            print(info)
            s = testTorchEnv.reset()

    testEnv = Fourrooms()
    for t in range(20000):
        s, r, d, _ = testEnv.step(np.random.randint(4))
        if d:
            print("D")
            print(testEnv.goal)
            testEnv.set_goal(np.random.choice(104))
            s = testEnv.reset()

