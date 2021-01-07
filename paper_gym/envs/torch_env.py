import gym
import math
import torch
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os.path as path
import time
from collections import deque

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(RecordEpisodeStatisticsTorch, self).__init__(env)
        self.envcount = env.envcount
        self.device = env.device
        self.t0 = np.full(self.envcount, time.time())  # TODO: use perf_counter when gym removes Python 2 support
        self.episode_returns = torch.zeros(self.envcount, device=self.device)
        self.episode_lengths = torch.zeros(self.envcount, device=self.device).long()
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super(RecordEpisodeStatisticsTorch, self).reset(**kwargs)
        self.episode_returns = torch.zeros(self.envcount, device=self.device)
        self.episode_lengths = torch.zeros(self.envcount, device=self.device).long()
        return observation

    def step(self, action):
        observations, rewards, dones, infos = super(RecordEpisodeStatisticsTorch, self).step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        if torch.any(dones):
            rs = self.episode_returns.data.cpu().numpy()
            ls = self.episode_lengths.data.cpu().numpy()
            ts = np.round(time.time() - self.t0, 6)
            for idx in np.where(dones.data.cpu().numpy())[0]:
                infos[idx]['episode'] = {'r': rs[idx],'l': ls[idx],'t': ts[idx]}
                self.return_queue.append(rs)
                self.length_queue.append(ls)
            self.episode_returns[dones] = 0.0
            self.episode_lengths[dones] = 0
        return observations, rewards, dones, infos


class Pendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, env_count=1, device="cpu", g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.env_count = env_count
        self.device = device
        """
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        """
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        # TODO seeding
        # self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @torch.no_grad()
    def step(self, u):
        th, thdot = self.state[:, 0], self.state[:, 1]  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = torch.clamp(u, -self.max_torque, self.max_torque).squeeze(1)
        self.last_u = u  # for rendering

        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + math.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)

        self.state = torch.stack((newth, newthdot), dim=1)
        return self._get_obs(), -costs, False, {}

    @torch.no_grad()
    def reset(self):
        theta = math.pi * (2. * torch.rand(self.env_count, device=self.device) - 0.5)
        thetadot = (2. * torch.rand(self.env_count, device=self.device) - 0.5)
        self.state = torch.stack((theta, thetadot), dim=1)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state[:, 0], self.state[:, 1]
        return torch.stack(([torch.cos(theta), torch.sin(theta), thetadot]), dim=1)
    @torch.no_grad()
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            fname = path.join(path.dirname(""), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + math.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, torch.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""


class CartPole(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, env_count=1, device="cpu"):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.env_count = env_count

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.done = torch.full([env_count], True, dtype=torch.bool, device=device)
        self.state = torch.zeros([self.env_count, 4], dtype=torch.float32, device=device)

        self.device = device

    def seed(self, seed=None):
        return [seed]

    @torch.no_grad()
    def step(self, action):

        # breakpoint()
        # All env must already have been reset.
        self.done[:] = False
        x, x_dot, theta, theta_dot = self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3]
        # breakpoint()
        force = self.force_mag * ((action * 2.) - 1.)

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = ((self.gravity * sintheta - costheta * temp)
                    / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3] = x, x_dot, theta, theta_dot

        self.done = (
                (x < -self.x_threshold)
                | (x > self.x_threshold)
                | (theta < -self.theta_threshold_radians)
                | (theta > self.theta_threshold_radians)
        )
        reward = ~self.done

        self.state = self.reset()
        return self.state, reward, self.done, {}

    @torch.no_grad()
    def reset(self):
        # breakpoint()
        self.state = torch.where(self.done.unsqueeze(1),
                                 (torch.rand(self.env_count, 4, device=self.device) - 0.5) / 10., self.state)
        # self.state = (torch.rand((self.env_count, 4)) -0.5) / 10.
        return self.state

    @torch.no_grad()
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
