#from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
#from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from gym.spaces import Box
import numpy as np

class TMazePointMass(MJCFBasedRobot):
    def __init__(self):
        MJCFBasedRobot.__init__(self, 'twod_tmaze_1target.xml', 'particle', action_dim=2, obs_dim=2)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position(u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert(np.isfinite(a).all())
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0
        self.slider.set_motor_torque(100*float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        x, vx = self.slider.current_position()
        self.theta, theta_dot = self.j1.current_position()
        assert(np.isfinite(x))

        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        qpos = np.array([x, self.theta])  # shape (2,)
        qvel = np.array([vx, theta_dot])  # shape (2,)

        return np.array([
            qpos,   # self.sim.data.qpos
            qvel])  # self.sim.data.qvel

class TMazeEnvironment(MJCFBaseBulletEnv):
    def __init__(self, model_path, frame_skip, xbounds, ybounds):
        super().__init__()
        assert isinstance(self.observation_space, Box)
        assert self.observation_space.shape == (2,)
        pass