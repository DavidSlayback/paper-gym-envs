#from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
#from pybulletgym.envs.mujoco.envs.env_bases import BaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from gym.spaces import Box
import numpy as np


import os
def get_asset_xml(xml_name):
    return os.path.join(os.path.join(os.path.dirname(__file__), 'env_assets'), xml_name)

class PointMazeRobot(MJCFBasedRobot):
    def __init__(self, model_xml, robot_name='particle', action_dim=2, obs_dim=2, self_collision=True, power=.01):
        super().__init__(model_xml, robot_name, action_dim, obs_dim, self_collision)
        self.power = power

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.particle = self.parts['particle']
        self.jx = self.jdict['ball_x']
        self.jy = self.jdict['ball_y']
        self.jx.reset_current_position(self.np_random.uniform(low=-0.01, high=0.01), self.np_random.uniform(low=-0.01, high=0.01))
        self.jx.set_motor_torque(0)
        self.jy.reset_current_position(self.np_random.uniform(low=-0.01, high=0.01), self.np_random.uniform(low=-0.01, high=0.01))
        self.jy.set_motor_torque(0)

    # x, y force
    def apply_action(self, a):
        assert (np.isfinite(a).all())
        a = np.clip(a, -1, 1, dtype=float)  # Clip action
        self.jx.set_motor_torque(a[0] * self.power * self.jx.power_coef)
        self.jy.set_motor_torque(a[1] * self.power * self.jy.power_coef)

    def calc_state(self):
        return self.particle.current_position()[0:2]  # x,y position

class TwoDTMazeEnv(MJCFBaseBulletEnv):
    def __init__(self, model_xml='twod_tmaze_1target.xml', render=False, distance_threshold=0.1):
        robot = PointMazeRobot(get_asset_xml(model_xml))
        super().__init__(robot, render)
        self.distance_threshold = distance_threshold
        assert isinstance(self.observation_space, Box)
        assert self.observation_space.shape == (2,)

    def create_single_player_scene(self, bullet_client):
        return SinglePlayerStadiumScene(bullet_client, gravity=1., timestep=0.01, frame_skip=2)  # Frameskip shown was 2, no gravity, .01 timestep

    # Additionally save goal position
    def reset(self):
        r = super().reset()
        self.goal = self.robot.parts['target'].get_position()[0:2]
        return r

    # Change goal location
    def set_goal(self, new_goal):
        self.robot.parts['target'].reset_position(new_goal)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        # potential = self.robot.calc_potential()  # Baseline is 0
        # power_cost = -0.1 * np.square(a).sum()  # No power cost
        state = self.robot.calc_state()
        dist = np.linalg.norm(state - self.goal)
        reward = 0. if dist > self.distance_threshold else 1.
        return state, reward, bool(reward), {}

class MultiGoalTwoDTMazeEnv(MJCFBaseBulletEnv):
    def __init__(self, model_xml='twod_tmaze_2target.xml', render=False, distance_threshold=0.1):
        robot = PointMazeRobot(get_asset_xml(model_xml))
        super().__init__(robot, render)
        self.distance_threshold = distance_threshold
        self.target_count = np.array([0,0])
        assert isinstance(self.observation_space, Box)
        assert self.observation_space.shape == (2,)

    def create_single_player_scene(self, bullet_client):
        return SinglePlayerStadiumScene(bullet_client, gravity=1., timestep=0.01, frame_skip=2)  # Frameskip shown was 2, no gravity, .01 timestep

    # Additionally save goal position(s)
    def reset(self):
        r = super().reset()
        if not hasattr(self, 'goals'):
            self.goals = []
            self.goal_names = []
            for k, v in self.robot.parts.items():
                if k.startswith('target'):
                    self.goals.append(v.get_position()[0:2])
                    self.goal_names.append(k)
        # self.goal = self.robot.parts['target'].get_position()[0:2]
        return r

    # Change goal location for target
    def set_goal(self, new_goal):
        self.robot.parts['target'].reset_position(new_goal)

    # Remove goal. Tiebreak with first goal
    def remove_most_frequent_goal(self):
        most_frequent = np.argmin(self.target_count)
        self.goals.pop(most_frequent)
        self.goal_names.pop(most_frequent)
        return most_frequent

    def transfer(self, goal=0.):
        return self.remove_most_frequent_goal()

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        # potential = self.robot.calc_potential()  # Baseline is 0
        # power_cost = -0.1 * np.square(a).sum()  # No power cost
        state = self.robot.calc_state()
        dists = np.linalg.norm(state - np.array(self.goals), axis=1)
        reached_goal = np.argwhere(dists <= self.distance_threshold).flatten()
        if reached_goal.size == 0:
            reward = 0.
        else:
            reward = 1.
            self.target_count[reached_goal] += 1  # Add 1 to reached goal
        return state, reward, bool(reward), {}


if __name__ == "__main__":
    import time
    #xml = get_asset_xml('twod_tmaze_1target.xml')
    #robot = PointMazeRobot(xml)
    # env = MJCFBaseBulletEnv(robot)
    # e1 = TwoDTMazeEnv(render=True)
    e2 = MultiGoalTwoDTMazeEnv(render=True)
    e2.reset()
    # e1.reset(); # e2.reset();
    for i in range(1000):
        time.sleep(0.01)
        a = e2.action_space.sample()
        s, r1, d, _ = e2.step(a)
        if d: print("done")
        # _, r2, d, _ = e2.step(a)
        # print("{}_{}".format(r1, r2))
