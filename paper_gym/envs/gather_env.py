import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from ctypes import byref
import numpy as np
import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase
import pybullet_data
pth = osp.join(pybullet_data.getDataPath(), "mjcf", 'ant.xml')  # Path to base ant file

class ModifiedAnt(WalkerBase):
    """
    PyBullet Ant with a different xml file
    """
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, modified_xml):
        WalkerBase.__init__(self, modified_xml, "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

APPLE = 0
BOMB = 1

class GatherEnv(WalkerBaseBulletEnv):
    def __init__(self, robot_class=ModifiedAnt, render=False,
                 n_apples=8,  # Number of apples
                 n_bombs=8,  # Number of bombs
                 activity_range=6.,  # Size of spawning area for generating apples/bombs (x,y)
                 robot_object_spacing=2.,  # Space between robot and objects on episode reset
                 catch_range=1.,  # How close robot needs to be to pickup object
                 n_bins=10,  #
                 sensor_range=6.,  # Robot sensor range
                 sensor_span=math.pi,  # Robot sensor span (radians)
                 coef_inner_rew=0.,  # Coefficient for inner reward
                 dying_cost=-10,
                 ):
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.dying_cost = dying_cost
        self.objects = []
        tree = ET.parse(pth)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"  # Base object parameters
        )
        # Create a wall around
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)
        super().__init__(robot_class(file_path), render)

    def reset(self):
        s = super().reset()
        self._generate_apples_and_bombs()
        return self._get_current_obs(s)
    def _generate_apples_and_bombs(self):
        """
        Generates apple and bomb objects (but does not actually place them in the xml tree, so they won't render)
        """
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

    def _get_readings(self):
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.robot.body_xyz[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins
        ori = self.robot.body_rpy[-1]  # rotation about z axis (x,y plane)

        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb; ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings

    def step(self, action):
        s_r, inner_rew, done, info = super().step(action)
        info['inner_rew'] = inner_rew  # Reward based on robot power cost and such. Done if dies
        info['outer_rew'] = 0
        if done:
            return self._get_current_obs(s_r), self.dying_cost, done, info  # give a -10 rew if the robot dies
        x, y = self.robot.body_xyz[:2]
        reward = self.coef_inner_rew * inner_rew
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                    info['outer_rew'] = 1
                else:
                    reward = reward - 1
                    info['outer_rew'] = -1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return self._get_current_obs(s_r), reward, done, info


    def _get_current_obs(self, robot_state):
        """Append sensor readings to robot readings"""
        apple_readings, bomb_readings = self._get_readings()
        return np.concatenate([robot_state, apple_readings, bomb_readings])

if __name__ == "__main__":
    test = GatherEnv(render=True)
    s = test.reset()
    for t in range(10000):
        s,r,d,_ = test.step(test.action_space.sample())
        print(r)
        if d:
            s = test.reset()
            print(d)