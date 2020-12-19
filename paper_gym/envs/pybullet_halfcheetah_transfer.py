import gym
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env import HalfCheetahMuJoCoEnv
import numpy as np

# Half cheetah with reversible direction
class HalfCheetahMuJoCoEnv_Directional(HalfCheetahMuJoCoEnv):
    def __init__(self, dir=1.):
        super().__init__()
        self.goal_dir = dir

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        potential = self.goal_dir * self.robot.calc_potential()  # Apply goal direction to progress reward
        power_cost = -0.1 * np.square(a).sum()
        state = self.robot.calc_state()
        done = False
        self.rewards = [
            potential,
            power_cost
        ]
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def transfer(self, goal_dir=0.):
        self.goal_dir = 1. if self.goal_dir == -1 else -1.
        return self.goal_dir

# half cheetah with target velocity
class HalfCheetahMuJoCoEnv_Velocity(HalfCheetahMuJoCoEnv):
    def __init__(self, velocity=0.):
        super().__init__()
        self.goal_velocity = velocity

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        potential = -1 * abs(self.robot.calc_potential() - self.goal_velocity)
        power_cost = -0.05 * np.square(a).sum()
        state = self.robot.calc_state()
        done = False
        self.rewards = [
            potential,
            power_cost
        ]
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def transfer(self, target_velocity=0.):
        self.goal_velocity = target_velocity
        return target_velocity

if __name__ == "__main__":
    e1 = HalfCheetahMuJoCoEnv()
    e2 = HalfCheetahMuJoCoEnv_Velocity()
    e3= HalfCheetahMuJoCoEnv_Directional()
    e1.reset(); e2.reset(); e3.reset()
    for i in range(1000):
        a = e1.action_space.sample()
        _, r1, d, _ = e1.step(a)
        _, r2, d, _ = e2.step(a)
        _, r3, d, _ = e3.step(a)
        print("{}_{}_{}".format(r1, r2, r3))