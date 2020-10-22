from gym.envs.registration import register

register(
    id='IOC-FourRooms-v0',
    entry_point='paper_gym.envs:Fourrooms',
    max_episode_steps=2000,
    reward_threshold=1,
)
