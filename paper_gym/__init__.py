from gym.envs.registration import register

register(
    id='IOC-FourRooms-v0',
    entry_point='paper_gym.envs:Fourrooms',
    max_episode_steps=2000,
    reward_threshold=1,
)


register(
    id='Torch-FourRooms-v0',
    entry_point='paper_gym.envs:FourRoomsTorch',
    reward_threshold=1,
)

register(
    id='Torch-CartPole-v0',
    entry_point='paper_gym.envs:CartPole',
    max_episode_steps=2000,
    reward_threshold=1,
)

register(
    id='Torch-Pendulum-v0',
    entry_point='paper_gym.envs:Pendulum',
    max_episode_steps=2000,
    reward_threshold=1,
)

register(
    id='HalfCheetah-Directional-v0',
    entry_point='paper_gym.envs:HalfCheetahMuJoCoEnv_Directional',
    max_episode_steps=1000,
)

register(
    id='HalfCheetah-Velocity-v0',
    entry_point='paper_gym.envs:HalfCheetahMuJoCoEnv_Velocity',
    max_episode_steps=1000,
)

register(
    id='HalfCheetahPGym-v0',
    entry_point='paper_gym.envs:HalfCheetahMuJoCoEnv',
    max_episode_steps=1000,
)


