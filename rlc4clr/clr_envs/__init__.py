from gym.envs.registration import register

register(
    id='LoadRestoration13BusUnbalanced-v0',
    entry_point='clr_envs.envs:LoadRestoration13BusUnbalancedV0'
)

register(
    id='LoadRestoration13BusUnbalanced-v1',
    entry_point='clr_envs.envs:LoadRestoration13BusUnbalancedV1'
)
