from gymnasium.envs.registration import register

register(
    id='LoadRestoration13BusUnbalancedSimplified-v0',
    entry_point='clr_envs.envs:LoadRestoration13BusUnbalancedSimplified'
)

register(
    id='LoadRestoration13BusUnbalancedFull-v0',
    entry_point='clr_envs.envs:LoadRestoration13BusUnbalancedFull'
)
