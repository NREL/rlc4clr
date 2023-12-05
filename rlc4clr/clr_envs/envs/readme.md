# CLR Environments

## Environments

We created three CLR environment classes in [clr_13bus_envs.py](clr_13bus_envs.py), following the [Gymnasium](https://gymnasium.farama.org/api/env/) API format.

They are:
- `LoadRestoration13BusBaseEnv` inherits from `gymnasium.Env`. It serves as the base CLR environment that loads the system/data and provides common functions.
- `LoadRestoration13BusUnbalancedSimplified` inherits from the base CLR environment. It is the environment for the simplified Stage I problem, which controls the generators' set points only, while loads are picked up via a heuristic greedy way. Perfect 1-hour renewable forecasts are provided in the state.
- `LoadRestoration13BusUnbalancedFull` inherits from the base CLR environment. It is the environment for the original harder problem, which controls both generators' set points and load pickup. Renewable forecasts can be configured to have different lookahead lengths as well as different error levels.


Note: the environment is currently hard-coded for the 13-bus system, later we can update the code to make it more generic.

## Using the environments

### Create an instance

Because the environments have been registered in [\_\_init__.py](../__init__.py), we can use the Gymnasium API to load environments.

```
import gymnasium as gym
import clr_envs

env_sim = gym.make('LoadRestoration13BusUnbalancedSimplified-v0')

env_full = gym.make('LoadRestoration13BusUnbalancedFull-v0')
```

Because the full environment supports different lookahead lengths and error levels, the following example shows how configure it after creation.

```
env_config = {
    'error_level': 0.15,  # 0.0, 0.05, 0.1, 0.15, 0.2, 0.25
    'forecast_len': 2  # 1, 2, 4, 6
}
env_full.set_configuration(env_config)
```
By default, the environment uses the configuration of `{'error_level': 0.1, 'forecast_len': 1}`.

### Initialize an episode

The environment can be initialized by calling the reset function.

```
obs_full, _ = env_full.reset()
```

What happenes under the hood is that one scenario (a scenario is defined as a six-hour control horizon with specific renewable generation profiles) is randomly (uniformly, to be specific) sampled from the training scenario set (8856 in total).

When testing the trained RL policy, the following example allows specifying a specific unseen testing scenario (with an index larger than 8856):

```
obs_full, _ = env_full.reset(options={'start_index': 10000})
```

## Synthetic Forecasts

The figure below illustrates the synthetic forecasts used in this study.

- Black curve: the actual renewable generation profile over the six hour horizon. Unknown in advance to the realization.
- Blue curves: the forecasts available to the grid operator/RL controller. The solid blue line is the forecast for the current step and the trailing faded blue curves are forecasts at the previous few steps. 

This shows that the forecasts can be corrected as time passes by. For example, for the renewable generation at 0:30, it is predicted to be 0.4 (normalized) at 20:00, but as we get to 0:00 (realizing that the wind does not blow as hard as we expected), the forecast value is corrected to 0.3, which is closer to the actual realization (around 0.25).

<p align="center">
    <img src="../../../train/figs/wind_09.gif" alt="synthetic forecasts" width="70%"><br>
    <em>An example of the synthetic forecasts.</em>
</p>
