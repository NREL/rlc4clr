# Followed the example below.
# https://github.com/ray-project/ray/blob/master/rllib/examples/sb2rllib_rllib_example.py

import os

import gymnasium as gym

from ray.rllib.algorithms.es import ES
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.tune.registry import get_trainable_cls


CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

env_name = 'LoadRestoration13BusUnbalancedSimplified-v0'

def env_creator(config=None):
    import clr_envs
    env = gym.make(env_name)
    return env

register_env(env_name, env_creator)

# Use the same configuration as the one for training except the rollout worker
# number, which can be limited at 1.
config = (
        get_trainable_cls('ES')
        .get_default_config()
        .framework('tf2')
        .training(model={"fcnet_hiddens": [256, 256, 128, 128, 64, 64, 38]})
        .rollouts(num_rollout_workers=1)
    )

# Loading trained RL agent
agent = ES(config, env=env_name)
checkpoint_path = os.path.join(CURRENT_FILE_DIR.replace('rollout', 'train'), 
                               'results/ES1/ES_Policy/checkpoint_000002')
agent.restore(checkpoint_path)
print('agent restored.')

# Rollout the trained agent.
env = env_creator()
state, _ = env.reset()
terminated = False
total_rew = 0.0

while not terminated:
    act = agent.compute_single_action(state)
    state, reward, terminated, truncated, info = env.step(act)
    total_rew += reward

print(total_rew)
