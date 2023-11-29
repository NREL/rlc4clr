#!/usr/bin/env python
"""

"""

import os

from pathlib import Path

import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.es import ES
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env
from tqdm import tqdm


CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_PATH = os.path.join(
    CURRENT_FILE_DIR, 'results/STG1/ES/trained_policy/checkpoint_000079')


def get_trained_agent(policy_checkpoint=None):
    """ Obtain trained agent from checkpoint.

    Args:
      policy_checkpoint: str, the absolute path to a trained RL policy.
    """

    env_name = 'LoadRestoration13BusUnbalancedSimplified-v0'

    def env_creator(config):
        import clr_envs
        env = gym.make(env_name)
        return env
    
    register_env(env_name, env_creator)

    config = (
        get_trainable_cls('ES')
        .get_default_config()
        .framework('tf2')
        .rollouts(num_rollout_workers=1)
        .training(model={"fcnet_hiddens": [256, 256, 128, 128, 64, 64, 38]})
    )

    agent = ES(config, env=env_name)
    checkpoint_path = (policy_checkpoint 
                       if policy_checkpoint is not None 
                       else DEFAULT_POLICY_PATH)
    agent.restore(checkpoint_path)
    
    print(config['env_config'])
    
    return agent

def rollout(agent, forecast_len):

    import clr_envs

    env_config = {"forecast_len": forecast_len}

    env = gym.make('LoadRestoration13BusUnbalancedSimplified-v0')
    # env.set_configuration(env_config)
    # env_full.unwrapped.set_configuration(clr_env_config)

    env_mh = gym.make('LoadRestoration13BusUnbalancedFull-v0')
    env_mh.set_configuration(env_config)

    features = []
    outputs = []

    
    # 0-8856 are the scneario ids used for training.
    for idx in tqdm(range(8856)):
        
        done = False
        state, _ = env.reset(options={'start_index': idx})
        state_mh, _ = env_mh.reset(options={'start_index': idx})

        while not done:

            action = agent.compute_single_action(state)

            p_st, st_angle, p_mt, mt_angle, wt_angle, pv_angle = action
            p_mt = (p_mt + 1) / 2.0

            # Active power
            p_pv = env.pv_profile[env.simulation_step]
            p_wt = env.wt_profile[env.simulation_step]
            p_st *= env.st_max_gen
            p_mt *= env.mt_max_gen

            p_st = env.st.validate_power(p_st)
            p_mt = env.mt.validate_power(p_mt)

            total_gen = [p_pv + p_wt + p_st + p_mt, 0]

            load_pickup_decision = [0.0 for _ in range(env.num_of_load)]
            total_gen_p = total_gen[0]
            load_idx = 0
            # Greedy pickup
            while total_gen_p > 0.0:
                try:
                    lp = min([1.0, total_gen_p / env.base_load[load_idx][0]])
                    load_pickup_decision[load_idx] = lp
                except IndexError:
                    break
                total_gen_p -= env.base_load[load_idx][0]
                load_idx += 1

            load_pickup_decision = [x * 2 - 1 for x in load_pickup_decision]

            action_new = (load_pickup_decision 
                          + [action[0], st_angle, wt_angle, pv_angle])
            
            assert (state[:12] == state_mh[:12]).all()

            state_new = np.append(state_mh[: 24 * forecast_len], state[24:])
            assert len(state_new) == 24 * forecast_len + 20

            features.append(state_new)
            outputs.append(action_new)

            next_state, reward, done, truncated, info = env.step(action)
            state = next_state

            state_mh, r, terminated, truncated, i = env_mh.step(action_new)

    features = np.array(features)
    outputs = np.array(outputs)

    folder_name = 'trajectory_data/' + str(forecast_len) + '_hours/'
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    np.savetxt(folder_name + 'features.csv', features, delimiter=',')
    np.savetxt(folder_name + 'outputs.csv', outputs, delimiter=',')


if __name__ == "__main__":

    for fl in [1, 2, 4, 6]:
        print("Generating data for forecast length: %d" % fl)
        agent = get_trained_agent()
        rollout(agent, fl)
