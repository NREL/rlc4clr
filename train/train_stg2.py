"""Example of using a custom ModelV2 Keras-style model.

https://github.com/ray-project/ray/blob/ray-2.7.1/rllib/examples/custom_keras_model.py
"""


import os
import sys

import gymnasium as gym
import numpy as np
import ray

from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env


CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'results/STG2')
sys.path.append(CURRENT_FILE_PATH)

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


# 13-bus unbalanced system
INPUT_LEN_CHOICE = {'1': 44, '2': 68, '4': 116, '6': 164}
OUTPUT_LEN = 19
NETWORK_SHAPE = [256, 256, 128, 128, 64, 64, 38]


def build_pretrained_network(look_ahead_len):

    input_len = INPUT_LEN_CHOICE[look_ahead_len]

    weights_file_path = os.path.join(
        CURRENT_FILE_PATH, 
        'transferred_model/' + look_ahead_len 
        + '_hours/sl_model/model_checkpoint')

    class CustomFullyConnectedNetwork4PPO(TFModelV2):
        """Custom model for policy gradient algorithms."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, 
                     name):
            super(CustomFullyConnectedNetwork4PPO, self).__init__(
                obs_space, action_space, num_outputs, model_config, name
            )
            # Read in weights learned from supervised learning.
            checkpoint_reader = tf.train.load_checkpoint(weights_file_path)

            inputs = tf.keras.layers.Input(shape=(input_len,), name="obs1")

            # 1. Policy network
            next_layer = inputs

            for idx in range(len(NETWORK_SHAPE)):

                init_kernel = tf.constant_initializer(
                    checkpoint_reader.get_tensor(
                        'layer_with_weights-' + str(idx) 
                        + '/_module/kernel/.ATTRIBUTES/VARIABLE_VALUE'))
                init_bias = tf.constant_initializer(
                    checkpoint_reader.get_tensor(
                        'layer_with_weights-' + str(idx) 
                        + '/_module/bias/.ATTRIBUTES/VARIABLE_VALUE'))

                next_layer = tf.keras.layers.Dense(
                    NETWORK_SHAPE[idx], name='fc_' + str(idx+1),
                    kernel_initializer=init_kernel, bias_initializer=init_bias, 
                    activation=tf.tanh)(next_layer)

            # Output layer
            init_kernel = tf.constant_initializer(
                checkpoint_reader.get_tensor(
                    'layer_with_weights-7' 
                    + '/_module/kernel/.ATTRIBUTES/VARIABLE_VALUE'))
            init_bias = tf.constant_initializer(
                checkpoint_reader.get_tensor(
                    'layer_with_weights-7' 
                    + '/_module/bias/.ATTRIBUTES/VARIABLE_VALUE'))
            mean_out = tf.keras.layers.Dense(
                OUTPUT_LEN, name='layer_out', kernel_initializer=init_kernel,
                bias_initializer=init_bias, activation=None)(next_layer)

            # We arbitrarily set std output to exp(-2.0) for exploration.
            init_bias = tf.constant_initializer(
                np.array([-5.0] * OUTPUT_LEN).reshape((OUTPUT_LEN, 1)))
            log_std_out = tf.keras.layers.Dense(
                OUTPUT_LEN, name='out_variance', kernel_initializer='zeros',
                bias_initializer=init_bias, activation=None)(next_layer)

            layer_out = tf.concat((mean_out, log_std_out), 1)

            # 2. Value Network
            next_layer = inputs

            for idx in range(len(NETWORK_SHAPE)):
                init_kernel = tf.constant_initializer(
                    checkpoint_reader.get_tensor(
                        'layer_with_weights-' + str(idx)
                        + '/_module/kernel/.ATTRIBUTES/VARIABLE_VALUE'))
                init_bias = tf.constant_initializer(
                    checkpoint_reader.get_tensor(
                        'layer_with_weights-' + str(idx) 
                        + '/_module/bias/.ATTRIBUTES/VARIABLE_VALUE'))

                next_layer = tf.keras.layers.Dense(
                    NETWORK_SHAPE[idx], name='fc_value_' + str(idx + 1), 
                    kernel_initializer=init_kernel, bias_initializer=init_bias, 
                    activation=tf.tanh)(next_layer)

            value_out = tf.keras.layers.Dense(
                1, name='fc_value_out', activation=None)(next_layer)

            self.base_model = tf.keras.Model(inputs, [layer_out, value_out])
            # self.register_variables(self.base_model.variables)

        def forward(self, input_dict, state, seq_lens):
            model_out, self._value_out = self.base_model(input_dict["obs"])
            return model_out, state

        def value_function(self):
            return tf.reshape(self._value_out, [-1])
        
    return CustomFullyConnectedNetwork4PPO


if __name__ == "__main__":

    from config_parser import create_parser

    parser = create_parser()

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    if args.ip_head is not None:
        ray.init(address=args.ip_head, 
                 _redis_password=args.redis_password,
                 local_mode=False)
    else:
        ray.init(local_mode=args.local_mode)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    env_name = 'LoadRestoration13BusUnbalancedFull-v0'

    def env_creator(config):
        import clr_envs
        env = gym.make(env_name)
        env.set_configuration(config)
        return env
    
    register_env(env_name, env_creator)

    pretrained_network = build_pretrained_network(str(args.forecast_len))

    ModelCatalog.register_custom_model(
        "cl_pretrained_network", pretrained_network
    )

    env_config = {'forecast_len': args.forecast_len,
                  'error_level': args.error_level}

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env_name, env_config=env_config)
        .framework(args.framework)
        .rollouts(num_rollout_workers=args.worker_num)
        .training(lr=args.lr, train_batch_size=args.train_batch_size,
                  model={"custom_model": 'cl_pretrained_network'},
                  _enable_learner_api=False)
        .rl_module(_enable_rl_module_api=False)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        # "time_total_s": args.run_hour * 3600 - 300
        # "episode_reward_mean": args.stop_reward,
    }

    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    exp_note = args.exp_note if args.exp_note is not None else ''
    tuner = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop, 
            storage_path=os.path.join(LOG_PATH, exp_note,
                                      str(args.forecast_len) + '_hours',
                                      str(int(args.error_level * 100)) + 'p'),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_frequency,
                num_to_keep=args.checkpoint_to_save,
                checkpoint_score_attribute='sampler_results/episode_reward_mean'
            )),
    )
    results = tuner.fit()

    if args.as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
