import argparse
import os

import gymnasium as gym
import ray

from ray import air, tune
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env


CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'results')

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="ES",
    help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf2",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=80000,
    help="Number of iterations to train."
)
parser.add_argument(
    "--worker-num", type=int, default=51,
    help="Number of parallel workers."
)
parser.add_argument(
    "--ip-head", type=str, default=None,
    help="The IP address of the head node of the ray cluster."
)
parser.add_argument(
    "--redis-password", type=str, default=None,
    help="The password to connect to the ray cluster."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=2e8,
    help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=23.5,
    help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--episodes_per_batch', type=int, default=4000)
parser.add_argument('--sigma', type=float, default=0.02)


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    if args.ip_head is not None:
        ray.init(address=args.ip_head, 
                 _redis_password=args.redis_password,
                 local_mode=False)
    else:
        ray.init(local_mode=args.local_mode)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    env_name = 'LoadRestoration13BusUnbalancedSimplified-v0'

    def env_creator(config):
        import clr_envs
        env = gym.make(env_name)
        return env
    
    register_env(env_name, env_creator)

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env_name, env_config={})
        .framework(args.framework)
        .rollouts(num_rollout_workers=args.worker_num)
        .training(episodes_per_batch=args.episodes_per_batch, 
                  stepsize=args.lr,
                  model={"fcnet_hiddens": [256, 256, 128, 128, 64, 64, 38]},
                  noise_stdev=args.sigma)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    tuner = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop, local_dir=LOG_PATH,
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5)),
    )
    results = tuner.fit()

    if args.as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()