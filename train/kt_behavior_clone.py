import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from tensorflow.python.keras.layers import Dense


parser = argparse.ArgumentParser()
parser.add_argument(
    "--forecast-len", type=int, default=2,
    help="The length (hour) of the forecasts used in RL state."
)


class BehaviorCloner(object):
    """ Cloning the behavior of Stage I trained policy in the Stage II format.

    See Fig. 3 and Algorithm I of our paper for more details.
    (https://ieeexplore.ieee.org/abstract/document/9903581)
    
    """

    def __init__(self, forecast_len) -> None:
        
        self.states = np.genfromtxt('trajectory_data/' + str(forecast_len)
                                    + '_hours/features.csv', delimiter=',')
        self.desired_actions = np.genfromtxt('trajectory_data/' 
                                             + str(forecast_len)
                                             + '_hours/outputs.csv', 
                                             delimiter=',')
        
        self.model_saving_path = ('transferred_model/' + str(forecast_len) 
                                  + '_hours/sl_model/model_checkpoint')

    def clone_behavior(self):
        """ Cloning the controller behavior.
        """

        nn_structure = [256, 256, 128, 128, 64, 64, 38]
        output_dim = 19

        # Note the last layer is a linear output.
        model = tf.keras.Sequential(
            [Dense(x, activation='tanh') for x in nn_structure] 
            + [Dense(output_dim)])

        model.compile(loss=tf.losses.MeanAbsoluteError(),
                      optimizer=tf.optimizers.Adam(learning_rate=0.00001))

        model.fit(self.states, self.desired_actions, epochs=120)

        model.save_weights(self.model_saving_path)

    
    def examine_behavior_cloning(self):

        ckpt_reader = tf.train.load_checkpoint(self.model_saving_path)

        # example_kernel = ckpt_reader.get_tensor(
        #     'layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE')
        # example_bias = ckpt_reader.get_tensor
        # ('layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE')

        nn_structure = [256, 256, 128, 128, 64, 64, 38]
        output_dim = 19

        model = tf.keras.Sequential([Dense(x, activation='tanh') 
                                     for x in nn_structure] 
                                     + [Dense(output_dim)])

        model.compile(loss=tf.losses.MeanAbsoluteError(), 
                      optimizer=tf.optimizers.Adam(learning_rate=0.00001))

        model.load_weights(self.model_saving_path).expect_partial()

        idx = 121  # A random index to start with.
        num_of_examples_to_show = 300 

        desired_outputs = self.desired_actions[idx: 
                                               idx + num_of_examples_to_show]
        model_outputs = model.predict(
            self.states[idx: idx + num_of_examples_to_show])
        
        desired_outputs = desired_outputs.reshape((-1,))
        model_outputs = model_outputs.reshape((-1,))

        plt.scatter(desired_outputs, model_outputs, alpha=0.3)
        plt.plot([-1, 1], [-1, 1], color='r', linestyle='dashed')
        plt.ylabel('Model outputs')
        plt.xlabel('Desired outputs')

        plt.savefig(self.model_saving_path.replace(
            '/sl_model/model_checkpoint', '/behavior_cloning_examine.png'),
            dpi=200)


if __name__ == "__main__":

    args = parser.parse_args()

    bc = BehaviorCloner(args.forecast_len)

    bc.clone_behavior()
    bc.examine_behavior_cloning()
