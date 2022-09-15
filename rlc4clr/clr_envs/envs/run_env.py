import gym

import clr_envs

# Get an environment
env = gym.make('LoadRestoration13BusUnbalanced-v0')

# Initialize the episode
s = env.reset()
done = False
reward = 0.0
step_cnt = 0

# Simulate until the episode terminates
while not done:

    # Determine the action
    # a = pi(s)  # In RL, the action is determined by the policy
    act = env.action_space.sample()  # Randomly choose action in this example.

    s, r, done, info = env.step(act)

    reward += r
    step_cnt += 1

print(reward)
print(step_cnt)

