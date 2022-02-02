from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

##IMPORTANT THIS FILE IS FOR FOLLOWING THE TESNORFLOW EXAMPLE 
##The code is credited towards Tensorflow and can be found on their github


#HYPER PARAMETERS
num_iterations = 20000 # integer

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000

env_name = 'gym_qubit:qubit-v0'

env = suite_gym.load(env_name)
env.reset()

print('Observation Spec:')
print(env.time_step_spec().observation)

print('Reward Spect')
print(env.time_step_spec().reward)

print('Action Spec: ')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array(0.5, dtype=np.dtype('float32'))
next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

train_py_env = suite_gym(env_name)
eval_py_env = suite_gym(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)



