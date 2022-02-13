import gym
import numpy as np

from stable_baselines.common.vec_env import dummy_vec_env
from stable_baselines.common.policies import ActorCriticPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from gym_qubit.envs.qubit_env import QubitEnv
import tensorflow as tf

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                        act_fun=tf.tanh, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False )
        
        self._setup_init()

    def step(self, obs, state=None, mask=None, derministic=True):
        action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp], {self.obs_ph: obs})

        return action, value, neglogp
    
    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
        

# multiprocess environment
env = QubitEnv()

env_vec = make_vec_env(lambda: env, n_envs=30)

policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])

model = PPO2(MlpPolicy, env_vec, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=50000)

env.reset()

optimal_input = model.predict(np.array([0,-1]))

print('Optimal Input: {0}'.format(optimal_input[0]))

sum = 0
for i in range(0,100):
    env.reset()
    sum += env.step(optimal_input[0])[1]

print('Average Return After 100 Trials: {0}'.format(sum / 100))

