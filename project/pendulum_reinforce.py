from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools
import pandas as pd
from PIL import Image
import time
import datetime

from config_space import get_config_space

if "../lib/envs" not in sys.path:
    sys.path.append("../lib/envs")
from pendulum import PendulumEnv

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
reward_func = True
env = PendulumEnv(reward_function = reward_func)
state_space_size = env.observation_space.shape[0]
action_bound = 2
action_space_size = env.action_space.shape[0]
tau=0.001


class ActorNetwork():
    def __init__(self):
        tau = 0.001

        self.state, self.unscaled_output, self.output = self._build_model()
        #self.network_params = tf.trainable_variables()

        self.target = tf.placeholder(shape=[None], dtype=tf.float32)

        self.objective = -tf.reduce_mean(tf.log(self.output) * (self.target))
        #raise NotImplementedError("Softmax output not implemented.")

        self.optimizer = tf.train.AdamOptimizer(0.00001)
        self.train_op = self.optimizer.minimize(self.objective)

    def _build_model(self):
        states_pl = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

        fc1 = tf.contrib.layers.fully_connected(states_pl, 200, activation_fn=tf.nn.relu,
          weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))
        fc2 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=tf.nn.relu,
          weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))
        fc3 = tf.contrib.layers.fully_connected(fc2, 400, activation_fn=tf.nn.relu,
          weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))
        unscaled_output = tf.contrib.layers.fully_connected(fc3, action_space_size, activation_fn=tf.nn.tanh,
          weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

        output = tf.multiply(unscaled_output,action_bound)

        return(states_pl, unscaled_output, output)

    def predict(self, sess, states):

        return sess.run(self.output, { self.state: states })

    #def predict_target(self, sess, states):

    #   return sess.run(self.output_target, { self.state_target: states })

    def update(self, sess, states, targets):

        feed_dict = { self.state: states, self.target: targets  }
        sess.run(self.train_op, feed_dict)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def reinforce(sess, env, policy, noise, num_episodes, max_time_per_episode = 200, discount_factor=0.9):
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

  for i_episode in range(num_episodes):
    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    state = env.reset()

    # To log the best_policy 
    best_policy_reward = 0

    print("Episode {}/{} ({})".format(i_episode + 1, 
      num_episodes, stats.episode_rewards[i_episode - 1]),end='\r')
    sys.stdout.flush()
    
    for t in range(max_time_per_episode):

      # Stop if the max_time_per_episode is reached
      #if (stats.episode_lengths[i_episode] + 1) == max_time_per_episode:
      #  break
            
      action = policy.predict(sess, np.reshape(state, (1, 3))) + noise()

      next_state, reward, done, _ = env.step(action[0])

      episode.append((state, action, next_state, reward, done))

      # Update statistics
      stats.episode_rewards[i_episode] += reward
      stats.episode_lengths[i_episode] = t

      if done:

        break

      state = next_state

    for i,e in enumerate(episode):

      #func approximator to the baseline
      #constant baseline
      #target-baseline
      #compatible baseline in case of non-constatnt baseline 
      # Get the total return 
      G = sum([x[3]*(discount_factor**i) for i, x in enumerate(episode[i:])])
      #baseline = V[e[0]]

      # e[0]: state, e[1]: predicted_action
      # Update the policy 

      policy.update(sess,[e[0]], [G])

  #mean_G = np.mean(stats.episode_rewards, axis = 1)
  #print("Mean return:", mean_G)

  return stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig('episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig('reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)

if __name__ == "__main__":
  tf.reset_default_graph()
  p = ActorNetwork()
  noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size))

  sess = tf.Session()
  start_time = time.time()
  print("Starting at : ", datetime.datetime.now().strftime("%H:%M:%S"))
  sess.run(tf.global_variables_initializer())

  stats = reinforce(sess, env, p, noise, 3000)

  print("--- %s seconds ---" % (time.time() - start_time))

  print("Ended at : ", datetime.datetime.now().strftime("%H:%M:%S"))
  plot_episode_stats(stats)

