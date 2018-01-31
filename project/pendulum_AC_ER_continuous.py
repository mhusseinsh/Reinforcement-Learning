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
env = PendulumEnv(reward_function = True)
state_space_size = env.observation_space.shape[0]
action_bound = 2
action_space_size = env.action_space.shape[0]
tau=0.001

class CriticNetwork():
	"""
	Neural Network class based on TensorFlow.
	"""
	def __init__(self, num_actor_vars):
		tau = 0.001


		self.state, self.state_value = self._build_model()
		self.network_params = tf.trainable_variables()[num_actor_vars:]

		self.state_target, self.state_value_target = self._build_model()
		self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

		# Op for periodically updating target network with online network weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], tau) + \
				tf.multiply(self.target_network_params[i], 1. - tau))
				for i in range(len(self.target_network_params))]

		#[None, 1]
		self.predicted_v = tf.placeholder(shape=[None], dtype=tf.float32)

		self.losses = tf.squared_difference(self.predicted_v, self.state_value)
		self.loss = tf.reduce_mean(self.losses)


		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
		self.train_op = self.optimizer.minimize(self.loss)

	def _build_model(self):
		state = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

		# Create fully connected layers such that one is the others's input
		fc1 = tf.contrib.layers.fully_connected(state, 200, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		fc2 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		fc3 = tf.contrib.layers.fully_connected(fc2, 300, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		state_value = tf.contrib.layers.fully_connected(fc3, action_space_size,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		return (state, state_value)


	def predict(self, sess, states):
	
		prediction = sess.run(self.state_value, { self.state: states})
		# will return n action predictions
		return prediction

	def predict_target(self, sess, states):
	
		return sess.run(self.state_value_target, { self.state_target: states})

	def update(self, sess, states, predicted_v):
		
		# Get input (states), labels (actions), and predictions (targets) 
		feed_dict = { self.state: states, self.predicted_v: predicted_v }

		# Compute loss and update
		sess.run([self.state_value, self.train_op],feed_dict)

	def update_target(self, sess):
		sess.run(self.update_target_network_params)

class ActorNetwork():
	def __init__(self,batch_size):
		tau = 0.001
		self.batch_size = batch_size

		self.states_pl, self.output = self._build_model()
		self.network_params = tf.trainable_variables()

		self.state_target, self.output_target = self._build_model()
		self.target_network_params = tf.trainable_variables()[len(self.network_params):]

		# Op for periodically updating target network with online network weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], tau) + \
				tf.multiply(self.target_network_params[i], 1. - tau))
				for i in range(len(self.target_network_params))]

		self.predicted_td = tf.placeholder(shape=[None], dtype=tf.float32)

		self.objective = -tf.reduce_mean(tf.log(self.output) * (self.predicted_td))


		self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
		self.train_op = self.optimizer.minimize(self.objective)
		
		self.num_trainable_vars = len(
			self.network_params) + len(self.target_network_params)

	def _build_model(self):
		states_pl = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

		batch_size = tf.shape(states_pl)[0]

		fc1 = tf.contrib.layers.fully_connected(states_pl, 200, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		fc2 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		#fc3 = tf.contrib.layers.fully_connected(fc2, 400, activation_fn=tf.nn.relu,
		#  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		unscaled_output = tf.contrib.layers.fully_connected(fc2, action_space_size, activation_fn=tf.nn.tanh,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

		output = tf.multiply(unscaled_output, action_bound)

		return states_pl, output

	def predict(self, sess, states):

		return sess.run(self.output, { self.states_pl: states })

	def predict_target(self, sess, states):

		return sess.run(self.output_target, { self.state_target: states })

	def update(self, sess, states, td_targets):

		feed_dict = { self.states_pl: states, self.predicted_td: td_targets}
		sess.run(self.train_op, feed_dict)

	def update_target(self, sess):
		sess.run(self.update_target_network_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars

class ReplayBuffer:
	#Replay buffer for experience replay. Stores transitions.
	def __init__(self):
		self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
		self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])

	def add_transition(self, state, action, next_state, reward, done):
		self._data.states.append(state)
		self._data.actions.append(action)
		self._data.next_states.append(next_state)
		self._data.rewards.append(reward)
		self._data.dones.append(done)

	def sample_batch(self, batch_size):
		batch_indices = np.random.choice(len(self._data.states), batch_size)
		batch_states = np.array([self._data.states[i] for i in batch_indices])
		batch_actions = np.array([self._data.actions[i] for i in batch_indices])
		batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
		batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
		batch_dones = np.array([self._data.dones[i] for i in batch_indices])
		return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

def pendulum(sess, env, actor, critic, num_episodes = 200, max_time_per_episode = 200, discount_factor = 0.9, batch_size = 128):

	# Keeps track of useful statistics
	stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	# Create the experiences memory
	replay_memory = ReplayBuffer()

	#actor.update_target(sess)
	critic.update_target(sess)
	
	for i_episode in range(num_episodes):
		# Print out which episode we're on, useful for debugging.
		# Also print reward for last episode
		print("Episode {}/{} ({})".format(i_episode + 1, 
			num_episodes, stats.episode_rewards[i_episode - 1]),end='\r')
		sys.stdout.flush()

		state = env.reset()

		for t in itertools.count():

			# Stop if the max_time_per_episode is reached
			if (stats.episode_lengths[i_episode] + 1) == max_time_per_episode:
				break

			action = actor.predict(sess, np.reshape(state, (1, 3)))


			next_state, reward, done, _ = env.step(action[0])
			#print(reward)
			
			replay_memory.add_transition(state, action, next_state, reward, done)

			batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = replay_memory.sample_batch(batch_size)
			
			batch_v_next = critic.predict_target(sess, batch_next_states)

			batch_td_targets = batch_rewards + np.invert(batch_dones).astype(np.float32) * discount_factor * batch_v_next[0]

			batch_states = np.array(batch_states)

			critic.update(sess, batch_states, batch_td_targets)

			#???
			batch_v_current = critic.predict_target(sess, [state])
			batch_td_errors = batch_td_targets - batch_v_current[0]

			actor.update(sess, [state], batch_td_errors)
			critic.update_target(sess)

			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			state = next_state

			if done:
				break

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
	stats = []
	

	actor = ActorNetwork(batch_size = 128)

	critic = CriticNetwork(actor.get_num_trainable_vars())


	sess = tf.Session()
	start_time = time.time()
	print("Starting at : ", datetime.datetime.now().strftime("%H:%M:%S"))
	sess.run(tf.global_variables_initializer())

	stats = pendulum(sess, env, actor, critic)

	print("--- %s seconds ---" % (time.time() - start_time))

	print("Ended at : ", datetime.datetime.now().strftime("%H:%M:%S"))
	plot_episode_stats(stats)
