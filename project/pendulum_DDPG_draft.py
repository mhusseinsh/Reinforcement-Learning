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
import csv

#from config_space import get_config_space

if "../lib/envs" not in sys.path:
	sys.path.append("../lib/envs")
from pendulum import PendulumEnv

EpisodeStats = namedtuple("Stats",["episode_q_values", "episode_rewards", "episode_loss"])

class CriticNetwork():

	def __init__(self, state_space_size, action_space_size, num_actor_vars, tau):

		self.state, self.action, self.action_value = self._build_model()
		self.network_params = tf.trainable_variables()[num_actor_vars:]

		self.state_target, self.action_target, self.action_value_target = self._build_model()
		self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], tau) + \
				tf.multiply(self.target_network_params[i], 1. - tau))
				for i in range(len(self.target_network_params))]

		self.predicted_q = tf.placeholder(shape=[None, 1], dtype=tf.float32)

		self.losses = tf.squared_difference(self.predicted_q, self.action_value)
		self.loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.train_op = self.optimizer.minimize(self.loss)

		self.action_gradient = tf.gradients(self.action_value, self.action)

	def _build_model(self):
		state = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
		action = tf.placeholder(shape=[None, action_space_size], dtype=tf.float32)

		fc1 = tf.contrib.layers.fully_connected(state, 200, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

		concat = tf.concat([fc1, action], axis = 1)

		fc2 = tf.contrib.layers.fully_connected(concat, 400, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

		#concat = tf.concat([fc1, fc2], axis = 1)

		#concat = tf.concat([state, action], axis = 1)

		fc3 = tf.contrib.layers.fully_connected(fc2, 200, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

		action_value = tf.contrib.layers.fully_connected(fc3, action_space_size, activation_fn=None,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

		return (state, action, action_value)

	def action_gradients(self, sess, states, actions):

		return sess.run(self.action_gradient, { self.state: states, self.action: actions})

	def predict(self, sess, states, actions):

		prediction = sess.run(self.action_value, { self.state: states, self.action: actions})

		return prediction

	def predict_target(self, sess, states, actions):

		return sess.run(self.action_value_target, { self.state_target: states, self.action_target: actions})

	def update(self, sess, states, actions, predicted_q):

		feed_dict = { self.state: states, self.action: actions, self.predicted_q: predicted_q }

		return sess.run([self.action_value, self.train_op],feed_dict)

	def update_target(self, sess):

		sess.run(self.update_target_network_params)

class ActorNetwork():
	def __init__(self, action_space_size, batch_size, noise, tau):

		self.batch_size = batch_size

		self.state, self.output = self._build_model()
		self.network_params = tf.trainable_variables()

		"""for parameter in self.network_params:
									parameter + noise()"""

		self.state_target, self.output_target = self._build_model()
		self.target_network_params = tf.trainable_variables()[len(self.network_params):]

		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.multiply(self.network_params[i], tau) + \
				tf.multiply(self.target_network_params[i], 1. - tau))
				for i in range(len(self.target_network_params))]

		self.action_gradient = tf.placeholder(shape=[None, action_space_size], dtype=tf.float32)

		self.actor_gradients = tf.gradients(self.output, self.network_params, -self.action_gradient)

		self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).apply_gradients(zip(self.actor_gradients, self.network_params))

		self.num_trainable_vars = len(
			self.network_params) + len(self.target_network_params)

	def _build_model(self):
		states_pl = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

		#weights_regularizer = tf.contrib.layers.l1_regularizer(0.01)

		batch_size = tf.shape(states_pl)[0]

		fc1 = tf.contrib.layers.fully_connected(states_pl, 200, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))
		fc2 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))
		#fc3 = tf.contrib.layers.fully_connected(fc2, 300, activation_fn=tf.nn.relu,
		  #weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

		unscaled_output = tf.contrib.layers.fully_connected(fc2, action_space_size, activation_fn=tf.nn.tanh,
		  weights_initializer=tf.random_uniform_initializer(-0.003, 0.003))

		output = tf.multiply(unscaled_output,action_bound)

		return(states_pl, output)

	def predict(self, sess, states):

		return sess.run(self.output, { self.state: states })

	def predict_target(self, sess, states):

		return sess.run(self.output_target, { self.state_target: states })

	def update(self, sess, states, action_gradients):

		feed_dict = { self.state: states, self.action_gradient: action_gradients }

		sess.run(self.optimizer, feed_dict)

	def update_target(self, sess):

		sess.run(self.update_target_network_params)

	def get_num_trainable_vars(self):

		return self.num_trainable_vars

class ReplayBuffer:

	def __init__(self, num_episodes):
		self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards"])
		self._data = self._data(states=[], actions=[], next_states=[], rewards=[])
		self.num_episodes = num_episodes

	def add_transition(self, state, action, next_state, reward):
		self._data.states.append(state)
		self._data.actions.append(action)
		self._data.next_states.append(next_state)
		self._data.rewards.append(reward)

	def sample_batch(self, batch_size):
		batch_indices = np.random.choice(len(self._data.states), batch_size)
		batch_states = np.array([self._data.states[i] for i in batch_indices])
		batch_actions = np.array([self._data.actions[i] for i in batch_indices])
		batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
		batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
		return batch_states, batch_actions, batch_next_states, batch_rewards

	def clear(self):
		to_cut = int(round(self.num_episodes * 0.75))
		#to_cut = self.num_episodes / 2
		temp_states = self._data.states[to_cut:]
		temp_actions = self._data.actions[to_cut:]
		temp_next_states = self._data.next_states[to_cut:]
		temp_rewards = self._data.rewards[to_cut:]

		new_replay = ReplayBuffer(num_episodes)

		for t in range(len(temp_actions)):
			new_replay.add_transition(temp_states[t], temp_actions[t], temp_next_states[t], temp_rewards[t])

		return new_replay

	def size(self):
		return len(self._data.states)

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

def pendulum(sess, env, actor, critic, actor_noise, replay_memory, num_episodes, max_time_per_episode, discount_factor, batch_size):

	stats = EpisodeStats(episode_q_values=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes), episode_loss=np.zeros(num_episodes))

	actor.update_target(sess)
	critic.update_target(sess)

	for i_episode in range(num_episodes):

		print("Episode {}/{} ({})".format(i_episode + 1,
			num_episodes, stats.episode_rewards[i_episode - 1]),end='\r')
		sys.stdout.flush()

		state = env.reset()

		for t in itertools.count():

			if t + 1 == max_time_per_episode:
				break

			action = actor.predict(sess, np.reshape(state, (1, 3))) + actor_noise()

			next_state, reward, _, _ = env.step(action[0])
			loss_log = []
			loss = 0
			if replay_memory.size() >= num_episodes:
				replay_memory = replay_memory.clear()

			replay_memory.add_transition(state, action, next_state, reward)

			if replay_memory.size() > batch_size * 3:

				batch_states, batch_actions, batch_next_states, batch_rewards = replay_memory.sample_batch(batch_size)

				action_from_target = actor.predict_target(sess, batch_next_states)

				target_q_values = critic.predict_target(sess, batch_next_states, action_from_target)

				td_targets = []

				for k in range(batch_size):

					td_targets.append(batch_rewards[k] + discount_factor * target_q_values[k])

				loss_log = critic.update(sess,batch_states, np.reshape(batch_actions,(batch_size, 1)), np.reshape(td_targets,(batch_size, 1)))
				loss = np.mean(loss_log[0])

				action_from_actor = actor.predict(sess,batch_states)

				gradients = critic.action_gradients(sess,batch_states, action_from_actor)

				actor.update(sess,batch_states, gradients[0])

				actor.update_target(sess)
				critic.update_target(sess)

			stats.episode_loss[i_episode] = loss
			stats.episode_rewards[i_episode] += reward
			stats.episode_q_values[i_episode] = critic.predict(sess, np.reshape(state, (1, 3)), np.reshape(action, (1, 1)))

			state = next_state

	return stats

def plot_episode_stats(stats, title, smoothing_window=10, noshow=False):

	if title == "Training":
		fig1 = plt.figure(figsize=(10,5))

		for i, stat in enumerate(stats):
			name = 'q_values.txt' + str(i)
			np.savetxt(name,stat.episode_q_values, delimiter=',', fmt='%.8f')
			plt.plot(stat.episode_q_values, label = i)
			plt.xlabel("Episode")
			plt.ylabel("Episode Q Values")
			plt.title("Training - Episode Q Values")
			fig1.savefig('training_q_values.png')

		plt.legend()

		if noshow:
			plt.close(fig1)
		else:
			plt.show(fig1)

		fig2 = plt.figure(figsize=(10,5))

		for i, stat in enumerate(stats):
			name = 'rewards' + str(i) + '.txt'
			np.savetxt(name, stat.episode_rewards, delimiter=',', fmt='%.8f')
			rewards = []
			rewards_mean = []
			with open(name,'r') as csvfile:
			    plots = csv.reader(csvfile)
			    for row in plots:
			        rewards.append(float(row[0]))
			        cum_sum = np.cumsum(rewards)
			        size = len(rewards)
			        rewards_mean.append(cum_sum[-1] / size)
			name = 'rewards_mean' + str(i) + '.txt'
			np.savetxt(name, rewards_mean, delimiter=',', fmt='%.8f')
			rewards_smoothed = pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
			plt.plot(rewards_smoothed, label = i)
			plt.xlabel("Episode")
			plt.ylabel("Episode Reward (Smoothed)")
			plt.title("Training - Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
			fig2.savefig('training_reward.png')


		plt.legend()

		if noshow:
			plt.close(fig2)
		else:
			plt.show(fig2)

		fig3 = plt.figure(figsize=(10,5))

		for i, stat in enumerate(stats):
			name = 'loss_' + str(i) + '.txt'
			np.savetxt(name, stat.episode_loss, delimiter=',', fmt='%.8f')
			loss_smoothed = pd.Series(stat.episode_loss).rolling(smoothing_window, min_periods=smoothing_window).mean()
			plt.plot(loss_smoothed, label = i)
			plt.xlabel("Episode")
			plt.ylabel("Episode Loss (Smoothed)")
			plt.title("Training - Episode Loss over Time (Smoothed over window size {})".format(smoothing_window))
			fig3.savefig('training_loss.png')

		plt.legend()

		if noshow:
			plt.close(fig3)
		else:
			plt.show(fig3)

		fig4 = plt.figure(figsize=(10,5))

		for i, stat in enumerate(stats):
			name = 'rewards' + str(i) + '.txt'
			rewards = []
			with open(name,'r') as csvfile:
				plots = csv.reader(csvfile)
				for row in plots:
					rewards.append(float(row[0]))
			#IS IT NEEDED?
			#rewards_smoothed = pd.Series(stat.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
			ax = fig4.add_subplot(111)
			bp = ax.boxplot(rewards)
			#bp = ax.boxplot(stat.episode_rewards)

			plt.xlabel("Episode")
			plt.ylabel("Episode Reward (Smoothed)")
			plt.title("Percentile".format(smoothing_window))
			fig4.savefig('percentile_reward.png', bbox_inches='tight')


		plt.legend()

		if noshow:
			plt.close(fig4)
		else:
			plt.show(fig4)

	else:
		fig2 = plt.figure(figsize=(10,5))

		#for i, stat in enumerate(stats):
		np.savetxt('rew.txt',stats.episode_rewards, delimiter=',', fmt='%.8f')
		rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
		plt.plot(rewards_smoothed)
		plt.xlabel("Episode")
		plt.ylabel("Episode Reward (Smoothed)")
		plt.title("Evaluation - Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
		fig2.savefig('evaluation_reward.png')


		#plt.legend()

		if noshow:
			plt.close(fig2)
		else:
			plt.show(fig2)


if __name__ == "__main__":

	stats = []

	reward_func = True
	env = PendulumEnv(reward_function = reward_func)
	state_space_size = env.observation_space.shape[0]
	action_bound = 2
	action_space_size = env.action_space.shape[0]

	max_time_per_episode = 200
	num_episodes = 10
	discount_factor = 0.99
	batch_size = 64
	tau = 0.001

	train = 1
	stats = []
	actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space_size))
	replay_memory = ReplayBuffer(num_episodes)

	for t in range(train):
		print("Training epoch ",t," starts...")

		actor = ActorNetwork(action_space_size, batch_size, actor_noise, tau)
		critic = CriticNetwork(state_space_size, action_space_size, actor.get_num_trainable_vars(), tau)

		sess = tf.Session()
		start_time = time.time()
		print("Starting at : ", datetime.datetime.now().strftime("%H:%M:%S"))

		sess.run(tf.global_variables_initializer())
		stats.append(pendulum(sess, env, actor, critic, actor_noise, replay_memory, num_episodes, max_time_per_episode, discount_factor, batch_size))

		print("Ended at : ", datetime.datetime.now().strftime("%H:%M:%S"))

	plot_episode_stats(stats, title = "Training", noshow = True)

	start_time = time.time()
	print("Starting at : ", datetime.datetime.now().strftime("%H:%M:%S"))

	evaluation = 100
	evaluation_stats = EpisodeStats(episode_q_values=np.zeros(evaluation), episode_rewards=np.zeros(evaluation), episode_loss=np.zeros(evaluation))
	for e in range(evaluation):
		print("Evaluation phase ",e, " starts....")
		state = env.reset()
		for _ in range(max_time_per_episode):
		  #env.render()
		  action = actor.predict(sess, np.reshape(state, (1, 3)))
		  next_state, reward, _, _ = env.step(action)
		  state = next_state
		  evaluation_stats.episode_rewards[e] += reward
		  #rewards.append(reward)
		#all_rewards.append(rewards)

	plot_episode_stats(evaluation_stats, title = "Evaluation", noshow = True)

	"""mean_rewards_one_episode = np.mean(all_rewards, axis = 0)
		fig = plt.figure(figsize=(10,5))
		plt.plot(mean_rewards_one_episode)
		plt.xlabel("Step")
		plt.ylabel("Episode Reward ")
		plt.title("Mean of all episodes")
		fig.savefig('reward_mean_episode.png')
		plt.show(fig)"""
