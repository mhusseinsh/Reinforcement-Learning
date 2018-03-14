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
state_space_size = 3

class CriticNetwork():
	"""
	Neural Network class based on TensorFlow.
	"""
	def __init__(self, learning_rate,
					   num_fc_units_1, 
					   num_fc_units_2, 
					   num_fc_units_3, 
					   num_fc_units_4):

		self.learning_rate = learning_rate
		self.num_fc_units_1 = num_fc_units_1
		self.num_fc_units_2 = num_fc_units_2
		self.num_fc_units_3 = num_fc_units_3
		self.num_fc_units_4 = num_fc_units_4

		self._build_model()


	def _build_model(self):

		# Create placeholders for input, output and actions of the network 
		self.state = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
		# Added later
		batch_size = tf.shape(self.state)[0]

		self.target = tf.placeholder(shape=[None], dtype=tf.float32)
		#self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

		# Create fully connected layers such that one is the others's input
		self.fc1 = tf.contrib.layers.fully_connected(self.state, self.num_fc_units_1, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self.num_fc_units_2, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc3 = tf.contrib.layers.fully_connected(self.fc2, self.num_fc_units_3, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc4 = tf.contrib.layers.fully_connected(self.fc3, self.num_fc_units_4, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		# Output layer will have linear activation with "activation_fn=None"
		# Name changed to predictions
		self.state_value = tf.contrib.layers.fully_connected(self.fc4, 1, activation_fn=None,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		# Prepare loss by using mean squared error
		# [all_actions] wrong: this should be the target(s)
		# clarify what is y: target or prediction
		# when predicting check which action was taken, then choose the index of the action for the prediction

		# Get the predictions for the chosen actions only
		#gather_indices = tf.range(batch_size) * tf.shape(self.prediction)[1] + self.actions
		#self.action_prediction = tf.gather(tf.reshape(self.prediction, [-1]), gather_indices)

		# Calcualte the loss
		# self.losses = tf.losses.mean_squared_error([all_actions], self.y, reduction =tf.losses.Reduction.NONE)
		self.losses = tf.squared_difference(self.target, self.state_value)
		self.loss = tf.reduce_mean(self.losses)

		# not needed
		# self.loss = tf.reduce_mean(self.losses)

		# Use Adam as optimizer 
		# use a small lr
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)

		# Initiate global variables
		#init = tf.global_variables_initializer()

	def predict(self, sess, states):
		"""
		Args:
			sess: TensorFlow session
			states: array of states for which we want to predict the actions.
		Returns:
			The prediction of the output tensor.
		"""
		prediction = sess.run(self.state_value, { self.state: states})
		# will return n action predictions
		return prediction

	def update(self, sess, states, targets):
		"""
		Updates the weights of the neural network, based on its targets, its
		predictions, its loss and its optimizer.

		Args:
			sess: TensorFlow session.
			states: [current_state] or states of batch
			actions: [current_action] or actions of batch
			targets: [current_target] or targets of batch
		"""
		# Get input (states), labels (actions), and predictions (targets) 
		feed_dict = { self.state: states, self.target: targets }

		# Compute loss and update
		sess.run([self.train_op, self.loss],feed_dict)

class ActorNetwork():
	def __init__(self, learning_rate, 
					   action_space_size, 
					   num_fc_units_1, 
					   num_fc_units_2, 
					   num_fc_units_3, 
					   num_fc_units_4):

		self.learning_rate = learning_rate
		self.action_space_size = action_space_size
		self.num_fc_units_1 = num_fc_units_1
		self.num_fc_units_2 = num_fc_units_2
		self.num_fc_units_3 = num_fc_units_3
		self.num_fc_units_4 = num_fc_units_4

		self._build_model()

	def _build_model(self):
		"""
		Builds the Tensorflow graph.
		"""
		self.states_pl = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
		# The TD target value
		self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32)
		# Integer id of which action was selected
		self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32)

		batch_size = tf.shape(self.states_pl)[0]

		self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, self.num_fc_units_1, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self.num_fc_units_2, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc3 = tf.contrib.layers.fully_connected(self.fc2, self.num_fc_units_3, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc4 = tf.contrib.layers.fully_connected(self.fc3, self.num_fc_units_4, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc5 = tf.contrib.layers.fully_connected(self.fc4, self.action_space_size, activation_fn=None,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		self.predictions = tf.contrib.layers.softmax(self.fc5)

		# Get the predictions for the chosen actions only
		gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl

		#print("indices: ",gather_indices)

		self.action_predictions = tf.gather(tf.reshape(self.predictions,[-1]), gather_indices)

		# -----------------------------------------------------------------------
		# TODO: Implement the policy gradient objective. Do not forget to negate
		# -----------------------------------------------------------------------
		# the objective, since the predefined optimizers only minimize in
		# tensorflow.

		#-tf.reducemean(tf.log...)
		#self.objective = -tf.log(self.action_predictions) * self.targets_pl
		#self.objective = -tf.reduce_mean(tf.log(self.action_predictions) * (self.targets_pl - baseline))
		self.objective = -tf.reduce_mean(tf.log(self.action_predictions) * (self.targets_pl))
		#raise NotImplementedError("Softmax output not implemented.")

		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		self.train_op = self.optimizer.minimize(self.objective)

		#init = tf.global_variables_initializer()

	def predict(self, sess, s):
		"""
		Args:
		  sess: TensorFlow session
		  states: array of states for which we want to predict the actions.
		Returns:
		  The prediction of the output tensor.
		"""
		#exit()
		p = sess.run(self.predictions, { self.states_pl: s })
		p = p.reshape(action_space_size,)
		return np.random.choice(action_indices, p=p), p

	def update(self, sess, s, a, y):
		"""
		Updates the weights of the neural network, based on its targets, its
		predictions, its loss and its optimizer.

		Args:
		  sess: TensorFlow session.
		  states: [current_state] or states of batch
		  actions: [current_action] or actions of batch
		  targets: [current_target] or targets of batch
		"""
		feed_dict = { self.states_pl: s, self.targets_pl: y, self.actions_pl: a }
		sess.run(self.train_op, feed_dict)

def pendulum(sess, env, actor, critic, num_episodes, max_time_per_episode, discount_factor):

	# Keeps track of useful statistics
	stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	print("here")
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

			# get an action from policy : actor gives the action
			action_predicted, action_probs = actor.predict(sess, [state])
			#print("action: ",action_predicted)
			action = action_index[action_predicted]
			# take the action given by the actor
			next_state, reward, done, _ = env.step([action])
			#print("reward: ",reward)

			# predict value of next state
			v_next = critic.predict(sess, [next_state])
			# predict value of this state
			v_current = critic.predict(sess, [state])

			td_target = reward + discount_factor * v_next[0] 
			
			td_error = td_target - v_current[0]

			# update critic : policy evaluation 
			critic.update(sess, [state], td_target)

			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# update actor: policy improvement
			action_ind = inv_action_index[action]
			actor.update(sess, [state], [action_ind], td_error)

			state = next_state
			#print("end of episode")
			if done:
				break

	return stats

def plot_episode_stats(stats, i ,smoothing_window=10, noshow=True):
	# Plot the episode length over time
	fig1 = plt.figure(figsize=(10,5))
	plt.plot(stats.episode_lengths)
	plt.xlabel("Episode")
	plt.ylabel("Episode Length")
	plt.title("Episode Length over Time")
	fig1.savefig('Model_'+ str(i) + '_episode_lengths.png')
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
	fig2.savefig('Model_'+ str(i) + '_reward.png')
	if noshow:
		plt.close(fig2)
	else:
		plt.show(fig2)

if __name__ == "__main__":
	stats = []
	reward_func = True
	env = PendulumEnv(reward_func)

	config_dict = {}
	budget = 20
	for i in range(budget):
		config_space_dist = get_config_space().sample_configuration()
		config_dict["config_dict{0}".format(i)] = config_space_dist.get_dictionary()

	for i in range(budget):

		print("Model ",i)
		print("Configuration used: ", config_dict.get("config_dict{0}".format(i)))

		# An array to use for action indexes
		action_indices = config_dict.get("config_dict{0}".format(i)).get('action_indices')
		intervals = np.linspace(-2.0, 2.0, num=action_indices)
		action_index = {key: value for key, value in zip(range(action_indices),intervals)}
		inv_action_index = {value: key for key, value in action_index.items()}
		action_space_size = action_indices
		

		actor = ActorNetwork(config_dict.get("config_dict{0}".format(i)).get('learning_rate'),
							action_space_size,
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_1'),
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_2'),
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_3'),
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_4'))

		critic = CriticNetwork(config_dict.get("config_dict{0}".format(i)).get('learning_rate'),
								config_dict.get("config_dict{0}".format(i)).get('num_fc_units_1'),
								config_dict.get("config_dict{0}".format(i)).get('num_fc_units_2'),
								config_dict.get("config_dict{0}".format(i)).get('num_fc_units_3'),
								config_dict.get("config_dict{0}".format(i)).get('num_fc_units_4'))

		sess = tf.Session()
		start_time = time.time()
		print("Starting at : ", datetime.datetime.now().strftime("%H:%M:%S"))
		sess.run(tf.global_variables_initializer())

		stats.append(pendulum(sess, env, actor, critic, 
			config_dict.get("config_dict{0}".format(i)).get('num_episodes'), 
			200,
			config_dict.get("config_dict{0}".format(i)).get('discount_factor')))


		print("--- %s seconds ---" % (time.time() - start_time))

		print("Ended at : ", datetime.datetime.now().strftime("%H:%M:%S"))
		plot_episode_stats(stats[i], i)
		name = 'Model_' + str(i) + '_configs.txt'
		configs = ("Configuration used: ", config_dict.get("config_dict{0}".format(i)))
		np.savetxt(name, configs, fmt='%5s', delimiter=',')
		
	plot_episode_stats(stats, i)

	"""for _ in range(200):
					state = env.reset()
					for _ in range(1000):
						env.render()
						action_predicted, action_probs = actor.predict(sess, [state])
						action = action_index[action_predicted]
						state, _, done, _ = env.step([action])
			
						if done:
							break"""