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

if "../lib/envs" not in sys.path:
	sys.path.append("../lib/envs")
from pendulum import PendulumEnv

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

# An array to use for action indexes
state_space_size = 3

intervals = np.linspace(-2.0, 2.0, num=41)
action_indices = np.arange(0,41)
action_index = {key: value for key, value in zip(range(41),intervals)}
inv_action_index = {value: key for key, value in action_index.items()}
action_space_size = len(action_indices)

class CriticNetwork():
	"""
	Neural Network class based on TensorFlow.
	"""
	def __init__(self):
		self._build_model()

	def _build_model(self):

		# Create placeholders for input, output and actions of the network 
		self.state = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
		# Added later
		batch_size = tf.shape(self.state)[0]

		self.target = tf.placeholder(shape=[None], dtype=tf.float32)
		#self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

		# Create fully connected layers such that one is the others's input
		self.fc1 = tf.contrib.layers.fully_connected(self.state, 20, activation_fn=tf.nn.relu)
		self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu)

		# Output layer will have linear activation with "activation_fn=None"
		# Name changed to predictions
		self.state_value = tf.contrib.layers.fully_connected(self.fc2, 1, activation_fn=None)

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
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.000005)
		self.train_op = self.optimizer.minimize(self.loss)

		# Initiate global variables
		init = tf.global_variables_initializer()

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
 	def __init__(self):
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

		self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc3 = tf.contrib.layers.fully_connected(self.fc2, action_space_size, activation_fn=None,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		self.predictions = tf.contrib.layers.softmax(self.fc3)
		#self.predictions = tf.nn.softmax(self.fc3)
		#print("self.predictions: ",self.predictions.shape)
		#print("self.actions: ", self.actions_pl)
		#print("batch size: ",batch_size)
		
		#self.action_predictions = tf.gather(self.predictions, self.actions_pl)
		#self.action_predictions = tf.gather(self.predictions, self.actions_pl)

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

		self.optimizer = tf.train.AdamOptimizer(0.00001)
		self.train_op = self.optimizer.minimize(self.objective)

		init = tf.global_variables_initializer()

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

def pendulum(sess, env, actor, critic, num_episodes, max_time_per_episode, discount_factor=0.99, epsilon=0.1):

	# Keeps track of useful statistics
	stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	
	for i_episode in range(num_episodes):
		# Print out which episode we're on, useful for debugging.
		# Also print reward for last episode
		print("\r Episode {}/{} ({})".format(
             i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))

		state = env.reset()
		for t in itertools.count():

			# get an action from policy : actor gives the action
			action_predicted, action_probs = actor.predict(sess, [state])
			#print("action: ",action_predicted)
			action = action_index[action_predicted]
			# take the action given by the actor
			next_state, reward, done, _ = env.step([action])

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

			# Stop if the max_time_per_episode is reached
			if (stats.episode_lengths[i_episode] + 1) == max_time_per_episode:
				break

			# update actor: policy improvement
			action_ind = inv_action_index[action]
			actor.update(sess, [state], [action_ind], td_error)

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
	env = PendulumEnv()
	actor = ActorNetwork()
	critic = CriticNetwork()

	sess = tf.Session()
	start_time = time.time()
	print("Starting at : ", datetime.datetime.now().strftime("%H:%M:%S"))
	sess.run(tf.global_variables_initializer())

	stats = pendulum(sess, env, actor, critic, 200, 1000)
	print("--- %s seconds ---" % (time.time() - start_time))
	print("Ended at : ", datetime.datetime.now().strftime("%H:%M:%S"))
	plot_episode_stats(stats)

	for _ in range(200):
		state = env.reset()
		for _ in range(1000):
			env.render()
			action_predicted, action_probs = actor.predict(sess, [state])
			action = action_index[action_predicted]
			state, _, done, _ = env.step([action])

			if done:
				break