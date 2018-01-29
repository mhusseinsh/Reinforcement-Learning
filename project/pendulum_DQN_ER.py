from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
import itertools

from config_space import get_config_space


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

max_time_per_episode = 200
initial_epsilon = 1.
final_epsilon = 0.1
decay_until = max_time_per_episode * 0.1
decay_factor = 1 - ((initial_epsilon - final_epsilon) / decay_until)
epsilon = initial_epsilon

class NeuralNetwork():
	"""
	Neural Network class based on TensorFlow.
	"""
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

		# Create placeholders for input, output and actions of the network 
		self.state_q = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)
		# Added later
		batch_size = tf.shape(self.state_q)[0]

		self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
		self.action_q = tf.placeholder(shape=[None], dtype=tf.int32)

		# Create fully connected layers such that one is the others's input
		self.fc1 = tf.contrib.layers.fully_connected(self.state_q, self.num_fc_units_1, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self.num_fc_units_2, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc3 = tf.contrib.layers.fully_connected(self.fc2, self.num_fc_units_3, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc4 = tf.contrib.layers.fully_connected(self.fc3, self.num_fc_units_4, activation_fn=tf.nn.relu,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))
		self.fc5 = tf.contrib.layers.fully_connected(self.fc4, self.action_space_size, activation_fn=None,
		  weights_initializer=tf.random_uniform_initializer(0, 0.5))

		# Output layer will have linear activation with "activation_fn=None"
		# Name changed to predictions
		self.action_values = tf.contrib.layers.fully_connected(self.fc5, action_space_size, activation_fn=None)

		# Prepare loss by using mean squared error
		# [all_actions] wrong: this should be the target(s)
		# clarify what is y: target or prediction
		# when predicting check which action was taken, then choose the index of the action for the prediction

		# Get the predictions for the chosen actions only
		#gather_indices = tf.range(batch_size) * tf.shape(self.prediction)[1] + self.actions
		#self.action_prediction = tf.gather(tf.reshape(self.prediction, [-1]), gather_indices)

		gather_indices = tf.range(batch_size) * tf.shape(self.action_values)[1] + self.action_q

		#print("indices: ",gather_indices)

		self.action_value = tf.gather(tf.reshape(self.action_values,[-1]), gather_indices)

		# Calcualte the loss
		# self.losses = tf.losses.mean_squared_error([all_actions], self.y, reduction =tf.losses.Reduction.NONE)
		self.losses = tf.squared_difference(self.target_q, self.action_value)
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
		predictions = sess.run(self.action_values, { self.state_q: states})

		# will return n action predictions
		return predictions

	def update(self, sess, states, actions, targets):
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
		feed_dict = { self.state_q: states, self.action_q: actions, self.target_q: targets }

		# Compute loss and update
		sess.run([self.train_op, self.loss],feed_dict)


class TargetNetwork(NeuralNetwork):
	"""
	Slowly updated target network. Tau indicates the speed of adjustment. If 1,
	it is always set to the values of its associate.
	"""
	def __init__(self, tau=0.001):
		NeuralNetwork.__init__(self, NeuralNetwork.learning_rate,
		NeuralNetwork.action_space_size,
		NeuralNetwork.num_fc_units_1,
		NeuralNetwork.num_fc_units_2,
		NeuralNetwork.num_fc_units_3,
		NeuralNetwork.num_fc_units_4)
		
		self.tau = tau
		self._associate = self._register_associate()

	def _register_associate(self):
		tf_vars = tf.trainable_variables()
		total_vars = len(tf_vars)
		op_holder = []

		for idx, var in enumerate(tf_vars[0:total_vars//2]):
			op_holder.append(tf_vars[idx+total_vars//2].assign((var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))

		return op_holder
		
	def update(self, sess):
		for op in self._associate:
			sess.run(op)

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

	def next_batch(self, batch_size):
		batch_indices = np.random.choice(len(self._data.states), batch_size)
		batch_states = np.array([self._data.states[i] for i in batch_indices])
		batch_actions = np.array([self._data.actions[i] for i in batch_indices])
		batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
		batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
		batch_dones = np.array([self._data.dones[i] for i in batch_indices])
		return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

def choose_action(estimator, epsilon):

	choice = np.random.choice(["greedy", "random"], p=[1-epsilon, epsilon])

	if choice is "greedy":
		q_values = estimator.predict(sess, [observation])
		best_action = np.argmax(q_values)
	else:
		best_action = np.random.choice(action_indices)
	return best_action


def decay_epsilon(epsilon, step):

	if step >= decay_until:
		epsilon = epsilon * decay_factor
	else:
		epsilon = final_epsilon
	return epsilon

def pendulum(sess,env, approx, epsilon, num_episodes, discount_factor, 
				use_experience_replay, batch_size, target):

	# Create the experiences memory
	replay_memory = ReplayBuffer()

	# The policy we're following
	#policy = make_epsilon_greedy_policy(approx, epsilon, action_space_size)

	# Keeps track of useful statistics
	stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	tn = target

	for i_episode in range(num_episodes):
		# Print out which episode we're on, useful for debugging.
		# Also print reward for last episode
		print("Episode {}/{} ({})".format(i_episode + 1, 
			num_episodes, stats.episode_rewards[i_episode - 1]),end='\r')
		sys.stdout.flush()

		state = env.reset()
		for t in itertools.count():

			#giving an index for one action
			action_ind = choose_action(approx, epsilon)
			
			action = action_index[action_ind]
			next_state, reward, done, _ = env.step([action])

			epsilon = decay_epsilon(epsilon, t)
			

			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# Stop if the max_time_per_episode is reached
			if (stats.episode_lengths[i_episode] + 1) == max_time_per_episode:
				break

			if(use_experience_replay):
				# Save transition to replay memory
				replay_memory.add_transition(state, action, next_state, reward, done)

				# Sample a minibatch from the replay memory
				batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = replay_memory.next_batch(batch_size)
				
				q_values_next_states_t = tn.predict(sess, batch_next_states)

				batch_targets = batch_rewards + np.invert(batch_dones).astype(np.float32) * discount_factor * np.amax(q_values_next_states_t, axis=1)

				target.update(sess)
				batch_action_ind = []
				for i in batch_actions:
					batch_action_ind.append(inv_action_index.get(i))

				approx.update(sess, batch_states, batch_action_ind, batch_targets)

			else:
				# Now we consider that we are in the next state, and we look further from the view of the 
				# greedy policy pi, that is we choose the next next action from the next state
				q_values_next = approx.predict(sess, [next_state])
				#print(q_values_next)

				# Pick the best action greedily
				best_action = np.argmax(q_values_next)
				if done:
					td_target = reward
				else:
					td_target = reward + discount_factor * q_values_next[0][best_action]

				# update actor: policy improvement
				action_ind = inv_action_index[action]

				approx.update(sess, [state], [action_ind], [td_target])

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
	reward_func= True
	env = PendulumEnv(reward_func)


	stats = []

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

		approx = NeuralNetwork(config_dict.get("config_dict{0}".format(i)).get('learning_rate'),
							action_space_size,
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_1'),
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_2'),
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_3'),
							config_dict.get("config_dict{0}".format(i)).get('num_fc_units_4'))
		target = TargetNetwork()

		sess = tf.Session()
		start_time = time.time()
		print("Starting at : ", datetime.datetime.now().strftime("%H:%M:%S"))
		sess.run(tf.global_variables_initializer())

		stats.append(pendulum(sess, env, actor, critic_v, critic_q, 
			config_dict.get("config_dict{0}".format(i)).get('num_episodes'), 
			200,
			config_dict.get("config_dict{0}".format(i)).get('discount_factor')))
			#,
			#use_experience_replay = True,
			#config_dict.get("config_dict{0}".format(i)).get('batch_size'),
			#target = target))


		print("--- %s seconds ---" % (time.time() - start_time))

		print("Ended at : ", datetime.datetime.now().strftime("%H:%M:%S"))
		plot_episode_stats(stats[i])

	plot_episode_stats(stats)