import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
import itertools

if "../../lib/envs" not in sys.path:
  sys.path.append("../../lib/envs")
from mountain_car import MountainCarEnv

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

# An array to use for action indexes
all_actions = [0, 1, 2]

class NeuralNetwork():
	"""
	Neural Network class based on TensorFlow.
	"""
	def __init__(self):
		self._build_model()

	def _build_model(self):
		"""
		Creates a neural network, e.g. with two
		hidden fully connected layers and 20 neurons each). The output layer
		has #A neurons, where #A is the number of actions and has linear activation.
		Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with
		a learning rate of 0.0005). For initialization, you can simply use a uniform
		distribution (-0.5, 0.5), or something different.
		"""
		# TODO: Implement this!

		self.x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
		self.y = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32)

		self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

		self.fc1 = tf.contrib.layers.fully_connected(self.x, 20, activation_fn=tf.nn.relu)
		self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu)

		# Output layer will have linear activation with "activation_fn=None"
		self.y = tf.contrib.layers.fully_connected(self.fc2, env.action_space.n, activation_fn=None)

		self.losses = tf.losses.mean_squared_error([all_actions], self.y, reduction =tf.losses.Reduction.NONE)
		self.loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
		self.train_op = self.optimizer.minimize(self.loss)

		init = tf.global_variables_initializer()

	def predict(self, sess, states):
		"""
		Args:
			sess: TensorFlow session
			states: array of states for which we want to predict the actions.
		Returns:
			The prediction of the output tensor.
		"""
		# TODO: Implement this!
		prediction = sess.run(self.y, { self.x: states})
		return prediction

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
			# TODO: Implement this!
		feed_dict = { self.x: states, self.actions: actions, self.y: targets }
		_, loss = sess.run([self.train_op, self.loss],feed_dict)

		return loss

class TargetNetwork(NeuralNetwork):
	"""
	Slowly updated target network. Tau indicates the speed of adjustment. If 1,
	it is always set to the values of its associate.
	"""
	def __init__(self, tau=0.001):
		NeuralNetwork.__init__(self)
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

def make_epsilon_greedy_policy(estimator, epsilon, nA):
	"""
	Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

	Args:
		estimator: An estimator that returns q values for a given state
		epsilon: The probability to select a random action . float between 0 and 1.
		nA: Number of actions in the environment.

	Returns:
		A function that takes the observation as an argument and returns
		the probabilities for each action in the form of a numpy array of length nA.

	"""
	def policy_fn(sess, observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		q_values = estimator.predict(sess, [observation])
		best_action = np.argmax(q_values)
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn

def q_learning(sess,env, approx, num_episodes, max_time_per_episode, discount_factor=0.99, epsilon=0.1, use_experience_replay=False, batch_size=128, target=None):
	"""
	Q-Learning algorithm for off-policy TD control using Function Approximation.
	Finds the optimal greedy policy while following an epsilon-greedy policy.
	Implements the options of online learning or using experience replay and also
	target calculation by target networks, depending on the flags. You can reuse
	your Q-learning implementation of the last exercise.

	Args:
		env: OpenAI environment.
		approx: Action-Value function estimator
		num_episodes: Number of episodes to run for.
		max_time_per_episode: maximum number of time steps before episode is terminated
		discount_factor: gamma, discount factor of future rewards.
		epsilon: Chance to sample a random action. Float betwen 0 and 1.
		use_experience_replay: Indicator if experience replay should be used.
		batch_size: Number of samples per batch.
		target: Slowly updated target network to calculate the targets. Ignored if None.

	Returns:
	An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
	"""

	# Create the experiences memory
	replay_memory = ReplayBuffer()

	# The policy we're following
	policy = make_epsilon_greedy_policy(approx, epsilon, 3)

	# if(use_experience_replay):
	# 	for i in range(5000):
	# 		state = env.reset()
	# 		action_probs = policy(sess, state)
	# 		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
	# 		next_state, reward, done, _ = env.step(all_actions[action])
	# 		# Save transition to replay memory
	# 		replay_memory.add_transition(state, action, next_state, reward, done)

	# Keeps track of useful statistics
	stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

	for i_episode in range(num_episodes):
	
		# Print out which episode we're on, useful for debugging.
		# Also print reward for last episode
		last_reward = stats.episode_rewards[i_episode - 1]
		print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward))
		sys.stdout.flush()
		# TODO: Implement this!
		state = env.reset()
		for t in itertools.count():
			# Take an epsilon greedy step
			action_probs = policy(sess, state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			next_state, reward, done, _ = env.step(all_actions[action])

			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# Stop if the max_time_per_episode is reached
			if stats.episode_lengths[i_episode] == max_time_per_episode:
				break

			if(use_experience_replay):
				# Save transition to replay memory
				replay_memory.add_transition(state, action, next_state, reward, done)

				# Sample a minibatch from the replay memory
				batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = replay_memory.next_batch(batch_size)

				# Get action values for the state that we are in
				#q_values_next = approx.predict(sess, [next_state])
				q_values_states = target.predict(sess, batch_states)

				# Get the action values of the batch_states from the target network
				q_values_next_states = target.predict(sess, batch_next_states)

				# Find the best actions of each state of the batch
				#best_actions = np.argmax(q_values_next_target,1)
				# Compute Q-target 
				#targets_batch = batch_rewards + discount_factor * q_values_next_target[np.arange(batch_size), best_actions]
				#q_values_next_target[0][best_actions] = targets_batch

				for i in batch_states:
					q_next = q_values_next_states[i]
					max_action = np.argmax(q_next)
					if done:
						target = batch_rewards[i]
					else:
						target = batch_rewards[i]+discount_factor * q_values_next_states[max_action]

					q_values_states[i][batch_actions[i]]=target

				loss = approx.update(sess, batch_states, batch_actions, q_values_states)
				#	targets[i] = batch_rewards[i]+discount_factor*q_values_next_target[i, best_actions]
				#	q_values_next_target[action] = target
				# Compute the loss between Q-target and Q-network
				#loss = approx.update(sess, batch_states, batch_actions, q_values_next_target)

			else:
			  	# Now we consider that we are in the next state, and we look further from the view of the 
				# greedy policy pi, that is we choose the next next action from the next state
				q_values_next = approx.predict(sess, [next_state])
				#q_values = approx.predict(sess, [state])
				best_action = np.argmax(q_values_next[0])

				
				# Now we can update action value function towards the value of alternative action of 
				# the greedy policy pi  
				if done:
					td_target = reward
				else:
					td_target = reward + discount_factor * q_values_next[0][best_action]
				q_values_next[0][best_action]=td_target
				#q_values[0][best_action]=td_target

				# Compute the loss between greedy policy and epsilon greedy policy
				loss = approx.update(sess, [state], [action], q_values_next)
				#loss = approx.update(sess, [state], [action], q_values)

			#if done:
			#	break

			state = next_state
	
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
	env = MountainCarEnv()
	approx = NeuralNetwork()
	target = TargetNetwork()

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# Choose one.
	stats = q_learning(sess, env, approx, 3000, 1000)
	#stats = q_learning(sess, env, approx, 1000, 1000, use_experience_replay=True, batch_size=128, target=target)
	plot_episode_stats(stats)

	for _ in range(100):
		state = env.reset()
		for _ in range(1000):
			env.render()
			state,_,done,_ = env.step(np.argmax(approx.predict(sess, [state])))
			if done:
				break