import sys
import numpy as np
from collections import defaultdict, namedtuple
import itertools

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def make_epsilon_greedy_policy(Q, epsilon, nA):
	"""
	Creates an epsilon-greedy policy based on a given Q-function and epsilon.

	Args:
		Q: A dictionary that maps from state -> action-values.
		Each value is a numpy array of length nA (see below)
		epsilon: The probability to select a random action . float between 0 and 1.
		nA: Number of actions in the environment.

	Returns:
		A function that takes the observation as an argument and returns
		the probabilities for each action in the form of a numpy array of length nA. 
	"""
  
	def policy_fn(observation):
		# Implement this!
		# For creating an epsilon greedy policy, we apply the formula from slides

		# We create an array of nA for the state that we observed, it will be probabilities 1
		# then we apply the formula step by step
		probabilities = np.ones(nA) * epsilon / nA

		# We make the best action's probability 1 - epsilon, which will be greedy action
		best_action = np.argmax(Q[observation])
		probabilities[best_action] += (1.0 - epsilon)

		# Now at the end, we return an array of probabilities for this state, with probability of 
		# the greedy action with 1 - epsilon
		return probabilities
	return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
	"""
	Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
	while following an epsilon-greedy policy

	Args:
		env: OpenAI environment.
		num_episodes: Number of episodes to run for.
		discount_factor: Lambda time discount factor.
		alpha: TD learning rate.
		epsilon: Chance the sample a random action. Float betwen 0 and 1.

	Returns:
		A tuple (Q, episode_lengths).
		Q is the optimal action-value function, a dictionary mapping state -> action values.
		stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
	"""
  
	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# Keeps track of useful statistics
	stats = EpisodeStats(
	episode_lengths=np.zeros(num_episodes),
	episode_rewards=np.zeros(num_episodes))    

	# The policy we're following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		# Print out which episode we're on, useful for debugging.
		if (i_episode + 1) % 100 == 0:
		# print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
		# We changed this print() line slightly, due to an error at end="" part
			print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes))
			sys.stdout.flush()

		# Implement this!

		# Draw a random state from the environment for the beginning
		state = env.reset()

		e_length = 0

		# For each state of the episode, repeat
		for s in itertools.count():
			# We need to choose an action from epsilon greedy policy. For foing that, 
			# we get all action probabilities of the current state.
			action_probs = policy(state)

			# Now we choose randomly from these actions - it will be epsilon greedy because 
			# of the setting in policy_fn()
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

			# Take the the action, and get the relevant information from the environment
			# about the next state, and reward, and terminal case check
			next_state, reward, done, _ = env.step(action)

			# Store relevant episode information
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = s
			
			# Now we consider that we are in the next state, and we look further from the view of the 
			# greedy policy pi, that is we choose the next next action from the next state
			next_next_action = np.argmax(Q[next_state])    

			# Now we can update action value function towards the value of alternative action of 
			# the greedy policy pi
			best_next_action = np.argmax(Q[next_state])    
			td_target = reward + discount_factor * Q[next_state][best_next_action]
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta

			# We should check if we reach end of the episode
			if done:
				break

			# Otherwise keep playing
			state = next_state

	return Q, stats