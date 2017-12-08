from collections import defaultdict
import numpy as np

def create_random_policy(nA):
	"""
	Creates a random policy function.

	Args:
		nA: Number of actions in the environment.

	Returns:
		A function that takes an observation as input and returns a vector
		of action probabilities
	"""
	A = np.ones(nA, dtype=float) / nA
	def policy_fn(observation):
		return A
	return policy_fn

def create_greedy_policy(Q):
	"""
	Creates a greedy policy based on Q values.

	Args:
		Q: A dictionary that maps from state -> action values
	  
	Returns:
		A function that takes an observation as input and returns a vector
		of action probabilities.
	"""

	def policy_fn(observation):
		# Implement this!
		A = np.zeros_like(Q[observation], dtype=float)
		best_action = np.argmax(Q[observation])
		A[best_action] = 1.0
		return A
	return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
	"""
	Monte Carlo Control Off-Policy Control using Importance Sampling.
	Finds an optimal greedy policy.

	Args:
		env: OpenAI gym environment.
		num_episodes: Nubmer of episodes to sample.
		behavior_policy: The behavior to follow while generating episodes.
			A function that given an observation returns a vector of probabilities for each action.
		discount_factor: Lambda discount factor.

	Returns:
		A tuple (Q, policy).
		Q is a dictionary mapping state -> action values.
		policy is a function that takes an observation as an argument and returns
		action probabilities. This is the optimal greedy policy.
	"""

	# The final action-value function.
	# A dictionary that maps state -> action values
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# Our greedily policy we want to learn
	target_policy = create_greedy_policy(Q)

	# Keeps track of sum and count of returns for each state
	# to calculate an average.
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# Implement this!
	for e in range(num_episodes):

		# In each episode, we will follow the sample policy given. This sample policy will take the state and and produce an
		# action to take. When we take this action, this time we will receive next state, reward, terminal state check and an 
		# empty dictionary (according to the def _step(self,action) of BlackJack environment)

		# So we need to keep track of these, we first create an array for an episode
		episode = []
		i = 0
		# Then we set the first state: env.reset() gives us sum of players cards, the first (and only open) card of the dealer
		# and a binary return for whether the player could use an ace as 10 or 1
		state = env.reset()
		# There are 200 possible states, so we thought we can limit the actions in one episode to 200. It can take less or more.
		while (i <= 200):
			# After folling the given sample policy we find the action that we should take
			probabilities = behavior_policy(state)
			action=np.random.choice(np.arange(len(probabilities)), p=probabilities)
			# We take the action: stick (0) or twist (1). We get the relevant information from the environment.
			next_state, reward, done, _ = env.step(action)
			# We need to store all information that we got in an episode for further usage
			episode.append((state, action, reward))
			# Stop playing if we won
			if done == True:
				break
			# Otherwise keep playing
			state = next_state
			i += 1 

		# Now we completed this episode, and we will compute the rest
		
		# This is the "first time visit" approach, hence we will check it
		first_visit = 0

		# We start reading the each element of episode: (state, action, reward)
		for e_episode in episode:
			# Store the useful information
			e_state = e_episode[0]
			e_action = e_episode[1]

			# State-Action pair will be used for counting
			sa_pair=(e_episode[0],e_episode[1])

			# Find the first visit of the state
			first_visit = next(i for i, x in enumerate(episode)
				if x[0]==e_state and x[1]==e_action)

			# Compute sum of the rewards
			G=sum([x[2]*(discount_factor**i) for i, x in enumerate(episode[first_visit:])])

			# Initialize variables for importance sampling
			prob_greedy = 1.0
			prob_behavior = 1.0

			# Now we will look all the state-actions in front of us, and then multiply the 
			# action probabilities for each each state-action pair.
			for i, read_episode in enumerate(episode[first_visit:]):
				# The state and the action taken in the episode
				episode_state = read_episode[0]
				episode_action = read_episode[1]

				# Computation for importance sampling
				prob_greedy *= target_policy(episode_state)[episode_action]
				prob_behavior *= behavior_policy(episode_state)[episode_action]

			# Update G according to importance sampling
			G_pi_mu = (prob_greedy/prob_behavior) * G

			returns_sum[sa_pair] += G_pi_mu

			# Count how many times this state-action visited
			returns_count[sa_pair] += 1.0

			# Update the action value function based on the running mean and the weighted sum o f rewards
			Q[e_state][e_action] = returns_sum[sa_pair] / returns_count[sa_pair]

	return Q, target_policy