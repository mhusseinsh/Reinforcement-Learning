import numpy as np

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
	"""
	Evaluate a policy given an environment and a full description of the environment's dynamics.
	
	Args:
	policy: [S, A] shaped matrix representing the policy.
	env: OpenAI env. env.P represents the transition probabilities of the environment.
		env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
	theta: We stop evaluation once our value function change is less than theta for all states.
	discount_factor: gamma discount factor.
	
	Returns:
	Vector of length env.nS representing the value function.
	"""
	num_states = env.nS
	num_actions = env.nA

	# Start with a random (all 0) value function
	V = np.zeros( num_states)
	#loop until policy evalutaion is completed
	completed = True
	while completed:
		# TODO: Implement!
		#the variable difference is used for comparing it with theta 
		#it represents the change in the state-value function
		difference = 0
		
		#for each state in the environment we will compute value function with respect to the current policy
		for state in range(num_states): 
			value = 0 
			#ENUMERATE: a is the counter of array and the elements are the
			#probabilities inside each action: for example [ 0.25,  0.25,  0.25,  0.25]
			for action, action_probabilities in enumerate(policy[state]):
				#the probability of each state and action is defined as P[s][UP], P[s][RIGHT], P[s][DOWN], P[s][LEFT] 
				#therefore we need to go through all of them and use them in the state-value function
				for transition_prob, next_state, reward, done in env.P[state][action]: 
					#from the environment we get this list of trantition tuple
					#Formula slide 8:
					#state value function += action_probabilities * (reward + discount_factor * probability * Value function of the next_state)	
					value += action_probabilities * (reward +  discount_factor * transition_prob  * V[next_state])
			#now we can compare the difference between two iterations
			#if the value improvement has stopped, we can stop iterating too because the function converged
			difference = max(difference, np.abs(value - V[state]))
			#now update the state value function with the new value for that state
			V[state] = value
		
		#We stop evaluation once our value function change is less than theta for all states.
		#otherwise we continue evaluating
		if difference < theta:
			completed = False
	#return the array of the value function of each state      
	return V

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
	"""
	Policy Improvement Algorithm. Iteratively evaluates and improves a policy
	until an optimal policy is found.

	Args:
	env: The OpenAI envrionment.
	policy_eval_fn: Policy Evaluation function that takes 3 arguments:
		policy, env, discount_factor.
	discount_factor: Lambda discount factor.

	Returns:
	A tuple (policy, V). 
	policy is the optimal policy, a matrix of shape [S, A] where each state s
	contains a valid probability distribution over actions.
	V is the value function for the optimal policy.
	"""
	# We are storing the useful information here for not to read them from the environment each time
	num_states = env.nS
	num_actions = env.nA

	# Start with a random policy
	#we create the random policy simply; for all states, and for all actions, have an equal probability
	policy = np.ones([num_states, num_actions]) / num_actions

	#loop until policy iteration converges
	converged = True
	
	while converged:
		# Implement this!
		# value function of the current policy
		# we need to evaluate each policy that we computed
		V = policy_eval_fn(policy, env, discount_factor)

		#we need to stop iterating if our policy iteration converged.
		policy_converged = True

		# for all states, 
		for state in range(num_states):
			#We need to act greedily for imrpoving the policy
			#therefore we need a variable which reprsents the action which gives the 
			#highest value with the currect policy
			max_action_policy = np.argmax(policy[state])

			#For each state, we need to check the action values of the action that we can take
			#to choose the action which gives the maximum value
			#we write these in an array called action_s which states action values of that state
			#by doing this, we will check Bellman Optimality Equation
			action_s = np.zeros(num_actions)
			for action in range(num_actions):
				# For each action of the current state, we need the transition probabilities, 
				# rewards, and next state information

				for transition_prob, next_state, reward, done in env.P[state][action]:
					#from lecture MDP slide 32, we got the equation of action-vaue function
					#We are looping over all transition states of the current action
					#and computing action values
					action_s[action] +=  (reward + discount_factor * transition_prob * V[next_state])

			#According to the formula we need to get the maximum action-value function from this 
			#state and action, so we need to know which action is that
			action_max  = np.argmax(action_s)

			# Verify if we need to continue through the loop. check if the max_action_policy of the current policy
			# is equal to the newly computed action_max. 
			if max_action_policy != action_max :
				#if the policy has changed for this state, that means it hasn't converged yet. so we say
				#we need to keep computing. once the policy converged, max_action_policy will always be equal
				#to action_max, and we will say the policy has converged.
				policy_converged = False

			#we have the action_value function. Now we need to save which action gets the maximum in each state
			#therefore we need to write a 1 on the action that got the maximum value.
			policy[state] = np.zeros(num_actions)
			policy[state][action_max] = 1.0

		#if the iteration is over, then we can get out of the loop because the policy is not changing anymore.
		if policy_converged:
			return policy, V