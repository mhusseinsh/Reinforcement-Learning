import numpy as np

def value_iteration(env, theta=0.0001, discount_factor=1.0):
	"""
	Value Iteration Algorithm.

	Args:
	env: OpenAI environment. env.P represents the transition probabilities of the environment.
	theta: Stopping threshold. If the value of all states changes less than theta
		in one iteration we are done.
	discount_factor: lambda time discount factor.

	Returns:
	A tuple (policy, V) of the optimal policy and the optimal value function.        
	"""
	# We are storing the useful information here for not to read them from the environment each time
	num_states = env.nS
	num_actions = env.nA

	# We start computation with zeros for all state values
	V = np.zeros(num_states)

	converged = True
	# Loop until we converge to the optimal value function
	while converged:
		# We need a control for getting out of the while loop
		# For doing this, we will check the change of the state values
		# If the change is smaller than the stopping threshold theta, we will stop
		change = 0
		# For each state compute state value function
		for s in range(num_states):
			# For each state, we need to check the action values of the action that we can take 
			# to choose the action which gives the maximum value
			# We write these in an array called actions_s which states action values of that state
			actions_s = np.zeros(num_actions)
			for a in range(num_actions):
				# For each action of the state s, we need the transition probabilities, 
				# rewards, and next state information
				for transition_prob, next_state, reward, done in env.P[s][a]:
					# We are looping over all transition states of the current action,
					# and computing action values
					actions_s[a] += reward + discount_factor * transition_prob * V[next_state]
			# Now for each state we have the action values. According to the formula, we need to choose the 
			# action which gives us the maximum value, so we get the maxiumum action value
			val_max = actions_s.max()
			# We need to check if our value changed. We are using absolute value because 
			# the difference is negative due to negative rewards (from gridworld.py)
			change = max(change, np.abs(val_max - V[s]))
			# After checking the difference, now we can update our values
			V[s] = val_max        
		# Check if we reached our stopping treshold  
		if change < theta:
			converged = False
	
	# Now we can compute the optimal policy after having the optimal value function
	# Instead of having zero value for each state, now we have the optimal values, 
	# so when we produce the policy it will yield the optimal one.

	# According to the expected policy from the tests, we have 1.0 for the action to be taken,
	# and zero for the others
	optimal_policy = np.zeros([num_states, num_actions])
	for s in range(num_states):
		# We are computing the maximum action values again, by using our optimal value function 
		actions_s = np.zeros(num_actions)
		for a in range(num_actions):
			for transition_prob, next_state, reward, done in env.P[s][a]:
				actions_s[a] += reward + discount_factor * transition_prob * V[next_state]
		# This time we need to know which action is giving us the highest value not the value 
		# of the action, for doing this, we get the indice
		# which tells us about the action
		action_max = np.argmax(actions_s)
		# We set the action corresponding to the optimal policy to 1.0 
		optimal_policy[s, action_max] = 1.0
	
	return optimal_policy, V