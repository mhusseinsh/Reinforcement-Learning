from collections import defaultdict
import numpy as np

def mc_evaluation(policy, env, num_episodes, discount_factor=1.0):
  """
  First-visit MC Policy Evaluation. Calculates the value function
  for a given policy using sampling.

  Args:
    policy: A function that maps an observation to action probabilities.
    env: OpenAI gym environment.
    num_episodes: Nubmer of episodes to sample.
    discount_factor: Lambda discount factor.

  Returns:
    A dictionary that maps from state to value.
    The state is a tuple -- containing the players current sum, the dealer's one showing card (1-10 where 1 is ace) 
    and whether or not the player holds a usable ace (0 or 1) -- and the value is a float.
  """

  # Keeps track of sum and count of returns for each state
  # to calculate an average.
  returns_sum = defaultdict(float)
  returns_count = defaultdict(float)

  # The final value function
  V = defaultdict(float)

  # Implement this!

  # Start iterationg on episodes
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
            action = policy(state)
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

        # Now we completed this episode, and we will compute G
        
        # This is the "first time visit" approach, hence we will check it
        first_visit = 0

        # We need all states that we visited from the episode
        visited_states = [x[0] for x in episode]
     
        # Compute value of each state by looking further from that state
        for state in visited_states:
            # We need to find the first time that we visited this state
            for index, e_state in enumerate(episode):
            	if e_state[0] == state:
            		first_visit = index
            # Initiate the G of this state
            G = 0
            # Now we will sum over all the rewards since the first visit before
            # we compute the average 
            for i, read_episode in enumerate(episode[first_visit:]):
            	G += read_episode[2] * discount_factor ** i
            # Compute the average value of this state, and store it in the dictionary
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

  return V