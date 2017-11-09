from collections import defaultdict

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

  return V