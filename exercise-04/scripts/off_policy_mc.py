from collections import defaultdict

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
    pass
    # Implement this!
  return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
  """
  Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
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
  
  # Implement this!
      
  return Q, target_policy