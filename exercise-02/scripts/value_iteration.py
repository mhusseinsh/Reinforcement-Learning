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
  

  V = np.zeros(env.nS)
  policy = np.zeros([env.nS, env.nA])
  
  # Implement!
  return policy, V