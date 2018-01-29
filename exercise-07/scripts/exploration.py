import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats as sci
import math

"""
Implement the different exploration strategies (do not forget the schedule in
line 83).

mab is a MAB (MultiArmedBandit) object as defined below
epsilon is a scalar, which influences the amount of random actions
schedule is a callable decaying epsilon

You can get the approximated Q-values via mab.bandit_q_values and the different
counters for the bandits via mab.bandit_counters.
"""
def epsilon_greedy(mab, epsilon):
  # Implement this!
  # Create an array where all action probs are equal to epsilon/num_actions
  probabilities = np.ones(mab.no_actions) * epsilon / mab.no_actions

  # We make the best action's probability 1 - epsilon, which will be greedy action
  best_action = 0

  probabilities[best_action] += (1.0 - epsilon)

  # Which arm(bandit)?
  action_index = np.random.choice(mab.no_actions,1, p=probabilities)

  return action_index[0]

"""
	take a rand sample random.uniforsm

	if rand<epsilom return random(mab)
	else return np.argmax(mab.q.bandit_q_values)
"""

def decaying_epsilon_greedy(mab, epsilon, schedule):
  # Implement this!

  decayed_epsilon = schedule(mab, epsilon)

  # Which arm(bandit)?
  return epsilon_greedy(mab, decayed_epsilon)

def random(mab):
  # Implement this!
  # Which arm(bandit)?
  action_index = np.random.choice(mab.no_actions,1)
  
  return action_index[0]

def ucb1(mab):
  # Implement this!

  """
	return np.argmax(mab.banditq val + sqrt(2*log(mab.step count + 1)/(mab.bandit_counter + 1)))
  """
  actions = []
  for i in range(mab.no_actions):
    if mab.bandit_counters[i] == 0:
        # if the action i is never tried before, give it a high upper bound so that maximize its q-value
        actions.append(mab.bandit_q_values[i] + np.sqrt(2 * np.log(mab.step_counter + 1) * np.reciprocal(mab.bandit_counters[i] + 1)))
    else:
        actions.append(mab.bandit_q_values[i] + np.sqrt(2 * np.log(mab.step_counter) * np.reciprocal(mab.bandit_counters[i])))
  # Which arm(bandit)?
  return np.argmax(actions)

class Bandit:
  def __init__(self, bias, q_value=0, counter=0):
    self.bias = bias
    self.q_value = q_value
    self.counter = counter

  def pull(self):

    # Increment counter by 1 - This is a specific counter for this Bandit
    self.counter += 1

    # Get the reward
    reward = np.clip(self.bias + np.random.uniform(), 0, 1)

    # Compute the q_value 
    self.q_value = self.q_value + 1/self.counter * (reward - self.q_value)

    return reward

class MAB:
  def __init__(self, best_action, *bandits):
    self.bandits = bandits

    # number of bandits is 10
    self._no_actions = len(bandits)
    self.step_counter = 0

    # best_actions initialized as 0
    self.best_action = best_action

  def pull(self, action):

    # Increment counter by one - This is a general counter for MAB
    self.step_counter += 1

    # Call pull() of specified bandit
    return self.bandits[action].pull()

  def run(self, no_rounds, exploration_strategy, **strategy_parameters):
    regrets = []
    rewards = []
    actions = []
    print("In run .................", exploration_strategy.__name__)
    regret = 0
    for i in range(no_rounds):

      # Returns an index as an action
      action = exploration_strategy(self, **strategy_parameters)

      #send the action(bandit) index to pull, and get the reward
      reward = self.pull(action)

      # Set the best_action_reward if the action was the best action, else the best_action_reward = 0.7
      best_action_reward = reward if action == best_action(self) else 0.7

      # If the best action was taken, then regret is 0
      regret += best_action_reward - reward

      regrets.append(regret)
      rewards.append(reward)
      actions.append(action)

    return regrets, rewards, actions

  @property
  def bandit_counters(self):
    # Take the specific counter of each Bandit
    return np.array([bandit.counter for bandit in self.bandits])

  @property
  def bandit_q_values(self):
    # Take the specific q_value of each Bandit
    # Expected return of each Bandit(action)
    return np.array([bandit.q_value for bandit in self.bandits])

  @property
  def no_actions(self):
    return self._no_actions

def plot(regrets):
  for strategy, regret in regrets.items():
    total_regret = np.cumsum(regret)
    plt.ylabel('Total Regret')
    plt.plot(np.arange(len(total_regret)), total_regret, label=strategy)
  plt.legend()
  plt.savefig('regret.png')

if __name__ == '__main__':
  
  no_rounds = 1000000
  eps = []
  def schedule(mab, epsilon):
    # Implement this!
    """
    simple linear decay
    return epsilon - 1e-6*mab.step_counter
    """
    # Simple exponential decay 
    decaying_factor = 100000

    return epsilon * np.exp(-((mab.step_counter + 1)/decaying_factor))

  epsilon = 0.5

  strategies = {
    #epsilon_greedy: {'epsilon': epsilon},
    #decaying_epsilon_greedy: {'epsilon': epsilon, 'schedule': schedule},
    #random: {},
    ucb1: {}
  }

  average_total_returns = {}
  total_regrets = {}
  all_actions = {}
  total_regret = []

  # Number of bandits also
  num_actions = 10

  # Creates 10 biases between 0-1
  biases = [1.0 / k for k in range(5, 5+num_actions)]

  best_action_index = 0

  def best_action(mab):
    return best_action_index

  start_time = time.time()

  for strategy, parameters in strategies.items():
    # Creates bandits with given biases, for bias 0.2, 0.2 is for bias and (1 - 0.2) = 0.8 is for q_value 
    # set 1-bias to zero for the schedule from lecture
    bandits = [Bandit(bias, 1-bias) for bias in biases]

    # with all the individual bandits, a multi armed bandit with best action index 0 is created
    mab = MAB(best_action, *bandits)

    # 1000000 is number of rounds
    total_regret, average_total_return, all_actions_dist = mab.run(no_rounds, strategy, **parameters)
    average_total_returns[strategy.__name__] = average_total_return
    total_regrets[strategy.__name__] = total_regret
    all_actions[strategy.__name__] = all_actions_dist

  print("--- %s seconds ---" % (time.time() - start_time))
  
  for strategy in (all_actions.keys()):
    print("Action frequencies of strategy ", strategy)
    print(sci.itemfreq(all_actions[strategy]))
  
  for i, key in enumerate(total_regrets):
    plt.plot(total_regrets[key], label = key)
    plt.title("Total regret")

  plt.legend()
  plt.show()
  print("END")