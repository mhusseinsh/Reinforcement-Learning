import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools
import pandas as pd
from PIL import Image
import time

if "../../lib/envs" not in sys.path:
  sys.path.append("../../lib/envs")
from mountain_car import MountainCarEnv

"""
* -------------------------------------------------------------------------------
* There are also TODOs in Policy Class!
* -------------------------------------------------------------------------------
"""

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
VALID_ACTIONS = [0, 1, 2]
max_episode_length = 500
baseline = -10

class Policy():
  def __init__(self):
    self._build_model()

  def _build_model(self):
    """
    Builds the Tensorflow graph.
    """

    self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    # The TD target value
    self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32)
    # Integer id of which action was selected
    self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32)

    batch_size = tf.shape(self.states_pl)[0]

    self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc3 = tf.contrib.layers.fully_connected(self.fc2, len(VALID_ACTIONS), activation_fn=None,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))

    # -----------------------------------------------------------------------
    # TODO: Implement softmax output
    # -----------------------------------------------------------------------
    #raise NotImplementedError("Softmax output not implemented.")
    #layers.softmax(fc3)

   # self.predictions = tf.contrib.layers.softmax(self.fc3)
    self.predictions = tf.squeeze(tf.nn.softmax(self.fc3))
    self.action_predictions = tf.gather(self.predictions, self.actions_pl)

    # Get the predictions for the chosen actions only
    #gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
    #self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

    # -----------------------------------------------------------------------
    # TODO: Implement the policy gradient objective. Do not forget to negate
    # -----------------------------------------------------------------------
    # the objective, since the predefined optimizers only minimize in
    # tensorflow.

    #-tf.reducemean(tf.log...)
    #self.objective = -tf.log(self.action_predictions) * self.targets_pl
    self.objective = -tf.reduce_mean(tf.log(self.action_predictions) * (self.targets_pl - baseline))
    #raise NotImplementedError("Softmax output not implemented.")

    self.optimizer = tf.train.AdamOptimizer(0.00001)
    self.train_op = self.optimizer.minimize(self.objective)

    init = tf.global_variables_initializer()

  def predict(self, sess, s):
    """
    Args:
      sess: TensorFlow session
      states: array of states for which we want to predict the actions.
    Returns:
      The prediction of the output tensor.
    """
    p = sess.run(self.predictions, { self.states_pl: s })
    return np.random.choice(VALID_ACTIONS, p=p), p

  def update(self, sess, s, a, y):
    """
    Updates the weights of the neural network, based on its targets, its
    predictions, its loss and its optimizer.

    Args:
      sess: TensorFlow session.
      states: [current_state] or states of batch
      actions: [current_action] or actions of batch
      targets: [current_target] or targets of batch
    """
    feed_dict = { self.states_pl: s, self.targets_pl: y, self.actions_pl: a }
    sess.run(self.train_op, feed_dict)

class BestPolicy(Policy):
  def __init__(self):
    Policy.__init__(self)
    self._associate = self._register_associate()

  def _register_associate(self):
    tf_vars = tf.trainable_variables()
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:total_vars//2]):
      op_holder.append(tf_vars[idx+total_vars//2].assign((var.value())))
    return op_holder

  def update(self, sess):
    for op in self._associate:
      sess.run(op)

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
  num_states = max_episode_length
  state_elements = 2
  state_space = (num_states, state_elements)
  num_actions = len(VALID_ACTIONS)

  # We start computation with zeros for all state values
  V = np.zeros(state_space)

  # We don't have P....
  transition_prob = 1.0
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
        next_state, reward, done, _ = env.step(a)
        next_pos = next_state[0]
        next_vel = next_state[1]
        #for transition_prob, next_state, reward, done in env.P[s][a]:
          # We are looping over all transition states of the current action,
          # and computing action values
          #actions_s[a] += reward + discount_factor * transition_prob * V[next_state]
        print(V[s])
        actions_s[a] += reward + discount_factor * transition_prob * V[[next_state]]
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

  return V

def reinforce(sess, env, policy, best_policy, num_episodes, discount_factor=1.0):
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

  for i_episode in range(num_episodes):
    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    state = env.reset()

    # To log the best_policy 
    best_policy_reward = 0

    print("\r Episode {}/{} ({})".format(
             i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))
    for t in range(max_episode_length):
      # -----------------------------------------------------------------------
      # TODO: Implement this
      # -----------------------------------------------------------------------
      #raise NotImplementedError("REINFORCE not implemented.")

      # Get the action from the policy
      action_predicted, action_probs = policy.predict(sess,[state])
      selected_action_prob = action_probs[action_predicted]
      next_state, reward, done, _ = env.step(action_predicted)

      episode.append((state, action_predicted, next_state, reward, done, selected_action_prob))

      # Update statistics
      stats.episode_rewards[i_episode] += reward
      stats.episode_lengths[i_episode] = t

      # Only log the best policy if the episode is done
      if done:
        # Only log the best_policy if it was better than the previous best_policy one
        if best_policy_reward < stats.episode_rewards[i_episode]:
          best_policy_reward = stats.episode_rewards[i_episode]
          best_policy.update(sess)

        break

      state = next_state

    for i,e in enumerate(episode):

      #func approximator to the baseline
      #constant baseline
      #target-baseline
      #compatible baseline in case of non-constatnt baseline 
      # Get the total return 
      G = sum([x[3]*(discount_factor**i) for i, x in enumerate(episode[i:])])
      #baseline = V[e[0]]

      # e[0]: state, e[1]: predicted_action
      # Update the policy 
      policy.update(sess,[e[0]],[e[1]],[G])

  #mean_G = np.mean(stats.episode_rewards, axis = 1)
  #print("Mean return:", mean_G)

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
  tf.reset_default_graph()
  env = MountainCarEnv()
  p = Policy()
  bp = BestPolicy()

  sess = tf.Session()

  sess.run(tf.global_variables_initializer())
  start_time = time.time()
  #V = value_iteration(env, 0.0001, 1.0)
  #stats = reinforce(sess, env, p, V, bp, 3000)
  stats = reinforce(sess, env, p, bp, 3000)

  print("--- %s seconds ---" % (time.time() - start_time))
  
  plot_episode_stats(stats)
  saver = tf.train.Saver()

  saver.save(sess, "./policies.ckpt")

  for _ in range(5):
    state = env.reset()
    for i in range(500):
      env.render()
      _, _, done, _ = env.step(p.predict(sess, [state])[0])
      if done:
        break
