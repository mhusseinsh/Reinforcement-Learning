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
    self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 3, activation_fn=None,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))

    # -----------------------------------------------------------------------
    # TODO: Implement softmax output
    # -----------------------------------------------------------------------
    #raise NotImplementedError("Softmax output not implemented.")

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
    self.objective = -tf.log(self.action_predictions) * self.targets_pl
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
    for t in range(500):
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
      # Get the total return 
      G = sum([x[3]*(discount_factor**i) for i, x in enumerate(episode[i:])])

      # e[0]: state, e[1]: predicted_action
      # Update the policy 
      policy.update(sess,[e[0]],[e[1]],[G])

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
