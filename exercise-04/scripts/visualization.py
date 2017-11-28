import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from q_learning import q_learning
from off_policy_mc import create_random_policy, mc_control_importance_sampling
import sys
if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.blackjack import BlackjackEnv
from lib.envs.cliff_walking import CliffWalkingEnv

def plot_value_function(V, axarr, title="Value Function"):
  """
  Plots the value function as a surface plot.
  """
  min_x = min(k[0] for k in V.keys())
  max_x = max(k[0] for k in V.keys())
  min_y = min(k[1] for k in V.keys())
  max_y = max(k[1] for k in V.keys())

  x_range = np.arange(min_x, max_x + 1)
  y_range = np.arange(min_y, max_y + 1)
  X, Y = np.meshgrid(x_range, y_range)

  # Find value for all (x, y) coordinates
  Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
  Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

  def plot_surface(X, Y, Z, ax, title):
    #fig = plt.figure(figsize=(20, 10))
    #ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf, ax=ax)

  plot_surface(X, Y, Z_noace, axarr[0], "{} (No Usable Ace)".format(title))
  plot_surface(X, Y, Z_ace, axarr[1], "{} (Usable Ace)".format(title))

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
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
  if noshow:
    plt.close(fig2)
  else:
    plt.show(fig2)

  # Plot time steps and episode number
  fig3 = plt.figure(figsize=(10,5))
  plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
  plt.xlabel("Time Steps")
  plt.ylabel("Episode")
  plt.title("Episode per time step")
  if noshow:
    plt.close(fig3)
  else:
    plt.show(fig3)

  return fig1, fig2, fig3

if __name__ == "__main__":
  env = BlackjackEnv()
  random_policy = create_random_policy(env.action_space.n)
  Q, _ = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)

  V = defaultdict(float)
  for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value

  fig, axarr = plt.subplots(2, subplot_kw={'projection': '3d'})
  plot_value_function(V, axarr, title="500.000 Steps")
  plt.show()

  env = CliffWalkingEnv()
  Q, stats = q_learning(env, 500)
  plot_episode_stats(stats)