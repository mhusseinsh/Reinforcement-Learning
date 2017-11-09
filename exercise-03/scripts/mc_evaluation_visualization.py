import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mc_evaluation import mc_evaluation
import sys
if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.blackjack import BlackjackEnv

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

if __name__ == "__main__":
  env = BlackjackEnv()
  #A policy that sticks if the player score is >= 20 and hits otherwise.
  sample_policy = lambda observation: 0 if observation[0] >= 20 else 1
  V_10k = mc_evaluation(sample_policy, env, num_episodes=10000)
  V_500k = mc_evaluation(sample_policy, env, num_episodes=500000)

  fig, axarr = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
  plot_value_function(V_10k, axarr[0], title="10.000 Steps")
  plot_value_function(V_500k, axarr[1], title="500.000 Steps")
  plt.show()  