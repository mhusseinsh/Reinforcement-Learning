import unittest
import numpy as np
import sys
if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.gridworld import GridworldEnv
if "../" not in sys.path:
  sys.path.append("../")
from scripts import policy_iteration
from scripts import value_iteration

def setUpModule():
  global env
  env = GridworldEnv()

class TestPolicyEval(unittest.TestCase):
  def test_policy_eval(self):
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_iteration.policy_eval(random_policy, env)
    expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

class TestIterationAlgorithm:
  def test_policy(self):
    expected_policy = [[1., 0., 0., 0.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [0., 0., 1., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 1., 0.],
                       [1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 1., 0., 0.],
                       [1., 0., 0., 0.]]
    np.testing.assert_array_equal(self.policy, expected_policy)
    
  def test_value(self):
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(self.v, expected_v, decimal=2)

class TestPolicyImprovement(unittest.TestCase, TestIterationAlgorithm):
  @classmethod
  def setUpClass(cls):
    cls.policy, cls.v = policy_iteration.policy_improvement(env)

class TestValueIteration(unittest.TestCase, TestIterationAlgorithm):
  @classmethod
  def setUpClass(cls):
    cls.policy, cls.v = value_iteration.value_iteration(env)

if __name__ == '__main__':
  unittest.main()