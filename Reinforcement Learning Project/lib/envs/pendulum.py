import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from PIL import Image

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, reward_function):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None
        self.use_reward_function = reward_function

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()

        """if reward_function is None:
                                    def reward(pendulum):
                                        return 1 if -0.1 <= angle_normalize(pendulum.state[0]) <= 0.1 else 0
                                    self.reward = reward
                                else:
                                    def reward_function(pendulum, th, thdot):
                                        cost = angle_normalize(th)**2 + .1*thdot**2 + .001*(pendulum.last_u**2)
                                        return -cost
                                    self.reward = reward_function"""
        if self.use_reward_function is False:
            def reward(pendulum):
                return 1 if -0.1 <= angle_normalize(pendulum.state[0]) <= 0.1 else 0
            self.reward = reward
        else:
            def reward_function(pendulum, th, thdot):
                cost = angle_normalize(th)**2 + .1*thdot**2 + .001*(pendulum.last_u**2)
                return -cost
            self.reward = reward_function

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state
        #print("in step func",th, thdot)

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        #print("action in env: ",self.last_u)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt

        newth = th + newthdot*dt

        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        if self.use_reward_function is False:
            reward = self.reward(self)
        else:
            reward = self.reward(self, th, thdot)
        #print("i am here: ", newth, newthdot)
        self.state = np.array([newth, newthdot])
        
        return self._get_obs(), reward, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state

        #print("in get obs: ",np.cos(theta), np.sin(theta))
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)