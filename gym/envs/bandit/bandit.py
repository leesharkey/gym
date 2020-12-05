import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class BanditEnv(gym.Env):
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 30
    # }

    # def __init__(self, g=10.0):
    #     self.max_speed = 8
    #     self.max_torque = 2.
    #     self.dt = .05
    #     self.g = g
    #     self.m = 1.
    #     self.l = 1.
    #     self.viewer = None
    #
    #     high = np.array([1., 1., self.max_speed], dtype=np.float32)
    #     self.action_space = spaces.Box(
    #         low=-self.max_torque,
    #         high=self.max_torque, shape=(1,),
    #         dtype=np.float32
    #     )
    #     self.observation_space = spaces.Box(
    #         low=-high,
    #         high=high,
    #         dtype=np.float32
    #     )
    #
    #     self.seed()

    def __init__(self, p_dist_dist_name=None):

        self.r_dist = [1, 1]
        self.p_dist_dist_name = 'FixedEasy' # From 'Dirichlet', 'Dirichlet_support_subset', 'FixedEasy'

        self.n_bandits = len(self.r_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.n_bandits + 1,),
                                            dtype=np.float32)
        self.prev_act, self.prev_rew = 0, 0.

        # Define the restricted subset of the support of the Dirichlet
        # distribution
        num_ele_subset = 51
        increment = 1/num_ele_subset
        self.dirichlet_support_subset = [[p,1-p] for p in
                                         np.arange(0,1+increment,increment)]

        self.seed()
        self.sample_p_dist_dist()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action[0]
        assert self.action_space.contains(action)
        reward = 0

        if np.random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        self.prev_act, self.prev_rew = action, reward

        info = {'p_dist': self.p_dist}

        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        return self.prev_act_and_reward_to_vec(self.prev_act, self.prev_rew)

    def reset(self):
        self.sample_p_dist_dist()
        self.prev_act, self.prev_rew = 0, 0.
        return self._get_obs()

    def to_one_hot(self, idx):
        one_hot_vec = np.zeros(self.n_bandits)
        one_hot_vec[idx] = 1
        return one_hot_vec

    def prev_act_and_reward_to_vec(self, prev_act, reward):
        one_hot_act = self.to_one_hot(prev_act)
        one_hot_act_and_rew = np.append(one_hot_act, reward)
        # print(one_hot_act_and_rew)
        return one_hot_act_and_rew

    def sample_p_dist_dist(self):
        # self.seed(self.orig_seed)

        if self.p_dist_dist_name == 'Dirichlet':
            self.p_dist = self.np_random.dirichlet([1]*self.n_bandits)# 0.1 is easy. 1 is average. 1000 is very hard.
        elif self.p_dist_dist_name == 'Dirichlet_support_subset':
            self.p_dist = self.np_random.choice(self.dirichlet_support_subset)
        elif self.p_dist_dist_name == 'FixedEasy':
            self.p_dist = np.array([0.9, 0.1])
        else:
            raise ValueError("Invalid distribution name for bandits")

    # def step(self, u):
    #     th, thdot = self.state  # th := theta
    #
    #     g = self.g
    #     m = self.m
    #     l = self.l
    #     dt = self.dt
    #
    #     u = np.clip(u, -self.max_torque, self.max_torque)[0]
    #     self.last_u = u  # for rendering
    #     costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    #
    #     newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    #     newth = th + newthdot * dt
    #     newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
    #
    #     self.state = np.array([newth, newthdot])
    #     return self._get_obs(), -costs, False, {}

    # def reset(self):
    #     high = np.array([np.pi, 1])
    #     self.state = self.np_random.uniform(low=-high, high=high)
    #     self.last_u = None
    #     return self._get_obs()
    #
    # def _get_obs(self):
    #     theta, thetadot = self.state
    #     return np.array([np.cos(theta), np.sin(theta), thetadot])
