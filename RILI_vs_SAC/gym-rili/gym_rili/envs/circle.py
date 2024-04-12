import numpy as np
import gym
from gym import spaces


# Ego agent localisation
ego_home = np.array([0.0, 0.5])


class Circle(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(
            low=-0.2,
            high=+0.2,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(2,),
            dtype=np.float32
        )
        self.action_space_other = spaces.Box(
            low=-np.pi,
            high=+np.pi,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space_other = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(2,),
            dtype=np.float32
        )

        self.radius = 1.0
        self.change_partner = 0.99
        self.reset_theta = 0.999
        self.ego = np.copy(ego_home)
        # self.step_size = np.random.random() * 2 * np.pi - np.pi
        self.other = np.array([self.radius, 0.])
        self.theta = 0.0
        self.partner = 0
        self.timestep = 0
        self.ego_state_last = self.ego
        
    def set_params(self, change_partner):
        self.change_partner = change_partner


    def _get_obs(self):
        return np.copy(self.ego)

    def _get_obs_other(self):
        return np.copy(self.other)

    def polar(self, theta):
        return self.radius * np.array([np.cos(theta), np.sin(theta)])

    def reset(self):
        state = self._get_obs()
        return state

    def reset_opponent(self):
        state = self._get_obs_other()
        return state

    def step(self, actions):
        if self.timestep == 0:
            theta=actions[1]
            self.other = self.polar(theta)
            self.other = [item for row in self.other for item in row]
            # print("At step 0, Hider:",self.other)
        self.timestep += 1
        self.ego += actions[0]
        reward_ego = -np.linalg.norm(self.other - self.ego) * 100
        reward_other = -reward_ego
        done = False
        if self.timestep == 10:
            self.timestep = 0
            # if np.random.random() > self.change_partner:
            #     self.partner += 1
            #     self.step_size = np.random.random() * 2 * np.pi - np.pi
            # self.theta += self.step_size
            self.ego_state_last = np.copy(self.ego)
            # print("Hider's position:",self.other)
            # print("Seeker's last position and reward: ",self.ego_state_last, reward_ego)
            self.ego = np.copy(ego_home)
        return [self._get_obs(),self._get_obs_other(), self.ego_state_last], [reward_ego, reward_other], done, {}

