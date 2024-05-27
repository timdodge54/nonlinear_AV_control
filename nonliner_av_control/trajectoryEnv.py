import gym
from gym import spaces
import pygame
import numpy as np
import os
import numpy.typing as npt
from scipy.integrate import odeint

class TrajectoryEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 4}
    def __init__(self, dt=.01, render_mode='rgb_array'):
        super(TrajectoryEnv, self).__init__()
        self.window_size = 512
        self.action_space = spaces.Box(
            low=np.array([-100., -np.pi], dtype=np.float32), 
            high=np.array([100., np.pi]), dtype=np.float32)
        self.obeservation_space = spaces.Box(
            low=np.array([-1000., -1000., 0.0]),
            high=np.array([1000., 1000., 2*np.pi]), 
            dtype=np.float32)
        file_total_path = os.path.join(os.path.dirname(
        __file__), '../data/output_acceleration.csv')
        data = np.genfromtxt(
            file_total_path,
            delimiter=',',
            skip_header=1)
        # Convert angles from degrees to radians
        self.thetad = data[:, 0] * np.pi / 180
        # x and y are swapped because of the sign convention of theta and theta_dot
        self.wheelbase = 2.8
        self.yd = data[:, 2]
        self.xd = data[:, 1]
        self.vd = data[:, 3]
        self.omega_d = data[:, 4]
        self.t_vec = data[:, 5]
        self.ang_acc_d = data[:, 6]
        self.dt = dt
        self.t = 0.

        self.state = np.array([self.xd[0], self.yd[0], self.thetad[0]], dtype=np.float32)

    def step(self, action):
        index = np.floor(self.t*100) + 1
        # Gather the desired values for the next waypoint
        vr = self.vd[index]
        wr = self.thetaddot[index]
        thetar = self.thetad[index]
        xr = self.xd[index]
        yr = self.yd[index]
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        rotation_matrix: npt.NDArray[np.float64] = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
        ])
        thetadiff: np.float64 = thetar - theta
        thetadiff = np.arctan(np.sin(thetadiff) / np.cos(thetadiff))
        vec_diff = np.array([xr - x, yr - y, thetar - theta])
        vec_e = np.dot(rotation_matrix, vec_diff)
        reward = -np.linalg.norm(vec_e)
        v = action[0]
        phi = action[1]
        # Update the state
        def f(x, y, theta, v, phi):
            return np.array(
                [v * np.cos(theta),
                 v * np.sin(theta),
                 v * np.tan(phi) / self.wheelbase])
        sol = odeint(f, self.state, [self.t, self.t + self.dt], args=(v, phi))
        new_x, new_y, new_theta = sol[-1]
        new_theta = np.arctan(np.sin(new_theta) / np.cos(new_theta))
        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
            done = True
        self.t += self.dt
        if self.t > self.t_vec[-1]:
            done = True
        self.state = np.array([new_x, new_y, new_theta], dtype=np.float32)
        return self.state, reward, done, {}


    def reset(self):
        self.state = np.array([0., 0.], dtype=np.float32)
        return self.state
    def close(self):
        pass

        