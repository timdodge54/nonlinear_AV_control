import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


class BikeModelEnv:
    def __init__(self, n, dt, max_timesteps) -> None:
        self.delta_t = 0.1
        self.state = np.random.uniform(-25, 25, (n, 3))
        self.state[:, 2] = np.random.uniform(-np.pi, np.pi, n)
        self.n = n
        self.wheelbase = 2.8
        self.space_size = 50
        self.t = np.zeros(n)
        self.max_t = max_timesteps
        self.dt = dt
        self.in_n_out = 6, 2
        file_total_path = os.path.join(
            os.path.dirname(__file__), "../data/output_acceleration.csv"
        )
        data = np.genfromtxt(file_total_path, delimiter=",", skip_header=1)
        self.xd = data[:, 1] - data[:, 1][0]
        self.yd = data[:, 2] - data[:, 2][0]
        # self.thetad = data[:, 0] - data[:, 0][0]
        # self.xd = np.random.uniform(-25, 25, (1,))
        # self.yd = np.random.uniform(-25, 25, (1,))
        self.thetad = np.random.uniform(0, 2 * np.pi, (1,))

    def step(self, action):
        xd = self.xd[0]
        yd = self.yd[0]
        thetad = self.thetad[0]

        v = action[:, 0]
        phi = action[:, 1]
        theta = self.state[:, 2]
        dvdt = np.array(
            [
                v * np.cos(theta),
                v * np.sin(theta),
                v * np.tan(phi) / self.wheelbase,
            ]
        )
        dvdt = np.transpose(dvdt)
        new_state = self.state + dvdt * self.delta_t * self.dt
        new_state[:, 2] = (
            np.arctan2(np.sin(new_state[:, 2]), np.cos(new_state[:, 2]) + (2 * np.pi))
        ) % (2 * np.pi)
        self.state = new_state
        reward = np.zeros(self.n)
        x, y = self.state[:, 0], self.state[:, 1]
        xdiff = x - xd
        ydiff = y - yd

        xy_diff = np.transpose(np.array([xdiff, ydiff]))
        x_y_norm = np.linalg.norm(xy_diff, axis=1)
        if_not_at_waypoint = x_y_norm > 1
        reward[if_not_at_waypoint] -= x_y_norm[if_not_at_waypoint] / 50
        diff_vec = np.array([xd, yd, thetad]) - self.state
        diff_vec[:, 2] = np.arctan2(np.sin(diff_vec[:, 2]), np.cos(diff_vec[:, 2])) 
            np.abs(self.state[:, 0]) > self.space_size,
            np.abs(self.state[:, 1]) > self.space_size,
        )
        reward[done] -= 10
        self.selective_reset(done)
        return self.get_state(), reward, done, None

    def selective_reset(self, done):
        count = np.count_nonzero(done)
        if count > 0:
            self.state[done] = np.random.uniform(-25, 25, (count, 3))
            self.state[done][:, 2] = np.random.uniform(-np.pi, np.pi, count)
            self.t[done] = 0.0

    def reset(self):
        done = np.ones(self.n, dtype=bool)
        self.selective_reset(done)
        self.thetad = np.random.uniform(0, 2 * np.pi, (1,))
        return self.get_state()

    def get_state(self):
        repeat_goal = np.tile(
            np.array([self.xd.item(0), self.yd.item(0), self.thetad.item(0)]),
            (self.n, 1),
        )
        return np.hstack([self.state, repeat_goal]) / np.array(
            [
                self.space_size,
                self.space_size,
                np.pi,
                self.space_size,
                self.space_size,
                np.pi,
            ]
        )

    def render(self):
        goalx = self.xd.item(0)
        goaly = self.yd.item(0)
        goaltheta = self.thetad.item(0)
        goalu, goalv = np.cos(goaltheta), np.sin(goaltheta)
        plt.quiver(goalx, goaly, goalu, goalv, color="g", alpha=0.2)
        x, y, theta = self.state[:, 0], self.state[:, 1], self.state[:, 2]
        u, v = np.cos(theta), np.sin(theta)
        plt.quiver(x, y, u, v, headwidth=1, pivot="mid", color="r", alpha=1)
        plt.axis("square")
        plt.xlim(-self.space_size, self.space_size)
        plt.ylim(-self.space_size, self.space_size)
        plt.pause(0.01)
