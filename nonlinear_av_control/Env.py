import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


class BikeModelEnv:
    """Class for training got to waypoint with bike model dynamics.

    Attributes:
        state: the state of the agents with shape (n, 3) state is ordered
            (x, y, theta)
        n: number of parallel enviroments
        wheelbase: the distance between the centerpoints of the wheel looking at
            the profile
        max_t: the max number of timesteps
        dt: the integration time step
        xd: the desired x positions (starting at 0)
        yd: the desired y positions (starting at 0)
        thetad: the desired orientation (not neccesarily starting at 0)
    """

    def __init__(self, n, dt, max_timesteps) -> None:
        """Initialize.

        Args:
            n: the number of parallel enviroments
            dt: the integration time step
            max_timesteps: the max number of simulation timesteps
        """
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
        """Perform as step in the enviroment and get the reward.

        Args:
            action: the action the agents will perform size (n, 2)

        Return:
            state for mlps, reward, done, []
        """
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
        new_state = self.state + dvdt * self.dt
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
        norm = np.linalg.norm(diff_vec, axis=1)
        if_at_waypoint_and_pointing = norm < 0.01
        not_quite_wp = np.logical_and(norm > 0.01, ~if_not_at_waypoint)
        reward[if_at_waypoint_and_pointing] += 10
        reward[not_quite_wp] -= norm[not_quite_wp] / (50 * 2 * np.pi)
        done = np.logical_or(
            np.abs(self.state[:, 0]) > self.space_size,
            np.abs(self.state[:, 1]) > self.space_size,
        )
        reward[done] -= 10
        done = done & if_at_waypoint_and_pointing
        self.selective_reset(done)
        return self.get_state(), reward, done, None

    def selective_reset(self, done):
        """Selectivly resets the enviroments that are finished:

        Args:
            done: boolean array that indicates if enviroment is done shape
                (n,1)
        """
        count = np.count_nonzero(done)
        if count > 0:
            self.state[done] = np.random.uniform(-25, 25, (count, 3))
            self.state[done][:, 2] = np.random.uniform(-np.pi, np.pi, count)
            self.t[done] = 0.0

    def reset(self):
        """Reset all enviroments."""
        done = np.ones(self.n, dtype=bool)
        self.selective_reset(done)
        self.thetad = np.random.uniform(0, 2 * np.pi, (1,))
        return self.get_state()

    def get_state(self):
        """Get the normalized state for the actor and the critic.

        Returns:
            the state and goal concatinated with shape (n,6)
            (x, y, theta, xd, yd, thetad)/(ss, ss, pi, ss, ss, pi)
        """
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
        """Render current state and goal."""
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
