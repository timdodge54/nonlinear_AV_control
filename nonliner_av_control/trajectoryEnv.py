import gym
from gym import spaces
import pygame
import numpy as np
import os
import numpy.typing as npt
from scipy.integrate import odeint

class TrajectoryEnv():
    def __init__(self, n, dt=.01, max_timesteps=1000, if_render=False) -> None:
        self.n = n
        self.window_size = 512
        self.agent_state = np.zeros((n, 3), dtype=np.float32)
        self.n_in_out =  3, 2
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
        self.x_y_history = np.zeros((n, 2, max_timesteps), dtype=np.float32)
        # set the inital state to the first waypoint with n number of instances
        state = np.array([self.xd[0], self.yd[0], self.thetad[0]], dtype=np.float32)
        # repeat this so that you have n number of instances
        self.state = np.repeat(state[np.newaxis, :], n, axis=0)
        self.pygame_init = False
        self.space_range = 512


    def step(self, action):
        index = np.floor(self.t*100) + 1
        # Gather the desired values for the next waypoint
        thetar = self.thetad[index]
        xr = self.xd[index]
        yr = self.yd[index]
        # get the current state of all instances
        x = self.state[:, 0]
        y = self.state[:, 1]
        theta = self.state[:, 2]
        # this is not needed for the norm but i wanna and you cant stop me :P
        rotation_matrix: npt.NDArray[np.float64] = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
        ])
        thetadiff: np.float64 = thetar - theta
        thetadiff = np.arctan(np.sin(thetadiff) / np.cos(thetadiff))
        vec_diff = np.array([xr - x, yr - y, thetar - theta])
        vec_e = np.dot(rotation_matrix, vec_diff)
        # reward function
        reward = -np.linalg.norm(vec_e)
        v = action[0]
        phi = action[1]
        # Update the state
        def f(state):
            theta_ = state[2]
            return np.array(
                [v * np.cos(theta_),
                 v * np.sin(theta_),
                 v * np.tan(phi) / self.wheelbase])
        # euler integration
        new_state = self.state + self.dt * f(self.state)
        new_theta = new_state[:, 2]
        new_theta = np.arctan(np.sin(new_theta) / np.cos(new_theta))
        new_state[:, 2] = new_theta
        self.state = new_state
        self.t += self.dt
        if self.t > self.t_vec[-1]:
            done = True
        return self.state, reward, done, {}

    def get_state(self):
        return self.state

    def render(self, mode='human'):
        if not self.pygame_init:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame
            self.pygame_init = True
        self.screen.fill((255, 255, 255))
        # draw the desired trajectory connecting the waypoints
        len_desired = len(self.xd)
        for i in range(len_desired - 1):
            # get the point and transform it to the screen coordinates
            x1 = self.xd[i]
            y1 = self.yd[i]
            x1, y1 = self._transform_coordinates(x1, y1)
            x2 = self.xd[i + 1]
            y2 = self.yd[i + 1]
            x2, y2 = self._transform_coordinates(x2, y2)
            pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x2, y2), 2)
        # draw the current position and orientation of the agent
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        # draw the agent as a triangle pointing in the direction of theta
        x, y = self._transform_coordinates(x, y)
        # draw the agent as a triangle pointing in the direction of theta
        triangle = [
            (x + 10 * np.cos(theta), y + 10 * np.sin(theta)),
            (x + 10 * np.cos(theta - 2 * np.pi / 3), y + 10 * np.sin(theta - 2 * np.pi / 3)),
            (x + 10 * np.cos(theta + 2 * np.pi / 3), y + 10 * np.sin(theta + 2 * np.pi / 3))
        ]
        pygame.draw.polygon(self.screen, (0, 0, 0), triangle)
        pygame.display.flip()
        self.clock.tick(60)

    def _transform_coordinates(self, x, y):
        screen_x = (x + self.space_range) * self.window_size / (2 * self.space_range)
        screen_y = (y + self.space_range) * self.window_size / (2 * self.space_range)
        return int(screen_x), int(screen_y)

    def reset(self):
        state = np.array([self.xd[0], self.yd[0], self.thetad[0]], dtype=np.float32)
        self.state = np.repeat(state[np.newaxis, :], self.n, axis=0)
        return self.state

        