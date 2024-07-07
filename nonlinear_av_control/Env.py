import numpy as np
import os
import numpy.typing as npt
from nonlinear_av_control.Utils import np_to_torch
import matplotlib.pyplot as plt

class Env():
    """Class that houses an eviroment for having a bike follow a trajectory.
    
    Attributes:
        n (int): The number of instances of the environment.
        window_size (int): The size of the window for the plot.
        agent_state (npt.ArrayLike): The state of the agent.
        n_in_out (tuple): The number of inputs and outputs for the model.
        wheelbase (float): The wheelbase of the bike.
        max_timesteps (int): The maximum number of timesteps.
        thetad (npt.ArrayLike): The series of desired angle for each waypoint.
        yd (npt.ArrayLike): The desired y position for each waypoint.
        xd (npt.ArrayLike): The desired x position for each waypoint.
        vd (npt.ArrayLike): The desired velocity for each waypoint.
        omega_d (npt.ArrayLike): The desired angular velocity for each waypoint.
        t_vec (npt.ArrayLike): The time vector for the waypoints.
        ang_acc_d (npt.ArrayLike): The desired angular acceleration for each waypoint.
        dt (float): The time step.
        t (float): The current time.
        x_y_history (npt.ArrayLike): The history of the x and y positions of the agent.
        state (npt.ArrayLike): The state of the agent.
        distance_to_waypoint (npt.ArrayLike): The distance to the waypoint.
        fig (matplotlib.figure.Figure): The figure for the plot.
        ax (matplotlib.axes.Axes): The axes for the plot.
        pygame_init (bool): Whether pygame is initialized.
        space_range (int): The range of the space.
        draw_traj (bool): Whether the trajectory is drawn.
    """
    def __init__(self, n, dt=.01, max_timesteps=1000) -> None:
        """Initialize.
        
        Args:
            n (int): The number of instances of the environment.
            dt (float, optional): The time step. Defaults to .01.
            max_timesteps (int, optional): The maximum number of timesteps. 
                Defaults to 1000.
            """
        self.n = n
        self.window_size = 1000
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
        self.max_timesteps = max_timesteps
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
        self.distance_to_waypoint = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(0, self.window_size)
        self.pygame_init = False
        self.space_range = 1000
        self.draw_traj = False


    def step(self, action):
        """Perform a step in the environment.
        
        Args:
            action (npt.ArrayLike): The action to take.
            
        Returns:
            npt.ArrayLike: The state.
            npt.ArrayLike: The reward.
            bool: Whether the episode is done.
            dict: The info.
        """
        index = int(np.floor(self.t*(1/self.dt)) + 1)
        # Gather the desired values for the next waypoint
        thetar = self.thetad[index]
        xr = self.xd[index]
        yr = self.yd[index]
        # get the current state of all instances
        x = self.state[:, 0]
        y = self.state[:, 1]
        theta = self.state[:, 2]
        # this is not needed for the norm but i wanna and you cant stop me :P
        thetadiff: np.float64 = thetar - theta
        thetadiff = np.arctan(np.sin(thetadiff) / np.cos(thetadiff))
        vec_diff = np.array([xr - x, yr - y, thetadiff])
        self.distance_to_waypoint.append(np.linalg.norm(np.mean(vec_diff[0:2], axis=1), axis=0))
        vec_diff_mean = np.mean(vec_diff, axis=1)
        # reward function
        reward = np_to_torch(np.array([-np.linalg.norm(vec_diff_mean)])) / 50
        mean_x = np.abs(vec_diff_mean.item(0))
        mean_y = np.abs(vec_diff_mean.item(1))
        mean_theta = np.abs(vec_diff_mean.item(2))
        if mean_x < 1e-3:
            reward += .33 
        if mean_y < 1e-3:
            reward += .33
        if mean_theta < 1e-3:
            reward += .33
        if mean_x < 1e-3 and mean_y < 1e-3 and mean_theta < 1e-3:
            reward += 1
        v = action[:,0]
        phi = action[:, 1]
        # Update the state
        dvdt = np.array([v*np.cos(theta), v*np.sin(theta), v*np.tan(phi)/self.wheelbase])
        dvdt = np.transpose(dvdt)
        # euler integration
        new_state = self.state + self.dt * dvdt
        new_theta = new_state[:, 2]
        new_theta = np.arctan(np.sin(new_theta) / np.cos(new_theta))
        new_state[:, 2] = new_theta
        self.state = new_state
        self.t += self.dt
        done = False
        if self.t > self.max_timesteps*self.dt:
            done = True
        return self.state, reward, done, {}

    def render(self):
        """Render the environment."""
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(0, self.window_size)
        
        self.ax.clear()
        # Draw the trajectory
        if not self.draw_traj:
            self.draw_traj = True
        self._draw_trajectory()

        # Draw current position and orientation of the agent
        self._draw_agent()

        plt.pause(0.001)

    def _draw_trajectory(self):
        len_desired = len(self.xd)
        for i in range(len_desired - 1):
            x1, y1 = self._transform_coordinates(self.xd[i], self.yd[i])
            x2, y2 = self._transform_coordinates(self.xd[i + 1], self.yd[i + 1])
            self.ax.plot([x1, x2], [y1, y2], 'k-')

    def _draw_agent(self):
        x, y, theta = self.state.item(0), self.state.item(1), self.state.item(2)
        x, y = self._transform_coordinates(x, y)
        triangle = [
            (x + 1 * np.cos(theta), y + 1 * np.sin(theta)),
            (x + 1 * np.cos(theta - 2 * np.pi / 3), y + 1 * np.sin(theta - 2 * np.pi / 3)),
            (x + 1 * np.cos(theta + 2 * np.pi / 3), y + 1 * np.sin(theta + 2 * np.pi / 3))
        ]
        polygon = plt.Polygon(triangle, color='black')
        self.ax.add_patch(polygon)

    def _transform_coordinates(self, x, y):
        screen_x = (x + self.space_range) * self.window_size / (2 * self.space_range)
        screen_y = (y + self.space_range) * self.window_size / (2 * self.space_range)
        return screen_x, screen_y    


    def reset(self):
        """Reset the environment."""
        state = np.array([self.xd[0], self.yd[0], self.thetad[0]], dtype=np.float32)
        self.t = 0.
        self.state = np.repeat(state[np.newaxis, :], self.n, axis=0)
        self.distance_to_waypoint = []
        return self.state

        