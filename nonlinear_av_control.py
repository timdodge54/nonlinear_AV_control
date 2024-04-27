import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import numpy.typing as npt

class Vehicle:
    """Class thata houses the vehicle dynamics and controller.
    The vehicle is modeled as a point mass with a bicycle model.
    The controller is a lyapunov controller that uses the error in the
    orientation and position of the vehicle to generate control inputs.
    
    Attributes:
        self.x0: Initial state of the vehicle
        self.l: Length of the wheelbase of the vehicle
        self.controller_type: Type of controller to use
        self.vd: list of desired velocities for waypoints
        self.thetad: list of desired orientations for waypoints
        self.yd: list of desired y positions for waypoints
        self.xd: list of desired x positions for waypoints
        self.thetaddot: list of desired angular accelerations for waypoints
        self.phid: list of desired steering angles for waypoints
        self.vr_prev: previous velocity of the vehicle
        self.phi_r_prev: previous steering angle of the vehicle
    """
    def __init__(
        self,
        x0: float,
        y0: float,
        theta0: float,
        l: float,
        vd: npt.NDArray,
        thetad: npt.NDArray,
        yd: npt.NDArray,
        xd: npt.NDArray,
        thetaddot: npt.NDArray,
        phid: npt.NDArray) -> None:
        """Initialize.
        """
        self.x0 = np.array([x0, y0, theta0])
        self.l = l
        self.controller_type = 1
        self.vd = vd
        self.thetad = thetad
        self.yd = yd
        self.xd = xd
        self.thetaddot = thetaddot
        self.phid = phid    
        self.vr_prev = 0.
        self.phi_r_prev = 0.


    def f(self, state: npt.NDArray , t: float) -> npt.NDArray:
        """Calculate the dynamics of the vehicle.
        
        Args:
            state: state of the vehicle
            t: time
        """
        # Get the next index for the next waypoint to follow
        # sampling frequency for the waypoints is 100 Hz
        index = np.floor(t * 100) + 2
        index = int(index)
        # Gather the desired values for the next waypoint
        vr = self.vd[index]
        wr = self.thetaddot[index]
        thetar = self.thetad[index]
        xr = self.xd[index]
        yr = self.yd[index]
        phir = self.phid[index]
        thetaddot = self.thetaddot[index]
        # Unpack the state
        x = state.item(0)
        y = state.item(1)
        theta = state.item(2)
        l = self.l
        # Calculate the error in orientation and position
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        thetadiff = thetar - theta
        thetadiff = np.arctan(np.sin(thetadiff) / np.cos(thetadiff))
        vec_diff = np.array([xr - x, yr - y, thetar - theta])
        vec_e = np.dot(rotation_matrix, vec_diff)
        xe = vec_e[0]
        ye = vec_e[1]
        thetae = vec_e[2]
        # Calculate the control inputs
        if self.controller_type == 1:
            v, phi = self.get_lyap_control(vr, wr, thetae, xe, ye)
        else:
            v, phi = 0, 0
        # Calculate the state derivatives
        xdot = v * np.cos(theta)
        ydot = v * np.sin(theta)
        thetadot = v * np.tan(phi) / l
        xdot = np.array([xdot, ydot, thetadot])
        return xdot
       


    def get_lyap_control(
        self,
        vr: float,
        wr: float,
        thetae: float,
        xe: float,
        ye: float) -> tuple[float, float]:
        """Calculate the control inputs using the lyapunov controller.
        
        Args:
            vr: desired velocity
            wr: desired angular velocity
            thetae: error in orientation
            xe: error in x position
            ye: error in y position
        """
        k1, k2, k3 = 1,3,2
        v = k1 * xe + vr*np.cos(thetae)
        print(v)
        w = wr + vr*(k2*ye + k3*np.sin(thetae))
        phi_r = np.arctan(self.l * w / v)
        return v, phi_r

    def simulate(self, tf, tspan):
        sol = odeint(self.f, self.x0, tspan)
        return sol, tspan
        



if __name__ == "__main__":

    # Import the data from the CSV file
    data = np.genfromtxt('output.csv', delimiter=',', skip_header=1)
    # Convert angles from degrees to radians
    thetad = data[:, 0] * math.pi / 180
    # x and y are swapped because of the sign convention of theta and theta_dot
    yd = data[:, 2]
    xd = data[:, 1]
    vd = data[:, 3]
    print(vd)
    thetaddot = data[:, 4]
    t_vec = data[:, 5]
    l = 2.8
    # calculate the given steering angle which can be done because the 
    # desired trajectory has a constant linear velocity
    deltad = np.arctan(l / vd[0] * thetaddot)

    x0 = [xd[0], yd[0], thetad[0], 0]
    tf = 15
    tspan = np.arange(0, tf, 0.01)
    controller_type = 1

    # Create the vehicle object
    vehicle = Vehicle(x0[0], x0[1], x0[2], l, vd, thetad, yd, xd, thetaddot, deltad)
    # Simulate the vehicle
    x, t = vehicle.simulate(tf, tspan)
    
    control = np.zeros((len(t), 2))

    for i in range(len(t)):
        time = t[i]
        index = int(time * 100) + 2
        index = min(index, len(vd) - 1)
        index = int(index)
        vd_ = vd[index]
        thetad_ = thetad[index]
        thetaddot_ = thetaddot[index]
        xd_ = xd[index]
        yd_ = yd[index]
        phid_ = deltad[index]
        x_ = x[i, 0]
        y_ = x[i, 1]
        theta_ = x[i, 2]
        rotation_matrix = np.array([
            [np.cos(theta_), -np.sin(theta_), 0],
            [-np.sin(theta_), np.cos(theta_), 0],
            [0, 0, 1]
        ])
        thetadiff = thetad_ - theta_
        thetadiff = np.arctan(np.sin(thetadiff) / np.cos(thetadiff))
        vec_diff = np.array([xd_ - x_, yd_ - y_, thetad_ - theta_])
        vec_e = np.dot(rotation_matrix, vec_diff)
        xe = vec_e[0]
        ye = vec_e[1]
        thetae = vec_e[2]
        control[i, :] = vehicle.get_lyap_control(vr=vd_, wr=thetaddot_, thetae=thetae, xe=xe, ye=ye)


    # Plotting
    plt.figure(figsize=(10, 20))
    plt.subplot(5, 1, 1)
    plt.plot(t, x[:, 0], label='x_r', linewidth=1.5)
    plt.plot(t_vec, xd, '--', label='x_d', linewidth=1.5)
    plt.title("X position vs desired X position")
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(t, x[:, 1], label='y_r', linewidth=1.5)
    plt.plot(t_vec, yd, '--', label='y_d', linewidth=1.5)
    plt.title("Y position vs desired Y position")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(t, x[:, 2], label='theta_r', linewidth=1.5)
    plt.plot(t_vec, thetad, '--', label='theta_d', linewidth=1.5)
    plt.title("Orientation vs Desired orientation")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(t, control[:, 0], label='v', linewidth=1.5)
    plt.title("Velocity control")

    plt.subplot(5, 1, 5)
    plt.plot(t, control[:, 1], label='phi', linewidth=1.5)
    plt.title("Steering control")
    # space plots so titles don't overlap
    plt.tight_layout()
    
    fig, ax = plt.subplots(1)
    ax.plot(xd, yd,'--', label='Desired path', linewidth=1.5, color='black')
    ax.plot(x[:, 0], x[:, 1], label='Path', linewidth=1.5, color='blue')
    ax.set_aspect('equal')
    plt.title("Path followed by the vehicle")
    plt.legend()
    plt.show()
    
