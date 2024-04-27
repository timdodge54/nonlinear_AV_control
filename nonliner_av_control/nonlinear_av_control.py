import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import numpy.typing as npt
import os


class VehicleSim:
    """Class that a houses the vehicle dynamics, controller, and sim.
    The vehicle is modeled as a point mass with a bicycle model.
    The controller is a lyapunov controller that uses the error in the
    orientation and position of the vehicle to generate control inputs.

    Attributes:
        self.x0: Initial state of the vehicle
        self.wheel_base: Length of the wheelbase of the vehicle
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
            wheel_base: float,
            vd: npt.NDArray[np.float64],
            thetad: npt.NDArray[np.float64],
            yd: npt.NDArray[np.float64],
            xd: npt.NDArray[np.float64],
            thetaddot: npt.NDArray[np.float64],
            phid: npt.NDArray[np.float64],
            angular_acceleration: npt.NDArray[np.float64]) -> None:
        """Initialize.
        """
        self.x0 = np.array([x0, y0, theta0])
        self.wheel_base = wheel_base
        self.controller_type = 1
        self.angular_accel = angular_acceleration
        self.vd = vd
        self.thetad = thetad
        self.yd = yd
        self.xd = xd
        self.thetaddot = thetaddot
        self.phid = phid
        self.vr_prev = 0.
        self.phi_r_prev = 0.
        self.v_c0 = 0.
        self.vc_prev = 0.
        self.prev_time = 0.

    def f(self, state: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
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
        # phir = self.phid[index]
        # thetaddot = self.thetaddot[index]
        # Unpack the state
        x = state.item(0)
        y = state.item(1)
        theta = state.item(2)
        wheel_base = self.wheel_base
        # Calculate the error in orientation and position
        rotation_matrix: npt.NDArray[np.float64] = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        thetadiff: np.float64 = thetar - theta
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
            rotation_matrix: npt.NDArray[np.float64] = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            thetadiff: np.float64 = thetar - theta
            thetadiff = np.arctan(np.sin(thetadiff) / np.cos(thetadiff))
            vec_diff = np.array([x-xr, y-yr, thetadiff])
            vec_e = np.dot(rotation_matrix, vec_diff)
            xe = vec_e.item(0)
            ye = vec_e.item(1)
            thetae = vec_e.item(2)
            angular_accel = self.angular_accel[index]
            v = state.item(3)
            vehicle_accel = self.vr_prev - v
            if np.isclose(t, self.prev_time):
                vehicle_accel = 0
            else:
                # vehicle_accel = vehicle_accel / (t - self.prev_time)
                vehicle_accel = 0
            print(vehicle_accel)
            vc_dot, phi = self.get_sliding_mode_control(
                xe=xe,
                ye=ye,
                thetae=thetae,
                vd=vr,
                omega_d=wr,
                ang_accel_d=angular_accel,
                vc=v,
                vehicle_accel=vehicle_accel)
            
        # Calculate the state derivatives
        xdot = v * np.cos(theta)
        ydot = v * np.sin(theta)
        thetadot = v * np.tan(phi) / wheel_base
        self.vc_prev = vr
        self.prev_time= t
        if self.controller_type == 1:
            xdot = np.array([xdot, ydot, thetadot])
        else:
            xdot = np.array([xdot, ydot, thetadot, vc_dot])
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
        k1, k2, k3 = 1.2, 3.8, 2.4
        v = k1 * xe + vr*np.cos(thetae)
        w = wr + vr*(k2*ye + k3*np.sin(thetae))
        phi_r = np.arctan(self.wheel_base * w / v)
        return v, phi_r

    def get_sliding_mode_control(
            self,
            xe: float,
            ye: float,
            thetae: float,
            vd: float,
            omega_d: float,
            ang_accel_d: float,
            vc: float,
            vehicle_accel:float) -> tuple[float, float]:
        
        thetae = np.arctan(np.sin(thetae) / np.cos(thetae))
        xe_dot = -vd + vc*np.cos(thetae) + ye*omega_d
        ye_dot = vc*np.sin(thetae)-xe*omega_d
        k0, k1, k2 = 1, 1.4, 1
        Q1, Q2 = 1, 1
        P1, P2 = 1, 1
        s1 = xe_dot + k1*xe
        s2 = ye_dot + k2 * ye + k0*np.sign(ye)*thetae
        omega = (omega_d + (vc*np.cos(thetae)+k0*np.sign(ye))**(-1)
                 * (-Q2*s2-P2*np.sign(s2) -
                    k2*ye_dot + ang_accel_d*xe + omega_d*xe - 
                    vehicle_accel*np.sin(thetae)))
        if np.abs(vc) < 1e-3:
            vc = 1
        phi_c = np.arctan(self.wheel_base*omega/vc)
        thetae_dot = omega - omega_d
        # if np.isclose(np.abs(thetae), np.pi/2):
        #     vc_dot = 0
        # else:
        vc_dot = 1/np.cos(thetae)*(-Q1*s1-P1*np.tanh(s1)-k1*xe_dot -
                                omega_d*ye_dot - ang_accel_d*ye +
                                vc*thetae_dot*np.sin(thetae))
        
        return vc_dot, phi_c

    def simulate(self, tspan: npt.NDArray[np.float64]
                 ) -> tuple[npt.NDArray[np.float64],
                            npt.NDArray[np.float64],
                            npt.NDArray[np.float64]]:
        # sol1: npt.NDArray[np.float64] = odeint(self.f, self.x0, tspan)
        self.controller_type = 2
        sol1 = np.array([0])
        x0 = np.concatenate((self.x0, np.array([self.v_c0])))
        x0 = x0.flatten()
        sol2: npt.NDArray[np.float64] = odeint(self.f, x0, tspan)
        return sol1, sol2, tspan
    # def getReward(self,
    #               xe: float,
    #               ye: float,
    #               thetae: float,
    #               vr: float,
    #               vd: float) -> float:
    #     reward=0
    #     if abs(xe) < 0.1:
    #         reward += 1
    #     elif abs(xe) < 0.5:
    #         reward += 0.5
    #     elif abs(xe) < 1:
    #         reward += 0.1
    #     else:
    #         reward -= 1
    #     if abs(ye) < 0.1:
    #         reward += 1
    #     elif abs(ye) < 0.5:
    #         reward += 0.5
    #     elif abs(ye) < 1:
    #         reward += 0.1
    #     else:
    #         reward -= 1
    #     if abs(thetae) < 0.01:
    #         reward += 1
    #     elif abs(thetae) < 0.05:
    #         reward += 0.5
    #     elif abs(thetae) < 0.1:
    #         reward += 0.1
    #     else:
    #         reward -= 2
    #     if abs(vr - vd) < 0.1:
    #         reward += .2
    #     return reward


if __name__ == "__main__":

    # Import the data from the CSV file
    file_total_path = os.path.join(os.path.dirname(
        __file__), '../output_acceleration.csv')
    data = np.genfromtxt(file_total_path, delimiter=',', skip_header=1)
    # Convert angles from degrees to radians
    thetad = data[:, 0] * math.pi / 180
    # x and y are swapped because of the sign convention of theta and theta_dot
    yd = data[:, 2]
    xd = data[:, 1]
    vd = data[:, 3]
    omega_d = data[:, 4]
    t_vec = data[:, 5]
    angular_acceleration = data[:, 6]
    wheel_base = 2.8
    # calculate the given steering angle which can be done because the
    # desired trajectory has a constant linear velocity
    phi_d = np.arctan(wheel_base / vd[0] * omega_d)

    x0 = [xd[0], yd[0], thetad[0], 0]
    # tf=63.8
    tf = 3
    tspan = np.arange(0, tf, 0.01)
    controller_type = 1

    # Create the vehicle object
    vehicle = VehicleSim(x0[0], x0[1], x0[2], wheel_base,
                         vd, thetad, yd, xd, omega_d, phi_d, angular_acceleration)
    # Simulate the vehicle
    x, x2, t = vehicle.simulate(tspan)

    control = np.zeros((len(t), 2))

    # Calculate the control inputs for the vehicle
    # for i in range(len(t)):
    #     time=t[i]
    #     index=int(time * 100) + 2
    #     index=min(index, len(vd) - 1)
    #     index=int(index)
    #     vd_=vd[index]
    #     thetad_=thetad[index]
    #     thetaddot_=omega_d[index]
    #     xd_=xd[index]
    #     yd_=yd[index]
    #     phid_=phi_d[index]
    #     x_=x[i, 0]
    #     y_=x[i, 1]
    #     theta_=x[i, 2]
    #     rotation_matrix=np.array([
    #         [np.cos(theta_), np.sin(theta_), 0],
    #         [-np.sin(theta_), np.cos(theta_), 0],
    #         [0, 0, 1]
    #     ])
    #     thetadiff=thetad_ - theta_
    #     thetadiff=np.arctan(np.sin(thetadiff) / np.cos(thetadiff))
    #     vec_diff=np.array([xd_ - x_, yd_ - y_, thetad_ - theta_])
    #     vec_e=np.dot(rotation_matrix, vec_diff)
    #     xe=vec_e[0]
    #     ye=vec_e[1]
    #     thetae=vec_e[2]
    #     control[i, :]=vehicle.get_lyap_control(
    #         vr=vd_, wr=thetaddot_, thetae=thetae, xe=xe, ye=ye)

    # # Plotting
    # plt.figure(figsize=(10, 20))
    # plt.subplot(5, 1, 1)
    # plt.plot(t, x[:, 0], label='x_r', linewidth=1.5)
    # plt.plot(t_vec, xd, '--', label='x_d', linewidth=1.5)
    # plt.title("Direct Lyapunov Controller: X position vs desired X position ")
    # plt.legend()

    # plt.subplot(5, 1, 2)
    # plt.plot(t, x[:, 1], label='y_r', linewidth=1.5)
    # plt.plot(t_vec, yd, '--', label='y_d', linewidth=1.5)
    # plt.title("Direct Lyapunov Controller: Y position vs desired Y position")
    # plt.legend()

    # plt.subplot(5, 1, 3)
    # plt.plot(t, x[:, 2], label='theta_r', linewidth=1.5)
    # plt.plot(t_vec, thetad, '--', label='theta_d', linewidth=1.5)
    # plt.title("Direct Lyapunov Controller: Orientation vs Desired orientation")
    # plt.legend()

    # plt.subplot(5, 1, 4)
    # plt.plot(t, control[:, 0], label='v', linewidth=1.5)
    # plt.title("Direct Lyapunov: Velocity control")

    # plt.subplot(5, 1, 5)
    # plt.plot(t, control[:, 1], label='phi', linewidth=1.5)
    # plt.title("Direct Lyapunov: Steering control")
    # # space plots so titles don't overlap
    # plt.subplots_adjust(hspace=0.5)

    # fig, ax=plt.subplots(1)
    # ax.plot(xd, yd, '--', label='Desired path', linewidth=1.5, color='black')
    # ax.plot(x[:, 0], x[:, 1], label='Path', linewidth=1.5, color='blue')
    # ax.set_aspect('equal')
    # plt.title("Direct Lyapunov: Path followed by the vehicle")
    # plt.legend()

    control = np.zeros((len(t), 2))

    vc_prev = 0.
    time_prev = 0.  
    # Calculate the control inputs for the vehicle
    for i in range(len(t)):
        time = t[i]
        index = int(time * 100) + 2
        index = min(index, len(vd) - 1)
        index = int(index)
        vd_ = vd[index]
        thetad_ = thetad[index]
        thetaddot_ = omega_d[index]
        xd_ = xd[index]
        yd_ = yd[index]
        phid_ = phi_d[index]
        angular_acc = angular_acceleration[index]
        x_ = x2[i, 0]
        y_ = x2[i, 1]
        theta_ = x2[i, 2]
        vc =  x2[i, 3]
        rotation_matrix = np.array([
            [np.cos(theta_), np.sin(theta_), 0],
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
        vehicle_accel = vc_prev - vc
        if np.isclose(time, time_prev):
            vehicle_accel = 0
        else:
            vehicle_accel = vehicle_accel / (time - time_prev)
        vc_dot, phir = vehicle.get_sliding_mode_control(
            xe=xe, ye=ye, thetae=thetae, vd=vd_, omega_d=thetaddot_,
             ang_accel_d=angular_acc, vc=vc, vehicle_accel=vehicle_accel)
        control[i, :] = [vc, phir]
        vc_prev = vc
        time_prev = time

    # Plotting
    fig, ax = plt.subplots(5, 1)
    # set fig size
    fig.set_size_inches(10, 20)
    ax_ = ax[0]
    ax_.plot(t, x2[:, 0], label='x_r', linewidth=1.5)
    ax_.plot(t_vec, xd, '--', label='x_d', linewidth=1.5)
    ax_.set_title("Sliding Mode Controller: X position vs desired X position ")
    ax_.legend()

    ax_ = ax[1]
    ax_.plot(t, x2[:, 1], label='y_r', linewidth=1.5)
    ax_.plot(t_vec, yd, '--', label='y_d', linewidth=1.5)
    ax_.set_title("Sliding Mode Controller: Y position vs desired Y position")
    ax_.legend()

    ax_ = ax[2]
    ax_.plot(t, x2[:, 2], label='theta_r', linewidth=1.5)
    ax_.plot(t_vec, thetad, '--', label='theta_d', linewidth=1.5)
    ax_.set_title(
        "Sliding Mode Controller: Orientation vs Desired orientation")
    ax_.legend()

    ax_ = ax[3]
    ax_.plot(t, x2[:,3], label='v', linewidth=1.5)
    ax_.set_title("Sliding Mode: Velocity control")

    ax_ = ax[4]
    ax_.plot(t, control[:, 1], label='phi', linewidth=1.5)
    ax_.set_title("Sliding Mode: Steering control")
    # space plots so titles don't overlap
    fig.subplots_adjust(hspace=0.5)

    # plt.subplots_adjust(hspace=0.5)

    fig, ax = plt.subplots(1)
    ax.plot(xd, yd, '--', label='Desired path', linewidth=1.5, color='black')
    ax.plot(x2[:, 0], x2[:, 1], label='Path', linewidth=1.5, color='blue')
    ax.set_aspect('equal')
    plt.title("Sliding Mode: Path followed by the vehicle")
    plt.legend()
    plt.show()
