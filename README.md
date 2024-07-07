# Comparing Non-Linear Trajectory Following Controllers (Non-Linear Traditional Controllers vs. Reinforcement Learning)

## Introduction

Autonomous driving has gained significant popularity in recent years. This trend has underscored the need for comprehensive control methods. In this project, I will be comparing and contrasting several nonlinear control for an autonomous vehicle to track time-varying trajectories. The control techniques I will be comparing are of two varieties of controllers traditional nonlinear controllers and reinforcement learing based controllers.

## Dynamics of the Vehicle

For the purposes of this project, the dynamics of the vehicle are modeled as a simple kinematic bicycle model. The state of the vehicle is defined as follows:

```math

state = \begin{bmatrix} x \\ y \\ \theta \end{bmatrix}

```

The dynamics of the vehicle are defined as follows:

```math

 \begin{cases} \dot{x}_{r}=v_{r}\cos(\theta_{r})\\ 
 \dot{y}_{r}=v_{r}\sin(\theta_{r})\\ 
 \dot{\theta}_{r}=\frac{v_{r}}{l}\tan(\phi_{r}) \end{cases}
```

Where `v_r` is the velocity of the vehicle, `l` is the length of the vehicle, and `phi_r` is the steering angle of the vehicle. The inputs into this system are the velocity of the vehicle and the steering angle of the vehicle. A trajectory can be thought of as a series of waypoints that an ideal vehicle is following. Each of these waypoints has a desired position, $(x_d, y_d)$, and a desired orientation, $\theta_d$. These waypoints are the desired trajectory that the vehicle must follow.

An example of the true and desired vehicles for following a single waypoint is shown below:

![Single Waypoint](figures/bike_model.png)

## Traditional Nonlinear Controllers

Two types of nonliner controllers are implemented in this project. The first controller is a lypunov based controller and the second controller is a feedback linearization controller.

### Error Dynamics

To derive these controllers the error dynamics for a vehicle following a waypoint must be defined. The error is the difference between the real vehicle position and this virtual vehicle. This error model is defined to be in the frame which is orthogonal to the path pane.

```math
\begin{bmatrix} 
x_{e}\\ 
y_{e}\\ 
\theta_{e} 
\end{bmatrix}=
\begin{bmatrix} 
\cos\theta_{d} & \sin\theta_{d} & 0\\ 
-\sin\theta_{d} & \cos\theta_{d} & 0\\ 
0 & 0 & 1 
\end{bmatrix}
\begin{bmatrix} 
x_{r}-x_{d}\\ 
y_{r}-y_{d}\\ 
\theta_{r}-\theta_{d} 
\end{bmatrix} 
```

The dynamics of this error model can be expressed as

```math
\begin{cases} 
\dot{x}_{e}=v_{r}+\omega_{r}y_{e}-v_{d}\cos(\theta_{e})\\ 
\dot{y}_{e}=-\omega_{r}x_{e}+v_{d}\sin(\theta_{e})\\
\dot{\theta}_{e}=\omega_{r}-\omega_{d} 
\end{cases} 
```

Where

```math
\omega_r = \dot{\theta_r} 
```

and

```math
\omega_d = \dot{\theta_d} 
```
