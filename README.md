# Comparing Non-Linear Trajectory Following Controllers (Non-Linear Traditional Controllers vs. Reinforcement Learning)

## Introduction

Autonomous driving has gained significant popularity in recent years. This trend has underscored the need for comprehensive control methods. In this project, I will be comparing and contrasting several nonlinear control techniques for an autonomous vehicle to track time-varying trajectories. The control techniques I will be comparing are of two varieties traditional nonlinear control techniques and reinforcement learing techniques.

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

The first category is traditional nonlinear controllers. In this project two controllers of this type were leveraged. The first
