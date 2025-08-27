# 2.3: Nonlinear PDE - 1D Burgers' Equation with DeepXDE

## 1. Problem Statement (First Principle)

The Burgers' equation is a fundamental nonlinear Partial Differential Equation (PDE) that appears in various areas of applied mathematics, including fluid mechanics, nonlinear acoustics, and traffic flow. It is one of the simplest PDEs that captures the interplay between **nonlinear convection** and **linear diffusion**.

We aim to find the solution \( u(x, t) \) governed by the following equation:

\[
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}
\]

where \( \nu \) is the viscosity coefficient.

The key challenge and the defining characteristic of this equation is the nonlinear convection term \( u \frac{\partial u}{\partial x} \). In this term, the solution \( u \) itself acts as the convection velocity, meaning the wave speed depends on the amplitude of the wave. This self-interaction can lead to the formation of steep gradients and shock-like structures, even from smooth initial conditions.

For this example, we will solve the equation on the domain \( x \in [-1, 1] \) and \( t \in [0, 1] \), with the following conditions:
-   **Initial Condition (IC)**: A sine wave to observe how it steepens over time.
    \[
    u(x, 0) = -\sin(\pi x)
    \]
-   **Boundary Conditions (BCs)**: We will use Dirichlet boundary conditions to pin the solution at the boundaries.
    \[
    u(-1, t) = 0, \quad u(1, t) = 0
    \]
-   **Viscosity**: We will use a small viscosity \( \nu = 0.01 / \pi \) to allow the nonlinear effects to be prominent.

## 2. Implementation with DeepXDE (Fundamentals)

The `burgers_equation.py` script will demonstrate how `DeepXDE` handles this nonlinear problem. The implementation process is very similar to the previous examples, highlighting the library's versatility.

1.  **Define Geometry and Time Domain**: We use `dde.geometry.Interval` and `dde.geometry.TimeDomain` to define the spatio-temporal domain, combining them into a `dde.geometry.GeometryXTime` object.

2.  **Define the PDE**: The core of this example is defining the nonlinear residual function `pde(x, u)`:
    \[
    \text{residual} = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
    \]
    `DeepXDE` makes this straightforward. The term \( u \frac{\partial u}{\partial x} \) is implemented simply by multiplying the network's output `u` with its automatically computed derivative `dde.grad.jacobian(u, x, i=0, j=0)`. This seamless handling of nonlinearities is a key advantage of the PINN framework.

3.  **Define Boundary and Initial Conditions**: We use `dde.DirichletBC` and `dde.IC` as in the heat equation example to apply the boundary and initial constraints.

4.  **Assemble and Train the Model**: We assemble a `dde.data.TimePDE` object and a `dde.Model`. The training process then seeks to minimize the combined loss from the PDE residual, the boundary conditions, and the initial condition.

5.  **Visualize the Solution**: The solution \( u(x, t) \) is a 2D surface. We will visualize it using a heatmap and also plot snapshots of \( u(x) \) at different time steps (e.g., t=0.25, 0.5, 0.75, 1.0) to clearly see the wave steepening and diffusion effects.
