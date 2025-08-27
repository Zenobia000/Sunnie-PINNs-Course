# 2.1: Parabolic PDE - 1D Heat Equation with DeepXDE

## 1. Problem Statement (First Principle)

The one-dimensional heat equation is a canonical parabolic Partial Differential Equation (PDE) that describes heat diffusion in a medium. It is a mathematical formulation of the first principle of **conservation of energy**.

We aim to find the temperature distribution \( u(x, t) \) over a spatial domain \( x \in [0, L] \) and a time domain \( t \in [0, T] \). The governing equation is:

\[
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
\]

where \( \alpha \) is the thermal diffusivity of the medium.

To ensure a unique solution, we must specify boundary and initial conditions:

-   **Initial Condition (IC)**: The initial temperature distribution at \( t=0 \). For this problem, we'll define it as:
    \[
    u(x, 0) = f(x)
    \]
-   **Boundary Conditions (BCs)**: The temperature at the spatial boundaries. We will use Dirichlet boundary conditions, where the temperature is fixed at the ends of the domain:
    \[
    u(0, t) = u_0, \quad u(L, t) = u_L
    \]

## 2. Implementation with DeepXDE (Fundamentals)

The `heat_equation.py` script demonstrates how to solve this problem using the `DeepXDE` library. The fundamental steps are:

1.  **Define Geometry and Time Domain**:
    -   We use `dde.geometry.Interval` to define the spatial domain \( x \in [0, L] \).
    -   We use `dde.geometry.TimeDomain` to define the time domain \( t \in [0, T] \).
    -   These are combined into a `dde.geometry.GeometryXTime` object, which represents the full spatio-temporal domain.

2.  **Define the PDE**:
    -   We create a function `pde(x, u)` that represents the residual of the heat equation: \( \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0 \).
    -   `DeepXDE` automatically handles the computation of the derivatives \( u_t \) and \( u_{xx} \) from the neural network output `u` with respect to the inputs `x` (which contains both spatial and temporal coordinates).

3.  **Define Boundary and Initial Conditions**:
    -   We use the `dde.DirichletBC` class to enforce the fixed temperatures at the boundaries. We need to define functions that identify the boundary points (e.g., `on_boundary_left`, `on_boundary_right`).
    -   We use the `dde.IC` class to enforce the initial temperature distribution, providing the initial condition function \( f(x) \).

4.  **Assemble the Model**:
    -   We create a `dde.data.TimePDE` object, passing in the geometry, the PDE function, the boundary/initial conditions, and parameters for sampling collocation points.
    -   We define the neural network architecture (e.g., a standard feed-forward network).
    -   Finally, we create a `dde.Model` object, passing the `TimePDE` data and the network definition.

5.  **Train and Predict**:
    -   We compile the model with an optimizer (e.g., "adam") and a learning rate.
    -   We call `model.train()` to start the training process.
    -   After training, we use `model.predict()` to obtain the solution on a test grid for visualization.
