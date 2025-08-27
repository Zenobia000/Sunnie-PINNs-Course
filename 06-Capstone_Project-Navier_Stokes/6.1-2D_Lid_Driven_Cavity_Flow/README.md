# 6.1: 2D Lid-Driven Cavity Flow

## 1. Problem Statement (First Principle)

This submodule contains the implementation for solving the 2D steady-state, incompressible Navier-Stokes equations for the classic lid-driven cavity benchmark problem. This problem is a cornerstone of computational fluid dynamics, used to validate new numerical methods.

**Physical Setup**:
-   A square domain \( \Omega = [0, 1] \times [0, 1] \) is filled with a fluid of density \( \rho=1 \) and kinematic viscosity \( \nu \).
-   The left, right, and bottom walls are stationary.
-   The top wall (lid) moves with a uniform horizontal velocity \( U = 1 \).

This setup forces the fluid into a recirculating motion, forming a primary vortex. The non-dimensional form of the problem is governed by the Reynolds number, \( Re = \frac{\rho U L}{\mu} = \frac{U L}{\nu} \), where \( L=1 \) is the characteristic length. For this problem, we will set \( Re = 100 \), which implies \( \nu = 1/100 = 0.01 \).

**Governing Equations (as implemented)**:
We will solve for the velocity components \( u(x, y), v(x, y) \) and the pressure \( p(x, y) \). The network will have three outputs.
1.  **Continuity**: \( u_x + v_y = 0 \)
2.  **x-Momentum**: \( u u_x + v u_y = -p_x + \nu (u_{xx} + u_{yy}) \)
3.  **y-Momentum**: \( u v_x + v v_y = -p_y + \nu (v_{xx} + v_{yy}) \)

**Boundary Conditions (BCs)**:
The no-slip condition applies on the stationary walls, and the lid velocity is prescribed on the top wall.
-   **Left wall** (\( x=0 \)): \( u=0, v=0 \)
-   **Right wall** (\( x=1 \)): \( u=0, v=0 \)
-   **Bottom wall** (\( y=0 \)): \( u=0, v=0 \)
-   **Top wall** (lid, \( y=1 \)): \( u=1, v=0 \)

The pressure field is only determined up to an arbitrary constant. To ensure a unique solution, we must pin the pressure at one point. A common choice is to set \( p=0 \) at the origin \( (0, 0) \).

## 2. Implementation with DeepXDE (Fundamentals)

The `navier_stokes.py` script will implement the solution. This is the most complex model we will build, as it involves a system of three coupled PDEs with three output variables.

1.  **Define Geometry**: A `dde.geometry.Rectangle` for the square domain.

2.  **Define the PDE System**:
    -   We define a function `pde_system(x, Y)` where `Y` is the network's output tensor containing \( [u, v, p] \).
    -   The function will return a list of three tensors, representing the residuals for the continuity, x-momentum, and y-momentum equations, respectively.
    -   This requires computing multiple first- and second-order derivatives of the different output components. For example, \( u_x \) is `dde.grad.jacobian(Y, x, i=0, j=0)`, while \( p_y \) is `dde.grad.jacobian(Y, x, i=2, j=1)`.

3.  **Define Boundary and Initial Conditions**:
    -   We need multiple `dde.DirichletBC` objects to set the \( u \) and \( v \) velocities on all four walls. This requires defining four boundary identification functions (e.g., `on_boundary_left`, `on_boundary_top`, etc.).
    -   A single `dde.PointSetBC` is used to pin the pressure \( p(0, 0) = 0 \).

4.  **Assemble and Train**:
    -   The `dde.data.PDE` object is created with the list of all BCs.
    -   The neural network will be defined to have 2 inputs (\( x, y \)) and 3 outputs (\( u, v, p \)).
    -   The model is compiled and trained. Solving this complex, coupled, nonlinear system is computationally intensive and may require a large number of training iterations.

5.  **Visualize**: After training, we will plot the learned velocity field (\( u, v \)) as a quiver plot overlaid on a heatmap of the velocity magnitude. We will also plot the learned pressure field \( p \). These visualizations are standard practice in CFD and will allow us to observe the characteristic vortex structure of the flow.
