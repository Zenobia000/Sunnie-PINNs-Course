# 2.2: Elliptic PDE - 2D Poisson Equation with DeepXDE

## 1. Problem Statement (First Principle)

The Poisson equation is a canonical elliptic Partial Differential Equation (PDE) that arises in numerous fields of physics, including electrostatics, gravitational potential theory, and steady-state heat conduction. It describes systems that have reached a time-independent equilibrium.

We aim to find the potential field \( u(x, y) \) over a two-dimensional spatial domain \( \Omega \). The governing equation is:

\[
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)
\]

where \( f(x, y) \) is a known source function.

For this example, we will solve the Poisson equation on a square domain \( \Omega = [-1, 1] \times [-1, 1] \). To ensure a unique solution, we must specify boundary conditions on the entire boundary \( \partial\Omega \). We will use Dirichlet boundary conditions, where the value of \( u \) is fixed on all four sides of the square.

Let's define a source function and corresponding boundary conditions such that the analytical solution is known, allowing for validation. A common choice is:
-   Source function: \( f(x, y) = -2\pi^2 \sin(\pi x) \sin(\pi y) \)
-   Analytical solution: \( u_{analytical}(x, y) = \sin(\pi x) \sin(\pi y) \)

From the analytical solution, we can derive the Dirichlet boundary conditions:
-   On boundaries \( x = \pm 1 \) or \( y = \pm 1 \), the value of \( u \) is \( \sin(\pm\pi) \sin(\pi y) = 0 \) or \( \sin(\pi x) \sin(\pm\pi) = 0 \). Thus, \( u(x, y) = 0 \) on \( \partial\Omega \).

## 2. Implementation with DeepXDE (Fundamentals)

The `poisson_equation.py` script will solve this 2D problem using `DeepXDE`. The key steps are:

1.  **Define Geometry**:
    -   We use `dde.geometry.Rectangle` to define the 2D spatial domain \( \Omega = [-1, 1] \times [-1, 1] \). Unlike the previous time-dependent problem, there is no `TimeDomain` here.

2.  **Define the PDE**:
    -   We create a function `pde(x, u)` that represents the residual: \( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} - f(x, y) = 0 \).
    -   The second-order derivatives are computed automatically by `DeepXDE` using `dde.grad.hessian`.

3.  **Define Boundary Conditions**:
    -   We use `dde.DirichletBC` to apply the condition \( u(x, y) = 0 \) on the entire boundary.
    -   `DeepXDE`'s geometry objects have a convenient `on_boundary` method that can be passed directly to the `DirichletBC` to identify all boundary points, simplifying the setup for this problem.

4.  **Assemble and Train the Model**:
    -   We create a `dde.data.PDE` object (note: not `TimePDE` as the problem is steady-state).
    -   We define the neural network architecture. The input layer will now have 2 neurons (for x and y), and the output layer remains 1 (for u).
    -   We create and compile a `dde.Model` as before.
    -   The model is trained to minimize the combined PDE residual loss and the boundary condition loss.

5.  **Visualize the Solution**:
    -   After training, we predict the solution on a 2D grid of points.
    -   The result is visualized as a heatmap or a 3D surface plot and compared against the analytical solution to assess accuracy.
