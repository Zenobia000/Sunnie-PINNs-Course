# 4.5: First-Order System Formulation (FO-PINN)

## 1. Introduction to the Method

As you correctly pointed out, the First-Order System Physics-Informed Neural Network (FO-PINN) is a common and powerful technique for solving high-order PDEs. It addresses the instability and high computational cost associated with high-order automatic differentiation by reformulating the original PDE into a coupled system of lower-order (often first-order) equations.

This method is also referred to in literature as FOSLS-PINN (First-Order System Least-Squares PINN).

## 2. Core Principle (First Principle)

The core principle of FO-PINN is **system decomposition**. Instead of directly tackling a high-order PDE, we introduce auxiliary variables to represent intermediate derivatives. This breaks down one complex problem into several simpler, coupled problems.

Consider a general 4th-order PDE, like the biharmonic equation from our example:
\[
\nabla^4 u = f(x, y)
\]
A standard PINN would need to compute 4th-order derivatives of the network's output, which can lead to noisy gradients and slow convergence.

The FO-PINN approach introduces an auxiliary variable, \( w \), to represent the Laplacian of \( u \). This allows us to rewrite the single 4th-order equation as a system of two 2nd-order equations:
\[
\begin{cases}
\nabla^2 u - w = 0 \\
\nabla^2 w - f = 0
\end{cases}
\]

### Advantages of the FO-PINN Formulation

1.  **Avoids High-Order Derivatives**: This is the primary benefit. The network now only needs to be twice-differentiable, significantly improving the stability and conditioning of the optimization problem (as discussed in section 4.1).
2.  **Improved Gradient Flow**: Training on lower-order systems often leads to better-behaved gradients and a smoother loss landscape.
3.  **Richer Information**: The network learns not just the solution `u`, but also its intermediate derivatives (like `w = nabla^2 u`), which can provide additional physical insights.

## 3. Implementation with DeepXDE (Fundamentals)

Implementing an FO-PINN in `DeepXDE` is quite straightforward:

1.  **Multi-Output Network**: The neural network must be defined with multiple outputs, one for the primary variable (\( u \)) and one for each auxiliary variable (\( w \)).
2.  **System of PDEs**: The `pde` function must return a list of residuals, one for each equation in the coupled system.
3.  **Boundary Conditions for All Variables**: It is crucial to provide boundary conditions for *all* output variables (\( u \) and \( w \)). Sometimes, the boundary condition for the auxiliary variable can be directly derived from the original problem's BCs.

The accompanying `fo_pinn_example.py` script demonstrates this exact process for solving the 4th-order biharmonic equation.
