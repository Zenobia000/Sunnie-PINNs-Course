# Part 2: Standard PINNs for Forward Problems (Knowledge Area)

## 1. Introduction to the Knowledge Area

This module transitions from the manual, "from-scratch" implementation of Module 1 to the application of a high-level PINN library, `DeepXDE`. The focus here is on solving **Forward Problems**, which form the bulk of traditional scientific computing tasks.

A forward problem is one where the governing equations and all necessary parameters (e.g., boundary conditions, material properties) are known, and the goal is to find the system's state (i.e., the solution to the PDE).

## 2. The Role of DeepXDE (Fundamentals)

In the first module, we manually constructed the loss functions and managed the automatic differentiation calls. This provided crucial insight into the inner workings of a PINN. However, for more complex problems, this approach becomes tedious and error-prone.

`DeepXDE` is a library designed to abstract away this boilerplate code. Its fundamental components allow us to define a PDE problem declaratively, mirroring its mathematical formulation:

-   `Geometry`: Defines the spatial and temporal domain of the problem (e.g., `Interval`, `Rectangle`, `TimeDomain`).
-   `PDE`: A function where you define the PDE's residual, similar to our manual implementation but within the library's structure.
-   `Boundary/Initial Conditions`: Classes like `DirichletBC`, `NeumannBC`, and `IC` are used to specify constraints on the solution.
-   `Data` & `Model`: These classes assemble all the components into a trainable PINN model.

By using `DeepXDE`, we shift our focus from the mechanics of PINN implementation to the physics of the problem definition.

## 3. Knowledge Body for this Module (BoK)

This module will cover three canonical types of PDEs, each representing a different class of physical phenomena:

1.  **Parabolic PDEs (Time-dependent, diffusive processes)**
    -   **Example**: `2.1-Parabolic_PDE-1D_Heat_Equation`
    -   **Concept**: We will model how a temperature profile evolves over time.

2.  **Elliptic PDEs (Steady-state problems)**
    -   **Example**: `2.2-Elliptic_PDE-2D_Poisson_Equation`
    -   **Concept**: We will solve for a steady-state potential field on a 2D domain, focusing on the application of boundary conditions.

3.  **Nonlinear PDEs (Problems with solution-dependent coefficients)**
    -   **Example**: `2.3-Nonlinear_PDE-1D_Burgers_Equation`
    -   **Concept**: We will tackle the Burgers' equation, which includes a nonlinear convection term (`u * du/dx`), a common challenge in fluid dynamics.

## 4. Next Steps

We will now proceed sequentially through the sub-modules, starting with the 1D Heat Equation, to see how `DeepXDE` simplifies the process of solving these fundamental PDEs.
