# Part 1: PINN Core Concepts (Knowledge Area)

## 1. Introduction to the Knowledge Area

This module establishes the foundational Body of Knowledge (BoK) for understanding Physics-Informed Neural Networks. It addresses the fundamental paradigm shift from traditional numerical methods to deep learning-based solvers for differential equations. As a graduate-level introduction, we will cover the conceptual underpinnings that make PINNs a powerful tool for scientific machine learning.

## 2. Core Principles (First Principles)

The entire concept of PINNs is built upon two first principles:

1.  **Physical Laws as Universal Constraints**: The behavior of physical systems is governed by differential equations (e.g., Navier-Stokes, Heat Equation). These equations are not merely models; they are mathematical expressions of fundamental physical laws (e.g., conservation of mass, energy, momentum). A valid solution to a physical problem *must* satisfy these equations.
2.  **Neural Networks as Universal Function Approximators**: The Universal Approximation Theorem states that a feed-forward neural network with a single hidden layer can approximate any continuous function to an arbitrary degree of accuracy. A PINN leverages this by postulating that the solution to a differential equation, \( u(\mathbf{x}, t) \), is a continuous function that can be represented by a neural network, \( u_{\theta}(\mathbf{x}, t) \).

The innovation of PINNs is the fusion of these two principles directly within the training process.

## 3. Fundamental Concepts & Terminology (Fundamentals)

### 3.1. The Paradigm Shift: From Discretization to Differentiation

-   **Traditional Solvers (FEM/FDM/FVM)**: These methods discretize the problem domain into a mesh or grid. They solve for the function's values at discrete points. The solution is inherently discrete and interpolation is required to find values between points.
-   **PINNs**: This approach is mesh-free. The neural network \( u_{\theta}(\mathbf{x}, t) \) is a continuous function, valid across the entire spatio-temporal domain. The "solution" is found not by solving for values at points, but by optimizing the network's parameters \( \theta \) to find the best continuous function representation.

### 3.2. The PINN Loss Function: A Multi-Objective Optimization

The training of a PINN is governed by a composite loss function, which enforces all necessary physical and data constraints simultaneously.

\[
\mathcal{L}(\theta) = w_{pde}\mathcal{L}_{pde} + w_{bc}\mathcal{L}_{bc} + w_{ic}\mathcal{L}_{ic} + w_{data}\mathcal{L}_{data}
\]

-   **Physics Residual Loss \( \mathcal{L}_{pde} \)**: This is the core of the "physics-informed" approach. For a PDE given by \( \mathcal{N}[u] = f \), the residual is defined as \( r = \mathcal{N}[u_{\theta}] - f \). The loss is typically the Mean Squared Error (MSE) of this residual over a set of collocation points sampled within the domain.
-   **Boundary/Initial Condition Loss (\( \mathcal{L}_{bc} \), \( \mathcal{L}_{ic} \))**: These terms enforce the specific constraints of the problem. They are calculated as the MSE between the network's predictions and the known values at the boundaries and initial time.
-   **Data Loss (\( \mathcal{L}_{data} \))**: (Optional) This term anchors the solution to any available experimental or observational data points.

### 3.3. Automatic Differentiation (AD): The Enabling Technology

-   **What it is**: AD is a computational technique that, given a function defined by a sequence of elementary operations, calculates its exact derivatives. It is not symbolic differentiation (which can lead to expression swell) nor numerical differentiation (which introduces approximation errors).
-   **Why it's critical**: To compute the PDE residual \( r \), we need to evaluate differential operators (e.g., \( \frac{\partial u}{\partial t} \), \( \frac{\partial^2 u}{\partial x^2} \)) on the network's output \( u_{\theta} \). AD, a core feature of frameworks like PyTorch, allows us to compute these derivatives analytically and efficiently, making the entire PINN methodology computationally feasible.

## 4. Next Steps

With this theoretical foundation, we will now proceed to the first hands-on example in the `1.1-Simple_ODE_Example` sub-module. There, we will translate these concepts into code to solve a simple but illustrative differential equation.
