# 5.3: Example - Causal PINN (Conceptual)

## 1. Introduction to the Example

This module provides a **conceptual code skeleton** for implementing a **Causal PINN** using the **Causal Loss Weighting** strategy. As discussed in the theory section, this approach provides an efficient "curriculum" for time-dependent problems by forcing the model to prioritize learning the solution at earlier times before moving to later times.

## 2. Problem Statement: 1D Inviscid Burgers' Equation

We will use the 1D Inviscid Burgers' equation as our test case, as it's a time-evolving problem that benefits from a causal training approach.
-   **PDE:** \( \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0 \)
-   **Domain:** \( x \in [-1, 1] \), \( t \in [0, 1] \)
-   **IC:** \( u(x, 0) = -\sin(\pi x) \)
-   **BC:** \( u(-1, t) = u(1, t) = 0 \)

## 3. The Fundamental Implementation: Modifying the PDE Residual

The key to implementing causal weighting in `DeepXDE` is to modify the standard `pde` residual function.

1.  **Define a Weighting Function**: We first define a function \( w(t) \) that increases as time \( t \) goes from 0 to the final time \( T \). A common choice is an exponential weight:
    \[
    w(t) = \exp(-\lambda(T-t))
    \]
    When this weight multiplies the squared residual, it effectively makes the loss contribution from points at later times smaller at the beginning of training. As the optimizer minimizes the dominant, early-time loss, the network learns the initial behavior first.

2.  **Apply Weight in `pde` Function**: Inside the `pde(x, y)` function provided to `DeepXDE`, we extract the time component \( t \) from the input tensor `x`. We then compute the standard PDE residual and simply return `w(t) * residual`. `DeepXDE` squares this entire output for the loss calculation, effectively weighting the squared residual by \( w(t)^2 \).

The following Python script, `causal_pinn_example.py`, provides a runnable example of this concept. It sets up the Burgers' equation problem and applies a causal weight to the PDE loss, demonstrating how this simple modification can be implemented.
