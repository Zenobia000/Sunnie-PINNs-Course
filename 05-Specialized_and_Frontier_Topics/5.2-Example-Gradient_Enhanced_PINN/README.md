# 5.2: Example - Gradient-Enhanced PINN (gPINN)

## 1. Introduction to the Example

This example provides a practical implementation of a **Gradient-Enhanced Physics-Informed Neural Network (gPINN)**. We will demonstrate how incorporating observational data of a solution's derivative can significantly improve the model's accuracy, especially when the data for the solution itself is sparse or noisy.

## 2. Problem Statement (First Principle)

We will solve a simple Ordinary Differential Equation (ODE) for which we can easily generate both solution data and gradient data. Consider the ODE:
\[
\frac{du}{dx} = \cos(2\pi x)
\]
with an initial condition \( u(0) = 0 \).

The analytical solution is \( u(x) = \frac{\sin(2\pi x)}{2\pi} \), and its derivative is, of course, \( \frac{du}{dx} = \cos(2\pi x) \).

**The gPINN Setup**:
1.  We will assume we have a few sparse "sensor" measurements of the solution \( u(x) \).
2.  Crucially, we will also assume we have a few sparse measurements of the gradient \( \frac{du}{dx} \).
3.  Our goal is to build a PINN that leverages **both** sets of measurements to find the solution.

The first principle here is that the additional gradient data provides a stronger, more direct constraint on the "slope" of the learned function, preventing the network from finding solutions that might fit the sparse \( u(x) \) data points but have incorrect physical behavior (i.e., wrong derivatives) between those points.

## 3. Implementation with DeepXDE (Fundamentals)

The `gpinn_example.py` script will show how to implement this. The key is to provide `DeepXDE` with the gradient data and associate it with the correct derivative of the network's output.

1.  **Define the PDE and Geometry**: This is standard. We define the geometry and the PDE residual \( \frac{du}{dx} - \cos(2\pi x) = 0 \).

2.  **Generate and Format Observational Data**:
    -   We will create two sets of observation points: `observe_x_u` for the solution and `observe_x_grad` for the gradient.
    -   We compute the corresponding true values \( u_{true} \) and \( (\frac{du}{dx})_{true} \) at these points.

3.  **Define Data Constraints**: This is the key step.
    -   The constraints on \( u(x) \) are provided using a standard `dde.PointSetBC` as we did in the inverse problem module.
    -   To provide constraints on the derivative, we need to tell `DeepXDE` that the data corresponds to a derivative of the output. This is done by defining a function that computes the required derivative and passing it to a `dde.PointSetOperatorBC`.
        -   The function, say `du_dx(x, u)`, will use `dde.grad.jacobian` to compute the derivative of the network output `u` with respect to its input `x`.
        -   `dde.PointSetOperatorBC` takes the observation points `observe_x_grad`, the true gradient values, and this derivative function `du_dx`.

4.  **Assemble and Train**:
    -   All constraints (IC, BCs, `PointSetBC` for u, and `PointSetOperatorBC` for du/dx) are collected into a list.
    -   The model is assembled and trained as usual. The loss function will now automatically include a term for the error in `u` and another term for the error in `du/dx`.

5.  **Visualize and Compare**: We will compare the gPINN's performance against a standard PINN that only uses the data for \( u(x) \), demonstrating the accuracy improvement gained from the gradient information.
