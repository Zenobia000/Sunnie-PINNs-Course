# 5.2: Example - Gradient-Enhanced PINN (gPINN)

## 1. Introduction to the Example

This example provides a practical implementation of a **Gradient-Enhanced Physics-Informed Neural Network (gPINN)**. We will demonstrate how incorporating observational data of a solution's derivative can significantly improve a model's accuracy, especially when the data for the solution itself is sparse.

## 2. The First Principle: Enforcing the Correct Slope

The core idea of a gPINN is that physical laws often directly govern a system's **derivatives, fluxes, or strains** (e.g., Fourier's law for heat flux, \( q = -k \nabla T \)). Standard PINNs primarily constrain the solution's *value*. A gPINN enforces an additional, powerful constraint: the solution's *slope* must also be correct.

By providing the model with sparse measurements of the gradient, we provide a much stronger constraint on the learned function's behavior between data points, leading to a more physically robust solution.

**Our Problem Setup**:
-   We will solve a simple ODE: \( \frac{du}{dx} = \cos(2\pi x) \) with \( u(0) = 0 \).
-   The analytical solution is \( u(x) = \frac{\sin(2\pi x)}{2\pi} \).
-   We assume we have a few sparse "sensor" measurements of both the solution \( u(x) \) and its gradient \( \frac{du}{dx} \).
-   The goal is to show that the PINN leveraging **both** sets of measurements is significantly more accurate than one using only \( u(x) \) data.

## 3. Implementation with DeepXDE: `PointSetOperatorBC`

The key to implementing a gPINN in DeepXDE is the `dde.PointSetOperatorBC`. While a standard `dde.PointSetBC` constrains the network's direct output, `PointSetOperatorBC` constrains the output of a differential operator applied to the network.

**The Fundamental Steps**:
1.  **Define the PDE and Geometry**: This is standard. We define the geometry and the PDE residual \( \frac{du}{dx} - \cos(2\pi x) = 0 \).

2.  **Generate Observational Data**: We create two sparse datasets: one for the solution `(observe_x_u, observe_u)` and one for the gradient `(observe_x_grad, observe_grad)`.

3.  **Define Data Constraints**: This is the crucial step.
    -   The solution data is constrained using a standard `dde.PointSetBC`.
    -   The gradient data is constrained using `dde.PointSetOperatorBC`. This requires defining a function, `du_dx_operator`, that uses `dde.grad.jacobian` to compute the derivative of the network's output. `PointSetOperatorBC` then forces this computed derivative to match our observational gradient data.

4.  **Assemble and Train**: We create two models for comparison:
    -   **gPINN**: Includes all constraints: IC, `PointSetBC` for u, and `PointSetOperatorBC` for du/dx.
    -   **Standard PINN**: Includes only the IC and `PointSetBC` for u.

5.  **Visualize and Compare**: By plotting both solutions against the analytical truth and comparing their L2 errors, we can quantify the significant accuracy improvement gained from the gradient information.

The key takeaway, in short: **"A correct value isn't enough; the slope must also be correct."**

## 4. When are gPINNs Most Effective?

This technique is particularly powerful in scenarios such as:
-   **Inverse Problems**: When using internal sensor data (e.g., strain gauges, heat flux sensors) to determine material properties or unknown source terms.
-   **Coupled-Field Problems**: For multi-physics simulations (e.g., thermo-mechanical stress) where the coupling between fields is defined by gradients.
-   **Boundary Layers / High-Gradient Regions**: Where accurately capturing steep changes in the solution is critical for modeling the physics correctly (e.g., viscous fluid flow near a surface).
