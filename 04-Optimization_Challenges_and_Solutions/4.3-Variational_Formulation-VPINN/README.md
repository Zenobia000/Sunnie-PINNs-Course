# 4.3: Variational Formulation - Variational PINN (VPINN)

## 1. Introduction to the Method

The Variational Physics-Informed Neural Network (VPINN) is another advanced PINN variant that addresses the optimization challenges of stiff PDEs, particularly those involving high-order derivatives. Instead of modifying the loss weighting scheme like SA-PINN, VPINN reformulates the PDE problem itself from its **strong form** into a **weak form**.

This approach is deeply rooted in classical numerical methods like the Finite Element Method (FEM) and is designed to create a smoother, better-conditioned loss landscape for the optimizer to navigate.

## 2. Core Principle (First Principle)

The first principle behind VPINN is the **calculus of variations**. It states that solving a differential equation in its strong (point-wise) form is often equivalent to finding a function that minimizes a certain integral functional.

### Strong Form vs. Weak Form

-   **Strong Form (used in standard PINNs)**: Requires the PDE to hold true at every single point in the domain. For a PDE \( \mathcal{N}[u] = f \), the residual is \( r = \mathcal{N}[u] - f \). This requires the neural network solution \( u_{\theta} \) to be differentiable to a high enough order (e.g., a 2nd-order PDE requires a twice-differentiable network).

-   **Weak Form (used in VPINNs)**: The strong form is multiplied by an arbitrary, smooth "test function" \( v \) and then integrated over the entire domain \( \Omega \). The requirement is that this integral equation holds for *all* possible test functions.
    \[
    \int_{\Omega} (\mathcal{N}[u] - f) v \, d\Omega = 0, \quad \forall v \in \mathcal{V}
    \]
    The key step is applying **integration by parts** (or Green's identity in higher dimensions). This process effectively "transfers" a derivative from the solution \( u \) to the test function \( v \). For example, for a 2nd-order PDE, the weak form might only contain first-order derivatives of \( u \).

### Advantages of the Weak Form

1.  **Lower-Order Derivatives**: The primary advantage is that the weak form reduces the order of derivatives required of the neural network solution. For a 2nd-order PDE, \( u_{\theta} \) may only need to be once-differentiable, which is much easier for a neural network to learn.
2.  **Smoother Loss Landscape**: By avoiding high-order derivatives, we mitigate the high-frequency amplification issue discussed in section 4.1. This leads to a better-conditioned Hessian and a smoother loss landscape, which is easier for gradient-based optimizers to handle.
3.  **Natural Handling of Boundary Conditions**: Neumann (derivative) boundary conditions are incorporated more naturally into the weak form formulation.

## 3. Implementation with DeepXDE (Fundamentals)

Implementing a true VPINN in `DeepXDE` requires significant customization because the library is primarily built around the strong-form, point-wise residual. The core challenge is the need to compute integrals over the domain for the loss function, rather than just summing squared residuals at collocation points.

A typical VPINN implementation involves:
1.  **Choosing a Set of Test Functions**: The weak form must hold for all test functions. In practice, a finite set of basis functions (e.g., polynomials, Fourier series) is chosen for \( v \).
2.  **Numerical Integration**: The integral in the weak form must be computed numerically. This is usually done using numerical quadrature methods, such as Gaussian quadrature, which requires sampling the domain at specific quadrature points.
3.  **Custom Loss Function**: The loss function is the sum of the squared integral residuals for each test function in the chosen set.
    \[
    \mathcal{L}_{vpin} = \sum_{i=1}^{M} \left( \int_{\Omega} (\nabla u_{\theta} \cdot \nabla v_i - f v_i) \, d\Omega \right)^2
    \]
    (This is an example for the Poisson equation after integration by parts).

Due to the complexity of implementing numerical integration within `DeepXDE`'s standard workflow, the accompanying `vpinn_example.py` will focus on conceptually illustrating the idea. It will solve a problem where the VPINN formulation is particularly advantageous and highlight the differences in the required derivatives compared to a standard PINN.
