# 4.4: Discontinuous Solutions - Conservative PINN (cPINN)

## 1. Introduction to the Method

The Conservative Physics-Informed Neural Network (cPINN) is a specialized architecture designed to solve hyperbolic conservation laws, a class of PDEs notorious for developing **discontinuous solutions** (i.e., shocks or sharp fronts), even from smooth initial conditions.

Standard PINNs, which rely on point-wise evaluation of the PDE residual, fail on these problems. The strong form of the PDE requires derivatives that are undefined at the discontinuity, causing the \( \mathcal{L}_{pde} \) to become ill-defined and preventing the network from learning the correct solution. cPINNs overcome this fundamental limitation by reformulating the problem in a way that does not require point-wise differentiability.

## 2. Core Principle (First Principle)

The first principle behind cPINN is the **integral form of conservation laws**, which is more fundamental than the differential (PDE) form. A conservation law states that the rate of change of a conserved quantity \( u \) in a domain \( \Omega \) is equal to the net flux \( \mathbf{F} \) of that quantity across the boundary \( \partial\Omega \). Mathematically:
\[
\frac{d}{dt} \int_{\Omega} u \, d\mathbf{x} = - \oint_{\partial\Omega} \mathbf{F} \cdot \mathbf{n} \, dS
\]
This integral form holds true even when \( u \) is discontinuous. The differential form, \( \frac{\partial u}{\partial t} + \nabla \cdot \mathbf{F} = 0 \), is derived from the integral form using the divergence theorem and assumes that \( u \) and \( \mathbf{F} \) are smooth enough to be differentiable everywhere.

cPINNs leverage this by borrowing a key idea from the classical **Finite Volume Method**:
1.  **Domain Decomposition**: The spatial domain is divided into multiple, non-overlapping subdomains.
2.  **Independent Networks**: A separate, standard PINN is trained within each subdomain.
3.  **Flux Continuity at Interfaces**: The crucial innovation is the introduction of a new loss term that enforces the conservation law in its integral form across the interfaces between subdomains. It ensures that the flux leaving one subdomain is equal to the flux entering the next one. This is enforced via the **Rankine-Hugoniot condition**, a mathematical statement of the conservation law across a discontinuity.

## 3. Implementation with DeepXDE (Fundamentals)

Implementing a cPINN in `DeepXDE` requires a custom model setup, as the standard `dde.Model` is not designed for domain decomposition with multiple independent networks and interface conditions.

A cPINN implementation involves:
1.  **Define Subdomains**: Decompose the main geometry (e.g., an `Interval`) into several smaller `Interval` objects.
2.  **Create a Network for Each Subdomain**: Instantiate a separate `dde.nn.FNN` for each subdomain.
3.  **Define a Custom Loss Function**: This is the most critical part. The total loss is a sum of:
    a.  The standard `TimePDE` loss (PDE residual, BCs, ICs) for *each* subdomain network, calculated only within its own domain.
    b.  **Interface Losses**: For each interface between subdomains, add a loss term that penalizes the mismatch in the flux. For a conservation law \( u_t + F(u)_x = 0 \), the loss at an interface \( x_i \) would be based on the square of the difference: \( (F(u_{left}(x_i, t)) - F(u_{right}(x_i, t)))^2 \), where \( u_{left} \) and \( u_{right} \) are the solutions from the networks on either side of the interface.
4.  **Custom Training Loop**: A custom training loop is needed to manage the multiple models and the complex custom loss function. The optimizer must update the parameters of all subdomain networks simultaneously.

Given this complexity, the `cpinn_example.py` will solve the **inviscid Burgers' equation** (\( u_t + (u^2/2)_x = 0 \)), a classic example that forms a shock. The script will be a simplified, conceptual demonstration of the domain decomposition and interface flux matching ideas.
