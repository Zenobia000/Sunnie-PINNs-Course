# 4.2: Adaptive Weights - Self-Adaptive PINN (SA-PINN)

## 1. Introduction to the Method

The Self-Adaptive Physics-Informed Neural Network (SA-PINN) is an advanced PINN variant designed to directly address the **gradient imbalance** pathology discussed in the previous section. Instead of relying on manual, static weighting factors for the different loss components, the SA-PINN introduces trainable, adaptive weights that are optimized concurrently with the main network's parameters.

This transforms the standard minimization problem into a more complex **saddle-point optimization problem**, which can lead to more stable and accurate training for stiff PDEs.

## 2. Core Principle (First Principle)

The first principle behind SA-PINNs is rooted in optimization theory, specifically **Lagrangian mechanics and duality**. The standard PINN loss is a simple sum:
\[
\mathcal{L} = \mathcal{L}_{pde} + \mathcal{L}_{bc}
\]
If \( \mathcal{L}_{bc} \) is much larger than \( \mathcal{L}_{pde} \), its gradients will dominate. The SA-PINN reformulates this by introducing learnable, non-negative weights (Lagrange multipliers), \( \lambda_i \), for each loss term. For simplicity, let's consider just two terms:

\[
\mathcal{L}(\theta, \lambda_{pde}, \lambda_{bc}) = \lambda_{pde}\mathcal{L}_{pde}(\theta) + \lambda_{bc}\mathcal{L}_{bc}(\theta)
\]

The goal is now to solve a minimax (or saddle-point) problem:
\[
\max_{\lambda} \min_{\theta} \mathcal{L}(\theta, \lambda)
\]
-   The inner loop **minimizes** the loss with respect to the network parameters \( \theta \), as usual.
-   The outer loop **maximizes** the loss with respect to the weights \( \lambda \).

This creates a dynamic where if a particular loss term (e.g., \( \mathcal{L}_{pde} \)) is large, its corresponding weight \( \lambda_{pde} \) will be increased by the maximization step. This, in turn, increases the gradient contribution of that term in the next minimization step, forcing the network to pay more "attention" to it. This process automatically balances the influence of each loss term throughout training.

## 3. Implementation with DeepXDE (Fundamentals)

`DeepXDE` provides built-in support for several adaptive weighting schemes, though a pure implementation of the original SA-PINN paper's saddle-point optimization requires a custom training loop. However, we can implement the core idea of trainable weights using `dde.Variable` and a custom loss function.

For a more direct approach, `DeepXDE` includes the `dde.callbacks.PDEPointResampler` which, while not a direct implementation of SA-PINN, addresses a similar issue by adaptively resampling collocation points in regions with high PDE residuals.

A simplified SA-PINN implementation would follow these steps:
1.  **Define Learnable Weights**: Create `dde.Variable` objects for the weights of the PDE loss and each of the boundary/initial condition losses.
2.  **Define Custom Loss Function**: Create a custom loss function that takes the individual loss components (which can be computed by `dde`) and combines them using the learnable weights.
3.  **Define Custom Training Loop**: This is the most complex part. A standard `model.train()` call only performs minimization. To solve the minimax problem, one would need to create a custom training loop that alternates between:
    a.  A gradient *descent* step for the network parameters \( \theta \).
    b.  A gradient *ascent* step for the loss weights \( \lambda \).
4.  **Monitor Convergence**: During training, monitor not only the loss but also the values of the adaptive weights to see how they evolve and balance the different training objectives.

In the accompanying `sa_pinn_example.py`, we will demonstrate a practical way to implement a simplified version of this adaptive weighting scheme to highlight the concept.
