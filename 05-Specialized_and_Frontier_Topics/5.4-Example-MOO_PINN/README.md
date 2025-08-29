# 5.4: Example - MOO PINN (Conceptual Skeleton)

## 1. Introduction to the Example

This module provides a **conceptual code skeleton** for implementing a PINN using a **Multi-Objective Optimization (MOO)** strategy. Specifically, it illustrates the logic behind the **GradNorm** algorithm.

As discussed in the theory section, standard PINNs often struggle to balance the different components of their loss function (PDE, BCs, ICs, etc.). GradNorm addresses this by dynamically adjusting the weights of each loss term to ensure that they all learn at a similar rate, preventing any single term from dominating and stalling the training process.

## 2. Why a Conceptual Skeleton?

Implementing GradNorm (and most other MOO techniques) requires a **custom training loop**. This is because we need to perform several steps that are not exposed in the standard `dde.Model.train()` API:
1.  Calculate the loss for each objective *separately*.
2.  Compute the gradient of each weighted loss term with respect to the network parameters (or a subset of them).
3.  Calculate the norms of these gradients.
4.  Update the loss weights based on these norms and a learning rate for the weights themselves.
5.  Finally, perform the standard backpropagation step using the newly updated weights.

Because this goes beyond the scope of a standard DeepXDE implementation, the `moo_pinn_example.py` script is provided as a **non-runnable, PyTorch-style pseudo-code**. Its purpose is to clearly illustrate the algorithmic steps of GradNorm in a familiar deep learning syntax.

## 3. The GradNorm Algorithm (Fundamentals)

The core idea is to balance the training rates of different tasks (loss terms) by ensuring their gradient magnitudes are of a similar order.

The algorithm proceeds as follows in each training step:
1.  **Compute Individual Losses**: Calculate \( \mathcal{L}_{PDE}, \mathcal{L}_{BC}, \dots \)
2.  **Compute Gradient Norms**: For each loss term \( \mathcal{L}_i \), compute the L2 norm of its gradient with respect to a chosen set of network weights \( \theta' \) (typically the last layer): \( G_i = ||\nabla_{\theta'} (w_i \mathcal{L}_i)||_2 \).
3.  **Compute Average Gradient Norm**: \(\bar{G} = \mathbb{E}[G_i]\).
4.  **Compute Task-Specific Learning Rates**: Determine the relative inverse training rate for each task: \( r_i = \frac{L_i}{L_{avg}} \). This indicates how fast each task is learning relative to the average.
5.  **Define GradNorm Loss**: Create a separate loss, \( \mathcal{L}_{grad} \), that measures the disparity between each task's gradient norm and the average norm, weighted by its learning rate:
    \[
    \mathcal{L}_{grad} = \sum_i | G_i - \bar{G} \times r_i^\alpha |
    \]
    Here, \( \alpha \) is a hyperparameter that controls the strength of the restoring force pulling tasks back to a common training rate.
6.  **Update Weights**: The loss weights \( w_i \) are now treated as trainable parameters. We compute the gradient of \( \mathcal{L}_{grad} \) with respect to each \( w_i \) and update them via gradient descent.
7.  **Update Network Parameters**: Finally, the total loss \( \mathcal{L}_{total} = \sum w_i \mathcal{L}_i \) is backpropagated to update the main network parameters \( \theta \).

The provided Python script will lay out these steps in a clear, commented structure.
