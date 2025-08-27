# 3.1: Inverse Problem - Heat Equation Parameter Discovery

## 1. Problem Statement (First Principle)

This example demonstrates how to solve an inverse problem using a PINN. We revisit the 1D heat equation, but with a crucial difference: we assume that the thermal diffusivity, \( \alpha \), is an **unknown constant** that we want to discover.

The governing PDE remains the same:
\[
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
\]
The core idea is to leverage sparse measurements of the temperature field \( u(x, t) \) to infer the value of \( \alpha \). This mimics a real-world scenario where a material's properties are unknown and must be determined from experimental data.

**Problem Setup**:
1.  **Generate "Experimental" Data**: We cannot perform a real experiment, so we will create synthetic data. First, we solve the *forward* problem using a known, true value for the thermal diffusivity, let's say \( \alpha_{true} = 0.1 / \pi \).
2.  **Sample Sparse Data**: From the solution of the forward problem, we will sample a small number of data points at random locations in the spatio-temporal domain. These points will serve as our "sensor measurements". Optionally, we can add some random noise to these measurements to simulate experimental error.
3.  **Define the Inverse Problem**: The goal is to build a new PINN that is given only the governing equation, the boundary/initial conditions, and the sparse data points. This new PINN will treat \( \alpha \) as a learnable variable and try to find its value.

## 2. Implementation with DeepXDE (Fundamentals)

The `heat_inverse.py` script will implement this entire workflow. The key new component introduced by `DeepXDE` for this task is `dde.Variable`.

1.  **Define the Unknown Parameter**:
    -   We define our unknown thermal diffusivity \( \alpha \) as a `dde.Variable`. We initialize it with a guess value (e.g., 1.0), which is different from the true value.
    -   `dde.Variable(1.0)` creates a TensorFlow/PyTorch variable that will be updated during the training process via gradient descent, just like the network's weights.

2.  **Modify the PDE Definition**:
    -   The `pde(x, u)` function is defined as before, but now the coefficient \( \alpha \) is the `dde.Variable` we just created, not a fixed Python constant.

3.  **Incorporate Observational Data**:
    -   The sparse data points we generated are fed into the `dde.data.TimePDE` object. `DeepXDE` provides a specific type of boundary condition, `dde.PointSetBC`, which is used to define a set of training points that are not on the boundary or initial slice. These points will be used to compute the data loss term \( \mathcal{L}_{data} \).

4.  **Assemble and Train the Model**:
    -   The model is assembled as usual.
    -   During training, the optimizer now minimizes a loss function that depends on both the network weights \( \theta \) and the unknown parameter \( \alpha \). The optimizer will simultaneously try to:
        a.  Find a function \( u_{\theta} \) that satisfies the PDE for the *current* estimate of \( \alpha \).
        b.  Find a value for \( \alpha \) that makes the function \( u_{\theta} \) best fit the provided observational data.

5.  **Check the Result**:
    -   `DeepXDE` provides a callback mechanism (`dde.callbacks.VariableValue`) to monitor the value of the `dde.Variable` (\( \alpha \)) during training.
    -   After training is complete, we can check the final learned value of \( \alpha \) and compare it to the true value (\( \alpha_{true} \)) to verify the success of the inverse problem solution.
