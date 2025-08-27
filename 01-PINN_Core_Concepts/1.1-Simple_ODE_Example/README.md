# 1.1: Simple ODE Example - A First Principles Implementation

## 1. Problem Statement (First Principle)

From first principles, an Ordinary Differential Equation (ODE) describes the relationship between a function and its derivatives. For this introductory example, we consider one of the most fundamental ODEs which models phenomena like exponential decay:

\[
\frac{dy}{dx} + y = 0
\]

To obtain a unique solution, we must provide an auxiliary constraint. We impose an initial condition:

\[
y(0) = 1
\]

This is a well-posed initial value problem. For validation purposes, we note its analytical solution is \( y(x) = e^{-x} \). Our goal is to train a neural network to discover this solution purely from the differential equation and the initial condition.

## 2. Implementation Fundamentals

The Python script `simple_ode.py` will translate the core concepts from the parent module into a concrete implementation. The fundamental steps are:

1.  **Neural Network Definition**: A simple feed-forward neural network is defined using `torch.nn.Module`. This network, \( y_{\theta}(x) \), will serve as our candidate solution. It takes a single scalar \( x \) as input and outputs a single scalar \( y \).

2.  **Loss Function Construction**: We will build a composite loss function \( \mathcal{L}(\theta) \) with two components:
    *   **ODE Residual Loss (\( \mathcal{L}_{ode} \))**: This loss enforces the governing equation. We define the residual function \( r(x) = \frac{dy_{\theta}}{dx} + y_{\theta}(x) \). The loss is the Mean Squared Error (MSE) of this residual, i.e., \( \mathbb{E}[r(x_i)^2] \), computed over a set of collocation points \( \{x_i\} \) sampled from the domain. The crucial step, computing \( \frac{dy_{\theta}}{dx} \), is handled precisely using `torch.autograd.grad`.
    *   **Initial Condition Loss (\( \mathcal{L}_{ic} \))**: This loss anchors the solution. It is simply the MSE of the prediction at the initial point: \( (y_{\theta}(0) - 1)^2 \).

3.  **Training**: An optimizer (e.g., Adam) is used to find the network parameters \( \theta \) that minimize the total loss \( \mathcal{L}_{total} = \mathcal{L}_{ode} + \mathcal{L}_{ic} \).

4.  **Validation & Visualization**: After training, the network's output \( y_{\theta}(x) \) is plotted against the analytical solution \( y(x) = e^{-x} \) to visually and quantitatively assess the accuracy of the PINN.
