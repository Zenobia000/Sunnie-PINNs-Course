"""
optimization_strategy_example.py

This script demonstrates the two-stage optimization strategy (Adam + L-BFGS)
for training Physics-Informed Neural Networks (PINNs). We will solve a simple
first-order ODE and compare the accuracy of a model trained only with Adam
versus a model trained with Adam followed by L-BFGS fine-tuning.

Problem Definition:
- ODE: dy/dx + y = 0
- Initial Condition: y(0) = 1
- Domain: x in [0, 5]
- Analytical Solution: y(x) = exp(-x)
"""

import deepxde as dde
import numpy as np

# 1. Define the Problem

# Define the computational geometry
geom = dde.geometry.Interval(0, 5)

# Define the ODE residual
def ode(x, y):
    dy_x = dde.grad.jacobian(y, x)
    return dy_x + y

# Define the initial condition
ic = dde.IC(geom, lambda x: 1, lambda _, on_initial: on_initial)

# Define the analytical solution for accuracy validation
def analytical_solution(x):
    return np.exp(-x)

# 2. Setup the PINN Model
# The data object combines the geometry, ODE, and conditions.
# We will create two separate models to demonstrate the two training strategies.
data = dde.data.PDE(
    geom,
    ode,
    ic,
    num_domain=100,
    num_boundary=2,
    solution=analytical_solution,
    num_test=1000,
)

# Define the neural network architecture
net = dde.nn.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")

# --- Strategy 1: Train with Adam only ---
print("--- Training Strategy 1: Adam Optimizer Only ---")
model_adam = dde.Model(data, net)

# Compile with Adam optimizer
model_adam.compile("adam", lr=1e-3, metrics=["l2 relative error"])

# Train the model
losshistory_adam, train_state_adam = model_adam.train(iterations=10000, display_every=2000)

final_error_adam = losshistory_adam.metrics_test[-1][0]
print(f"Final L2 relative error (Adam only): {final_error_adam:.4e}\n")


# --- Strategy 2: Train with Adam + L-BFGS ---
print("--- Training Strategy 2: Adam + L-BFGS Optimizers ---")
# We need to re-initialize the network weights for a fair comparison
net_reinitialized = dde.nn.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")
model_combined = dde.Model(data, net_reinitialized)

# Stage 1: Compile and train with Adam optimizer
print("Stage 1: Training with Adam...")
model_combined.compile("adam", lr=1e-3, metrics=["l2 relative error"])
losshistory_combined, train_state_combined = model_combined.train(iterations=10000, display_every=2000)

# Stage 2: Re-compile and fine-tune with L-BFGS optimizer
print("\nStage 2: Fine-tuning with L-BFGS...")
model_combined.compile("L-BFGS")
losshistory_lbfgs, train_state_lbfgs = model_combined.train()

final_error_combined = losshistory_lbfgs.metrics_test[-1][0]
print(f"Final L2 relative error (Adam + L-BFGS): {final_error_combined:.4e}\n")

# 3. Compare Results
print("--- Comparison of Final L2 Relative Errors ---")
print(f"Adam Only Strategy:      {final_error_adam:.4e}")
print(f"Adam + L-BFGS Strategy:  {final_error_combined:.4e}")

improvement = (final_error_adam - final_error_combined) / final_error_adam
print(f"\nAccuracy improvement with L-BFGS: {improvement:.2%}")

# The output will clearly show that the L-BFGS fine-tuning stage significantly
# reduces the final error, often by one or more orders of magnitude.
