"""
causal_pinn_example.py

This script provides a conceptual and runnable demonstration of a 
Causality-Informed PINN using the Causal Loss Weighting strategy.

We solve the 1D Inviscid Burgers' equation. Instead of using a standard
PDE residual, we multiply it by a time-dependent weight to encourage
the network to learn the solution sequentially in time.

Problem Definition:
- PDE: d(u)/d(t) + u * d(u)/d(x) = 0
- Domain: x in [-1, 1], t in [0, 1]
- Initial Condition: u(x, 0) = -sin(pi * x)
- Boundary Conditions: u(-1, t) = u(1, t) = 0
"""

import deepxde as dde
import numpy as np
import torch

# 1. Define the Causal Weighting Function
# The weight w(t) should increase from t=0 to t=T.
# We use a simple linear ramp for this example, but exponential is also common.
T = 1.0  # Final time
def causal_weight(x):
    t = x[:, 1:2]
    # Linear weight: starts small at t=0 and grows to 1 at t=T
    weight = t / T
    return weight

# 2. Define the PDE with Causal Weighting
def pde_causal(x, u):
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    
    residual = du_t + u * du_x
    
    # Apply the causal weight to the residual
    weight = causal_weight(x)
    
    # The loss will be the mean squared of this weighted residual
    return weight * residual

# 3. Standard Problem Setup (Geometry, IC, BC)
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# 4. Assemble and Train the Model
# The setup is identical to a standard PINN, except for the custom PDE function.
data = dde.data.TimePDE(
    geomtime,
    pde_causal,  # Use the causally-weighted PDE function
    [bc, ic],
    num_domain=2500,
    num_boundary=100,
    num_initial=100,
)

net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Using L-BFGS for demonstration as it can be sensitive to loss landscape
model.compile("adam", lr=1e-3)
model.train(iterations=5000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# Note: Visualization is omitted as this is a conceptual code example.
# The key takeaway is the modification of the pde function.
print("Causal PINN training finished.")
print(f"Final loss: {losshistory.loss_train[-1]}")
