"""
vpinn_example.py

This script implements a Variational Physics-Informed Neural Network (VPINN)
to solve a 1D Poisson equation. It uses a custom PDE residual based on the
weak form of the equation, approximated via Monte Carlo integration.

Problem Definition:
- PDE: -u_xx(x) = pi^2 * sin(pi * x)
- Domain: x in [-1, 1]
- Boundary Conditions: u(-1) = 0, u(1) = 0
- Analytical Solution: u(x) = sin(pi * x)
"""

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

# 1. Define Geometry
geom = dde.geometry.Interval(-1, 1)

# 2. Define Boundary Conditions
def boundary_func(x, on_boundary):
    return on_boundary

bc = dde.DirichletBC(geom, lambda x: 0, boundary_func)

# 3. Define the Analytical Solution (for comparison)
def analytical_solution(x):
    return np.sin(np.pi * x)

# 4. Define VPINN components
# 4.1 Source function f(x) for the PDE
def pde_source_func(x):
    return torch.pi**2 * torch.sin(torch.pi * x)

# 4.2 Define a set of test functions v(x) and their derivatives v'(x)
# Test functions should be zero at the boundary to satisfy Dirichlet conditions.
def v_0(x):
    return 1 - x**2
def v_0_x(x):
    return -2 * x

def v_1(x):
    return x * (1 - x**2)
def v_1_x(x):
    return 1 - 3 * x**2

test_functions = [(v_0, v_0_x), (v_1, v_1_x)]

# 4.3 Define the VPINN residual function (weak form)
def pde_vpinn(x, u):
    # u is the network output, shape: (batch_size, 1)
    # x is the network input, shape: (batch_size, 1)

    # Get derivatives of the network output u with respect to input x
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    
    # Get source function values for the input points
    f_val = pde_source_func(x)
    
    # Calculate the residual for each test function
    residuals = []
    for v, v_x in test_functions:
        v_val = v(x)
        v_x_val = v_x(x)
        
        # Weak form for -u_xx = f  is  Integral(u_x * v_x - f * v)dx = 0
        integrand = u_x * v_x_val - f_val * v_val
        
        # Approximate the integral over the domain by taking the mean over the batch.
        integral_residual = torch.mean(integrand)
        residuals.append(integral_residual)

    # The final PDE loss is the sum of the squared residuals from all test functions.
    pde_loss = torch.sum(torch.stack(residuals)**2)
    
    # Return the same loss value for each point in the batch.
    return pde_loss * torch.ones_like(u)

# 5. Define the Data object for the VPINN
# We need a large number of domain points for the Monte Carlo integration.
data = dde.data.PDE(
    geom,
    pde_vpinn,
    bc,
    num_domain=500,  # More points for better integral approximation
    num_boundary=2,
    solution=analytical_solution,
    num_test=100
)

# 6. Define the Neural Network
net = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot normal")

# 7. Compile and Train the Model
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=15000)

# 8. Plotting and Results
# Manually plot the loss history since dde.plot is deprecated
loss_train = np.sum(np.array(losshistory.loss_train), axis=1)
loss_test = np.sum(np.array(losshistory.loss_test), axis=1)

plt.figure(figsize=(8, 6))
plt.plot(losshistory.steps, loss_train, 'b-', label='Train Loss')
plt.plot(losshistory.steps, loss_test, 'r--', label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss History for VPINN (1D Poisson)')
plt.legend()
plt.grid(True)
plt.show()

# Compare prediction with analytical solution
X_test = geom.uniform_points(100, True)
_ = model.predict(X_test)
# The predict function requires the model to be in eval mode, which is handled internally.
# To get predictions for plotting, we can call predict again or use the last state.
y_pred = model.predict(X_test)
y_true = analytical_solution(X_test)

plt.figure(figsize=(8, 6))
plt.plot(X_test, y_true, 'b-', label='Analytical Solution')
plt.plot(X_test, y_pred, 'r--', label='VPINN Prediction')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('VPINN vs. Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()

print(f"L2 relative error: {dde.metrics.l2_relative_error(y_true, y_pred):.4e}")
