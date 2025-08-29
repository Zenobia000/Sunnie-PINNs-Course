"""
bpinn_example.py

This script implements a Bayesian Physics-Informed Neural Network (BPINN)
to solve a 1D Poisson equation and quantify the associated uncertainty.

It uses a Probabilistic Feed-forward Neural Network (PFNN) from DeepXDE,
which outputs both the mean and the standard deviation of the solution.

Problem Definition:
- PDE: -u_xx(x) = pi^2 * sin(pi * x)
- Domain: x in [-1, 1]
- Boundary Conditions: u(-1) = 0, u(1) = 0
- Analytical Solution: u(x) = sin(pi * x)
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. Define the PDE and analytical solution
def pde_func(x, y):
    """Standard strong-form PDE residual for -u_xx = f(x)"""
    # y has two components: y[:, 0:1] is the mean, y[:, 1:2] is the log(std)
    u_mean = y[:, 0:1]
    
    # We only apply the PDE to the mean of the prediction
    du_xx = dde.grad.hessian(u_mean, x, i=0, j=0)
    
    # Use torch functions for GPU compatibility
    source_term = torch.pi**2 * torch.sin(torch.pi * x)
    
    return -du_xx - source_term

def analytical_solution(x):
    return np.sin(np.pi * x)

# 2. Define Geometry and Boundary Conditions
geom = dde.geometry.Interval(-1, 1)

# The boundary condition is applied to the mean of the prediction
bc = dde.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary, component=0)

# 3. Define the BPINN Model
# We use a Probabilistic FNN (PFNN). It has 2 outputs: mean and log(variance)
# Note: DeepXDE's PFNN outputs log(variance), not log(std). std = sqrt(exp(log_var))
net = dde.nn.PFNN([1] + [20] * 3 + [2], "tanh", "Glorot normal")

# 4. Define the Data object
# For BPINNs, the loss is the negative log-likelihood, which is handled by default
data = dde.data.PDE(
    geom,
    pde_func,
    bc,
    num_domain=20,
    num_boundary=2,
    solution=analytical_solution,
    num_test=100,
)

# 5. Compile and Train the Model
model = dde.Model(data, net)
# The default loss for a probabilistic network in DeepXDE is the negative log-likelihood
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000)

# 6. Plotting and Results
# Manually plot the loss history
loss_train = np.sum(np.array(losshistory.loss_train), axis=1)
loss_test = np.sum(np.array(losshistory.loss_test), axis=1)

plt.figure(figsize=(8, 6))
plt.plot(losshistory.steps, loss_train, 'b-', label='Train Loss')
plt.plot(losshistory.steps, loss_test, 'r--', label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss History for BPINN (1D Poisson)')
plt.legend()
plt.grid(True)
plt.show()

# 7. Uncertainty Quantification and Visualization
X_test = geom.uniform_points(200, True)
y_pred = model.predict(X_test)
y_true = analytical_solution(X_test)

mean_pred = y_pred[:, 0]
log_var_pred = y_pred[:, 1]
std_dev_pred = np.sqrt(np.exp(log_var_pred))

plt.figure(figsize=(10, 7))
plt.plot(X_test, y_true, 'b-', label='Analytical Solution', linewidth=2)
plt.plot(X_test, mean_pred, 'r--', label='BPINN Mean Prediction', linewidth=2)

# Plot the uncertainty region (e.g., 2 standard deviations)
plt.fill_between(
    X_test.flatten(), 
    mean_pred - 2 * std_dev_pred, 
    mean_pred + 2 * std_dev_pred, 
    color='orange', 
    alpha=0.3, 
    label='Uncertainty (2 std. dev.)'
)

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('BPINN Prediction with Uncertainty Quantification')
plt.legend()
plt.grid(True)
plt.show()

# Report L2 relative error on the mean prediction
l2_error = dde.metrics.l2_relative_error(y_true, mean_pred.reshape(-1, 1))
print(f"L2 relative error on mean: {l2_error:.4e}")
