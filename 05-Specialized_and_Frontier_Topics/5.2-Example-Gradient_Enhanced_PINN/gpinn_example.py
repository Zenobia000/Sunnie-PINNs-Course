"""
gpinn_example.py

This script implements a Gradient-Enhanced PINN (gPINN) to demonstrate how
leveraging derivative data can improve model accuracy.

Problem Definition:
- PDE: du/dx = cos(2*pi*x)
- Domain: x in [0, 1]
- Initial Condition: u(0) = 0
- Analytical Solution: u(x) = sin(2*pi*x) / (2*pi)
- Analytical Derivative: du/dx = cos(2*pi*x)

We will provide sparse data for both u(x) and du/dx to the model.
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# 1. Define Geometry and PDE
geom = dde.geometry.Interval(0, 1)

def pde(x, u):
    du_dx = dde.grad.jacobian(u, x, i=0, j=0)
    return du_dx - dde.backend.cos(2 * np.pi * x)

# Analytical solution and derivative for generating data
def analytical_solution(x):
    return np.sin(2 * np.pi * x) / (2 * np.pi)

def analytical_derivative(x):
    return np.cos(2 * np.pi * x)

# 2. Generate and Format Observational Data
num_u_data = 10
num_grad_data = 10

observe_x_u = np.linspace(0, 1, num_u_data).reshape(-1, 1)
observe_u = analytical_solution(observe_x_u)

observe_x_grad = np.linspace(0, 1, num_grad_data).reshape(-1, 1)
observe_grad = analytical_derivative(observe_x_grad)

# 3. Define Data Constraints
def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc = dde.DirichletBC(geom, lambda x: 0, boundary)

# Standard PointSetBC for the solution u(x)
observe_u_bc = dde.PointSetBC(observe_x_u, observe_u, component=0)

# PointSetOperatorBC for the derivative du/dx
def du_dx_operator(x, u, _):
    return dde.grad.jacobian(u, x, i=0, j=0)

observe_grad_bc = dde.PointSetOperatorBC(
    observe_x_grad, observe_grad, du_dx_operator
)

# --- 4a. Assemble and Train gPINN Model ---
print("--- Training gPINN Model (with gradient data) ---")
data_gpinn = dde.data.PDE(
    geom,
    pde,
    [bc, observe_u_bc, observe_grad_bc],  # Include all constraints
    num_domain=100,
    num_boundary=2,
    num_test=100
)
net_gpinn = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot normal")
model_gpinn = dde.Model(data_gpinn, net_gpinn)
model_gpinn.compile("adam", lr=1e-3)
model_gpinn.train(iterations=15000)


# --- 4b. Assemble and Train Standard PINN Model (for comparison) ---
print("\n--- Training Standard PINN Model (without gradient data) ---")
data_std = dde.data.PDE(
    geom,
    pde,
    [bc, observe_u_bc],  # Only use IC and u(x) data
    num_domain=100,
    num_boundary=2,
    num_test=100
)
net_std = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot normal")
model_std = dde.Model(data_std, net_std)
model_std.compile("adam", lr=1e-3)
model_std.train(iterations=15000)


# --- 5. Visualize and Compare ---
x_test = np.linspace(0, 1, 200).reshape(-1, 1)
u_analytical_test = analytical_solution(x_test)
u_gpinn_pred = model_gpinn.predict(x_test)
u_std_pred = model_std.predict(x_test)

l2_error_gpinn = dde.metrics.l2_relative_error(u_analytical_test, u_gpinn_pred)
l2_error_std = dde.metrics.l2_relative_error(u_analytical_test, u_std_pred)

print(f"\nL2 relative error (gPINN): {l2_error_gpinn:.4e}")
print(f"L2 relative error (Standard PINN): {l2_error_std:.4e}")

plt.figure(figsize=(12, 8))
plt.plot(x_test, u_analytical_test, 'k-', label='Analytical Solution', linewidth=2)
plt.plot(observe_x_u, observe_u, 'ko', label='u(x) Data Points', markersize=8)
plt.plot(x_test, u_std_pred, 'r--', label=f'Standard PINN (Error: {l2_error_std:.2e})', linewidth=2)
plt.plot(x_test, u_gpinn_pred, 'b--', label=f'gPINN (Error: {l2_error_gpinn:.2e})', linewidth=2)
plt.title('gPINN vs. Standard PINN Performance')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()
