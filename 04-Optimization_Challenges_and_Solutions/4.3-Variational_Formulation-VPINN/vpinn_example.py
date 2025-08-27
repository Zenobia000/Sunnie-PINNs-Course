"""
vpinn_example.py

This script conceptually demonstrates the advantage of the Variational PINN (VPINN)
approach by solving a 4th-order PDE. A standard PINN would require 4th-order
derivatives, which is computationally expensive and difficult for a neural network.

Instead, we will solve the equivalent weak form of the PDE, which only contains
2nd-order derivatives. By using a standard PINN to solve this reformulated problem,
we highlight the core benefit of VPINNs (lower derivative order) without needing a full
custom implementation of numerical integration.

Problem Definition: The Biharmonic Equation
- PDE (Strong Form, 4th-order): nabla^4 u = f(x, y)
- PDE (Weak Form, 2nd-order): integral(nabla^2 u * nabla^2 v) dx = integral(f * v) dx
  For a PINN, we can enforce the inner part of the weak form: nabla^2 u = w, and nabla^2 w = f.
  This system of two 2nd-order PDEs is much easier to solve.

- Domain: Omega = [-1, 1] x [-1, 1]
- Source function: f(x, y) = -4 * pi^4 * sin(pi*x) * sin(pi*y)
- Analytical solution: u(x, y) = sin(pi*x) * sin(pi*y)
- Boundary Conditions:
    - u = 0 on the boundary
    - nabla^2 u = 0 on the boundary (simplified from du/dn=0 for this analytical solution)
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define Geometry
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

# Source function f(x, y)
def source_function(x):
    return -4 * (np.pi**4) * np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

# 2. Define the PDE system (Weak Form Analogue)
# We introduce an auxiliary variable w = nabla^2 u.
# The 4th-order PDE nabla^4 u = f becomes a system of two 2nd-order PDEs:
# PDE1: nabla^2 u - w = 0
# PDE2: nabla^2 w - f = 0
# The network will have two outputs: u (output 0) and w (output 1).
def pde_system(x, y):
    """
    Defines the system of two 2nd-order PDEs.
    y is the network output, where y[:, 0:1] is u and y[:, 1:2] is w.
    """
    u, w = y[:, 0:1], y[:, 1:2]
    
    # Laplacian of u
    laplacian_u = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
    # Laplacian of w
    laplacian_w = dde.grad.hessian(w, x, i=0, j=0) + dde.grad.hessian(w, x, i=1, j=1)
    
    f = source_function(x)
    
    pde1 = laplacian_u - w
    pde2 = laplacian_w - f
    
    return [pde1, pde2]

# Analytical solution for validation
def analytical_solution(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

# Analytical for w = nabla^2 u
def analytical_w(x):
    # nabla^2(sin(pi*x)sin(pi*y)) = -2*pi^2*sin(pi*x)sin(pi*y)
    return -2 * (np.pi**2) * np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

def boundary_func(x, on_boundary):
    return on_boundary

# 3. Define Boundary Conditions
# We have two variables (u, w) and thus need BCs for both.
# BC for u: u = 0 on the boundary
bc_u = dde.DirichletBC(geom, lambda x: 0, boundary_func, component=0)
# BC for w: w = nabla^2 u = 0 on the boundary
bc_w = dde.DirichletBC(geom, lambda x: 0, boundary_func, component=1)

# 4. Assemble and Train the Model
data = dde.data.PDE(
    geom,
    pde_system,
    [bc_u, bc_w],
    num_domain=2500,
    num_boundary=100,
    solution=analytical_solution,
    num_test=1000,
)

# Network architecture: 2 inputs (x,y), 4 hidden layers, 2 outputs (u,w)
net = dde.nn.FNN([2] + [64] * 4 + [2], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000)

# 5. Visualize the Solution
X = np.linspace(-1, 1, 100)
Y = np.linspace(-1, 1, 100)
X_grid, Y_grid = np.meshgrid(X, Y)
xy_test = np.vstack((X_grid.flatten(), Y_grid.flatten())).T

# Predict u and w
y_pred = model.predict(xy_test)
u_pred = y_pred[:, 0].reshape(100, 100)

u_analytical = analytical_solution(xy_test).reshape(100, 100)

l2_error = dde.metrics.l2_relative_error(u_analytical, u_pred)
print(f"L2 relative error for u: {l2_error:.4e}")

# Plotting
fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(X_grid, Y_grid, u_pred, cmap='hot')
ax1.set_title('VPINN Concept Solution (u)')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(X_grid, Y_grid, u_analytical, cmap='hot')
ax2.set_title('Analytical Solution (u)')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(X_grid, Y_grid, np.abs(u_pred - u_analytical), cmap='viridis')
ax3.set_title('Absolute Error')
plt.suptitle("Biharmonic Equation solved via Weak Form Concept", fontsize=16)
plt.show()
