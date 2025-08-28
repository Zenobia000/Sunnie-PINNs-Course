"""
poisson_equation.py

This script solves the 2D Poisson equation using the DeepXDE library.
The Poisson equation is a canonical example of an elliptic PDE.

Problem Definition:
- PDE: d^2(u)/d(x^2) + d^2(u)/d(y^2) = f(x, y)
- Source function: f(x, y) = -2 * pi^2 * sin(pi * x) * sin(pi * y)
- Domain: x in [-1, 1], y in [-1, 1]
- Boundary Conditions: u(x, y) = 0 on the entire boundary.
- Analytical Solution: u(x, y) = sin(pi * x) * sin(pi * y)
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style for plots
sns.set_style("whitegrid")

# 1. Define Geometry
geom = dde.geometry.Rectangle([-1, -1], [1, 1])

# 2. Define the PDE
def pde(x, u):
    """
    Defines the residual of the Poisson equation.
    x is a 2D tensor where x[:, 0:1] is x and x[:, 1:2] is y.
    """
    # Laplacian: d^2(u)/d(x^2) + d^2(u)/d(y^2)
    laplacian = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
    
    # Source function f(x, y)
    x_coord, y_coord = x[:, 0:1], x[:, 1:2]
    source = -2 * np.pi**2 * dde.backend.sin(np.pi * x_coord) * dde.backend.sin(np.pi * y_coord)
    
    return laplacian - source

# 用於驗證的解析解
def analytical_solution(x):
    # 根據輸入類型，選擇使用 np.sin 或 dde.backend.sin
    # dde.data.PDE 會傳入 NumPy 陣列，而模型內部會使用 Tensors
    # When called from dde.data.PDE, x is a NumPy array.
    sin = np.sin
    return sin(np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])

# 3. Define Boundary Conditions
# The on_boundary method of the geometry object conveniently identifies all boundary points.
bc = dde.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)

# 4. Assemble and Train the Model
data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=2500,
    num_boundary=100,
    solution=analytical_solution,
    num_test=1000,
)

# Network architecture: 2 inputs (x, y), 3 hidden layers of 50 neurons, 1 output (u)
net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

# Compile with Adam optimizer and a learning rate of 1e-3
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])

# Train the model for 20000 iterations
losshistory, train_state = model.train(iterations=20000, display_every=1000)

# Plot loss history
dde.plot.plot_loss_history(losshistory)
plt.title("Loss History for 2D Poisson Equation")
plt.show()

# 5. Visualize the Solution
# Create a grid for prediction
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
xy_test = np.vstack((X.flatten(), Y.flatten())).T

# Predict on the grid
u_pred = model.predict(xy_test)
u_pred_grid = u_pred.reshape(100, 100)

# Get analytical solution on the grid
u_analytical_grid = analytical_solution(xy_test).reshape(100, 100)

# Calculate L2 relative error
l2_error = dde.metrics.l2_relative_error(u_analytical_grid, u_pred_grid)
print(f"L2 relative error: {l2_error:.4e}")

# Plotting
fig = plt.figure(figsize=(18, 5))

# PINN Solution
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(X, Y, u_pred_grid, cmap='hot')
ax1.set_title('PINN Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Analytical Solution
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(X, Y, u_analytical_grid, cmap='hot')
ax2.set_title('Analytical Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Error
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
error = np.abs(u_pred_grid - u_analytical_grid)
ax3.plot_surface(X, Y, error, cmap='viridis')
ax3.set_title('Absolute Error')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

plt.suptitle("2D Poisson Equation Results", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
