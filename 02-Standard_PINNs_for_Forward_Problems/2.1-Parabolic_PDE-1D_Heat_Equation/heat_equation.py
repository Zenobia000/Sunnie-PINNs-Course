"""
heat_equation.py

This script solves the 1D heat equation using the DeepXDE library.
The heat equation is a canonical example of a parabolic PDE.

Problem Definition:
- PDE: d(u)/d(t) = alpha * d^2(u)/d(x^2)
- Domain: x in [0, 1], t in [0, 1]
- Initial Condition: u(x, 0) = sin(pi * x)
- Boundary Conditions:
    - u(0, t) = 0
    - u(1, t) = 0
- Thermal Diffusivity (alpha): 0.1 / pi
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# Define the thermal diffusivity
alpha = 0.1 / np.pi

# 1. Define Geometry and Time Domain
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Define the PDE
def pde(x, u):
    """
    Defines the residual of the heat equation.
    x is a 2D tensor where x[:, 0:1] is the spatial coordinate (x)
    and x[:, 1:2] is the temporal coordinate (t).
    u is the neural network's output.
    """
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t - alpha * du_xx

# 3. Define Boundary and Initial Conditions

# Boundary functions to identify points on the boundary
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

# Dirichlet boundary conditions: u(0, t) = 0 and u(1, t) = 0
bc_left = dde.DirichletBC(geomtime, lambda x: 0, boundary_left)
bc_right = dde.DirichletBC(geomtime, lambda x: 0, boundary_right)

# Initial condition: u(x, 0) = sin(pi * x)
ic = dde.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# Define the analytical solution for metric calculation
def analytical_solution(x):
    """Analytical solution for the 1D heat equation."""
    t = x[:, 1:2]
    x_spatial = x[:, 0:1]
    return np.sin(np.pi * x_spatial) * np.exp(-alpha * np.pi**2 * t)

# 4. Assemble the Model
# The data object combines all the problem definitions
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_left, bc_right, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
    solution=analytical_solution,
)

# Define the neural network architecture
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

# Create the DeepXDE model
model = dde.Model(data, net)

# 5. Train and Predict
# Compile the model with the Adam optimizer
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])

# Train the model for 15000 epochs
losshistory, train_state = model.train(iterations=15000, display_every=1000)

# Plot the loss history
plt.figure(figsize=(10, 6))
train_loss = np.array(losshistory.loss_train)
test_loss = np.array(losshistory.loss_test)
steps = losshistory.steps

plt.plot(steps, train_loss, label="Train Loss", color='blue')
plt.plot(steps, test_loss, label="Test Loss", color='orange')

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training and Test Loss History")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Predict the solution for visualization
x_plot = np.linspace(0, 1, 101)
t_plot = np.linspace(0, 1, 101)
X, T = np.meshgrid(x_plot, t_plot)
X_flat = X.flatten()
T_flat = T.flatten()
xt_test = np.vstack((X_flat, T_flat)).T

u_pred = model.predict(xt_test)
U_pred = u_pred.reshape(X.shape)

# Plotting the predicted solution
plt.figure(figsize=(10, 8))
plt.pcolormesh(T, X, U_pred, shading='auto', cmap='hot')
plt.colorbar(label="Temperature u(x,t)")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("PINN Solution of the 1D Heat Equation")
plt.show()
