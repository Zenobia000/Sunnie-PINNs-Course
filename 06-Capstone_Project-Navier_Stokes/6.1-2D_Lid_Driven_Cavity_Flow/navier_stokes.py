"""
navier_stokes.py

This script solves the 2D steady-state Navier-Stokes equations for the
lid-driven cavity flow problem using the DeepXDE library.

This is the capstone project for the course, combining concepts such as:
- Solving a system of coupled, nonlinear PDEs.
- Handling multiple output variables (u, v, p).
- Defining complex boundary conditions on different parts of the domain.

Problem Definition:
- Domain: [0, 1] x [0, 1]
- Reynolds Number (Re): 100  (nu = 1/Re = 0.01)
- Equations:
    1. Continuity: u_x + v_y = 0
    2. x-Momentum: u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)
    3. y-Momentum: u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)
- Boundary Conditions:
    - u=0, v=0 on left, right, bottom walls.
    - u=1, v=0 on the top wall (lid).
    - p=0 at point (0, 0) to enforce a unique pressure solution.
"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# Define physical parameters
nu = 0.01  # Kinematic viscosity (for Re=100)

# 1. Define Geometry
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# 2. Define the PDE System
def pde_system(x, Y):
    """
    Defines the Navier-Stokes equations.
    x is a 2D tensor where x[:, 0:1] is x and x[:, 1:2] is y.
    Y is the network's output tensor: Y = [u, v, p].
    """
    u, v, p = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]

    # First-order derivatives
    u_x = dde.grad.jacobian(Y, x, i=0, j=0)
    u_y = dde.grad.jacobian(Y, x, i=0, j=1)
    v_x = dde.grad.jacobian(Y, x, i=1, j=0)
    v_y = dde.grad.jacobian(Y, x, i=1, j=1)
    p_x = dde.grad.jacobian(Y, x, i=2, j=0)
    p_y = dde.grad.jacobian(Y, x, i=2, j=1)

    # Second-order derivatives (Laplacians)
    u_xx = dde.grad.hessian(Y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(Y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(Y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(Y, x, component=1, i=1, j=1)

    # PDE residuals
    continuity = u_x + v_y
    x_momentum = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    y_momentum = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return [continuity, x_momentum, y_momentum]

# 3. Define Boundary and Initial Conditions
def boundary_walls(x, on_boundary):
    # Identifies points on the left, right, and bottom walls
    return on_boundary and (dde.utils.isclose(x[0], 0) or dde.utils.isclose(x[0], 1) or dde.utils.isclose(x[1], 0))

def boundary_top(x, on_boundary):
    # Identifies points on the top wall (lid)
    return on_boundary and dde.utils.isclose(x[1], 1)

# No-slip BC for stationary walls (u=0, v=0)
bc_u_walls = dde.DirichletBC(geom, lambda x: 0, boundary_walls, component=0)
bc_v_walls = dde.DirichletBC(geom, lambda x: 0, boundary_walls, component=1)

# Lid velocity BC for the top wall (u=1, v=0)
bc_u_top = dde.DirichletBC(geom, lambda x: 1, boundary_top, component=0)
bc_v_top = dde.DirichletBC(geom, lambda x: 0, boundary_top, component=1)

# Pressure pinning point p(0,0)=0
pin_point = np.array([[0.0, 0.0]])
bc_p = dde.PointSetBC(pin_point, [[0]], component=2)


# 4. Assemble and Train
data = dde.data.PDE(
    geom,
    pde_system,
    [bc_u_walls, bc_v_walls, bc_u_top, bc_v_top, bc_p],
    num_domain=2500,
    num_boundary=400,
    num_test=10000,
)

# Network: 2 inputs (x,y), 8 hidden layers of 40 neurons, 3 outputs (u,v,p)
net = dde.nn.FNN([2] + [40] * 8 + [3], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=30000)

# Refine with L-BFGS
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 5. Visualize
# Create a grid for prediction
grid_res = 101
x = np.linspace(0, 1, grid_res)
y = np.linspace(0, 1, grid_res)
X, Y = np.meshgrid(x, y)
xy_test = np.vstack((X.flatten(), Y.flatten())).T

# Predict on the grid
Y_pred = model.predict(xy_test)
u_pred = Y_pred[:, 0].reshape(grid_res, grid_res)
v_pred = Y_pred[:, 1].reshape(grid_res, grid_res)
p_pred = Y_pred[:, 2].reshape(grid_res, grid_res)

# Plotting velocity field
velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2)
plt.figure(figsize=(12, 9))
plt.pcolormesh(X, Y, velocity_magnitude, shading='auto', cmap='viridis')
plt.colorbar(label="Velocity Magnitude")
# Quiver plot for velocity vectors (downsample for clarity)
skip = 5
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], u_pred[::skip, ::skip], v_pred[::skip, ::skip], color='white')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Velocity Field for Lid-Driven Cavity (Re=100)")
plt.axis('square')
plt.show()

# Plotting pressure field
plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, p_pred, shading='auto', cmap='coolwarm')
plt.colorbar(label="Pressure (p)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Pressure Field for Lid-Driven Cavity (Re=100)")
plt.axis('square')
plt.show()
