"""
burgers_equation.py

This script solves the 1D Burgers' equation using the DeepXDE library.
The Burgers' equation is a canonical example of a nonlinear PDE that combines
convection and diffusion terms.

Problem Definition:
- PDE: d(u)/d(t) + u * d(u)/d(x) = nu * d^2(u)/d(x^2)
- Domain: x in [-1, 1], t in [0, 1]
- Initial Condition: u(x, 0) = -sin(pi * x)
- Boundary Conditions:
    - u(-1, t) = 0
    - u(1, t) = 0
- Viscosity (nu): 0.01 / pi
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# Define the viscosity
nu = 0.01 / np.pi

# 1. Define Geometry and Time Domain
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Define the PDE
def pde(x, u):
    """
    Defines the residual of the Burgers' equation.
    x is a 2D tensor where x[:, 0:1] is x and x[:, 1:2] is t.
    u is the neural network's output.
    """
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    
    # The nonlinear convection term is simply u * du_x
    return du_t + u * du_x - nu * du_xx

# 3. Define Boundary and Initial Conditions
# Boundary function to identify points on the boundary
def boundary(x, on_boundary):
    return on_boundary and (dde.utils.isclose(x[0], -1) or dde.utils.isclose(x[0], 1))

# Dirichlet boundary condition: u(-1, t) = u(1, t) = 0
bc = dde.DirichletBC(geomtime, lambda x: 0, boundary)

# Initial condition: u(x, 0) = -sin(pi * x)
ic = dde.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# 4. Assemble and Train the Model
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=10000,
)

net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=15000)

model.compile("L-BFGS")
losshistory, train_state = model.train()


# 5. Visualize the Solution
dde.plot.plot_loss_history(losshistory)
plt.title("Loss History for 1D Burgers' Equation")
plt.show()

# Create a grid for prediction
x = np.linspace(-1, 1, 201)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)
xt_test = np.vstack((X.flatten(), T.flatten())).T

u_pred = model.predict(xt_test)
U_pred = u_pred.reshape(X.shape)

# Plot the 2D heatmap of the solution
plt.figure(figsize=(10, 8))
plt.pcolormesh(T, X, U_pred, shading='auto', cmap='hot')
plt.colorbar(label="Solution u(x,t)")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("PINN Solution of the 1D Burgers' Equation")
plt.show()

# Plot snapshots at different time steps
plt.figure(figsize=(12, 6))
time_snapshots = [0.0, 0.25, 0.5, 0.75, 1.0]
for i, t_val in enumerate(time_snapshots):
    t_index = np.argmin(np.abs(t - t_val))
    plt.plot(x, U_pred[:, t_index], label=f't = {t_val:.2f}')

plt.title("Solution u(x) at Different Time Snapshots")
plt.xlabel("Position (x)")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid(True)
plt.show()
