"""
cpinn_example.py

This script provides a conceptual demonstration of the Conservative PINN (cPINN)
methodology for solving PDEs with discontinuous solutions, like the inviscid
Burgers' equation.

A full cPINN implementation requires a custom training loop to handle multiple
networks and complex interface losses simultaneously, which is beyond the standard
DeepXDE API.

This script simplifies the concept by:
1.  Decomposing the domain into two subdomains.
2.  Creating and training an independent PINN for each subdomain.
3.  Stitching the results together for visualization.

This demonstrates the domain decomposition idea but omits the crucial
interface flux conservation loss term that a true cPINN would require.

Problem Definition: Inviscid Burgers' Equation
- PDE: d(u)/d(t) + u * d(u)/d(x) = 0, or d(u)/d(t) + d(u^2/2)/d(x) = 0
- Domain: x in [-1, 1], t in [0, 1]
- Initial Condition: u(x, 0) = -sin(pi * x)
- Boundary Conditions: u(-1, t) = u(1, t) = 0
- A shock is expected to form around x=0 as t increases.
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# --- 1. Domain Decomposition ---
# We split the domain x in [-1, 1] into two subdomains at x=0.
# Subdomain 1 (left): x in [-1, 0]
# Subdomain 2 (right): x in [0, 1]
geom1 = dde.geometry.Interval(-1, 0)
geom2 = dde.geometry.Interval(0, 1)

timedomain = dde.geometry.TimeDomain(0, 1)

geomtime1 = dde.geometry.GeometryXTime(geom1, timedomain)
geomtime2 = dde.geometry.GeometryXTime(geom2, timedomain)

# --- Define the PDE (Flux form) ---
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    # Flux F(u) = u^2 / 2
    F = u**2 / 2
    dF_x = dde.grad.jacobian(F, x, i=0, j=0)
    return du_t + dF_x

# --- Common Initial and Boundary Conditions ---
ic_func = lambda x: -np.sin(np.pi * x[:, 0:1])
ic = dde.IC(geomtime1, ic_func, lambda _, on_initial: on_initial) # Use geomtime1 for IC object, will be applied to both

# --- 2. Create and Train Independent PINNs ---

# --- Model for Subdomain 1 (Left) ---
print("--- Training Model for Subdomain 1 (x in [-1, 0]) ---")
bc_left1 = dde.DirichletBC(geomtime1, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], -1))
# Interface at x=0 is just another boundary for this simplified model
bc_right1 = dde.DirichletBC(geomtime1, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 0))
data1 = dde.data.TimePDE(geomtime1, pde, [bc_left1, bc_right1, ic], num_domain=1270, num_boundary=40, num_initial=80)
net1 = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model1 = dde.Model(data1, net1)
model1.compile("adam", lr=1e-3)
model1.train(iterations=10000)


# --- Model for Subdomain 2 (Right) ---
print("\n--- Training Model for Subdomain 2 (x in [0, 1]) ---")
# Interface at x=0
bc_left2 = dde.DirichletBC(geomtime2, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 0))
bc_right2 = dde.DirichletBC(geomtime2, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 1))
ic2 = dde.IC(geomtime2, ic_func, lambda _, on_initial: on_initial) # Need to redefine IC for the new geometry
data2 = dde.data.TimePDE(geomtime2, pde, [bc_left2, bc_right2, ic2], num_domain=1270, num_boundary=40, num_initial=80)
net2 = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model2 = dde.Model(data2, net2)
model2.compile("adam", lr=1e-3)
model2.train(iterations=10000)


# --- 3. Stitch and Visualize the Solution ---
print("\n--- Visualizing Combined Solution ---")
x = np.linspace(-1, 1, 201)
t = np.linspace(0, 1, 101)
X, T = np.meshgrid(x, t)
xt_test = np.vstack((X.flatten(), T.flatten())).T

# Predict on each subdomain and combine
x_left_mask = xt_test[:, 0] <= 0
x_right_mask = xt_test[:, 0] > 0

u_pred = np.zeros(len(xt_test))
u_pred[x_left_mask] = model1.predict(xt_test[x_left_mask])[:, 0]
u_pred[x_right_mask] = model2.predict(xt_test[x_right_mask])[:, 0]
U_pred = u_pred.reshape(X.shape)

# Plotting
plt.figure(figsize=(10, 8))
plt.pcolormesh(T, X, U_pred, shading='auto', cmap='hot')
plt.colorbar(label="Solution u(x,t)")
plt.xlabel("Time (t)")
plt.ylabel("Position (x)")
plt.title("cPINN Concept: Stitched Solution for Inviscid Burgers' Eq.")
plt.show()

# Plot snapshots
plt.figure(figsize=(12, 6))
time_snapshots = [0.0, 0.25, 0.5, 0.75, 1.0]
for t_val in time_snapshots:
    t_index = np.argmin(np.abs(t - t_val))
    plt.plot(x, U_pred[:, t_index], label=f't = {t_val:.2f}')

plt.title("Solution Snapshots Showing Shock Formation")
plt.xlabel("Position (x)")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid(True)
plt.show()
