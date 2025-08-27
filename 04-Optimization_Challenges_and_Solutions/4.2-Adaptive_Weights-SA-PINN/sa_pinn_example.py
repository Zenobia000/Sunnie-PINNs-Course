"""
sa_pinn_example.py

This script demonstrates a simplified implementation of the Self-Adaptive PINN (SA-PINN)
concept to solve a convection-diffusion equation, which can be prone to training
difficulties due to competing loss terms.

Problem Definition:
- PDE: d(u)/d(t) + beta * d(u)/d(x) = nu * d^2(u)/d(x^2)
  - Convection term (beta * u_x) can create sharp gradients.
  - Diffusion term (nu * u_xx) smooths the solution.
- Domain: x in [-1, 1], t in [0, 1]
- Initial Condition: u(x, 0) = -sin(pi * x)
- Boundary Conditions: u(-1, t) = u(1, t) = 0
- Parameters:
    - Convection coefficient (beta): 1.0
    - Diffusion coefficient (nu): 0.01 / pi
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# Define PDE parameters
beta = 1.0
nu = 0.01 / np.pi

# 1. Define Geometry and Time Domain
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- Define the simplified SA-PINN approach ---

# 1a. Define Learnable Weights for the losses
# We create dde.Variable for each loss component.
# The initial value can be a simple guess, e.g., 1.0.
# The optimizer will then adjust these values.
lambda_pde = dde.Variable(1.0)
lambda_bc = dde.Variable(1.0)
lambda_ic = dde.Variable(1.0)

# 2. Define the PDE with these weights in mind
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t + beta * du_x - nu * du_xx

# 3. Define Boundary and Initial Conditions
bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# 4. Assemble the Model
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=10000,
)

net = dde.nn.FNN([2] + [32] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# 4a. Define Custom Loss with adaptive weights
# We assign the learnable weights to the corresponding loss terms.
# DeepXDE's compile function has a `loss_weights` argument that accepts these variables.
loss_weights = [lambda_pde, lambda_bc, lambda_ic]

# 4b. Identify all trainable variables: network parameters AND adaptive weights
external_trainable_variables = [lambda_pde, lambda_bc, lambda_ic]
variable_filename = "sa_variables.dat"
variable_callback = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename=variable_filename
)


# 5. Train the Model
# The key is to pass both `loss_weights` and `external_trainable_variables`.
# The optimizer will now minimize the weighted loss with respect to both
# the network parameters and the weights themselves.
# NOTE: This is a simplified approach, not the full minimax optimization.
# It encourages the weights to decrease, but can still help balance gradients.
print("Training with simplified adaptive weights...")
model.compile("adam", lr=1e-3, loss_weights=loss_weights, external_trainable_variables=external_trainable_variables)
losshistory, train_state = model.train(iterations=25000, callbacks=[variable_callback])

# Plotting the results
dde.plot.plot_loss_history(losshistory)
plt.title("Loss History for Convection-Diffusion Eq.")
plt.show()

# Plot the evolution of the adaptive weights
lines = open(variable_filename, "r").readlines()
# Each line is "var_1: val_1, var_2: val_2, ..."
history = []
for line in lines:
    parts = line.strip().split(',')
    vals = [float(p.split(':')[-1]) for p in parts]
    history.append(vals)
history = np.array(history)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(history)) * 1000, history[:, 0], label='$\lambda_{PDE}$')
plt.plot(np.arange(len(history)) * 1000, history[:, 1], label='$\lambda_{BC}$')
plt.plot(np.arange(len(history)) * 1000, history[:, 2], label='$\lambda_{IC}$')
plt.xlabel("Epoch")
plt.ylabel("Loss Weight Value")
plt.title("Evolution of Adaptive Loss Weights ($\lambda$)")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()
