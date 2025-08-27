"""
heat_inverse.py

This script solves an inverse problem for the 1D heat equation using DeepXDE.
It demonstrates how to discover an unknown physical parameter (thermal diffusivity)
from sparse observational data.

Problem Workflow:
1.  Define a forward problem with a known, true value of alpha.
2.  Solve the forward problem to generate a reference solution.
3.  Sample sparse data points from this reference solution to simulate experimental measurements.
4.  Define an inverse problem where alpha is a learnable dde.Variable.
5.  Train the inverse model using the sparse data and check if the learned alpha
    converges to the true value.
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")

# --- Step 1 & 2: Generate Synthetic Data ---

# Define the true thermal diffusivity
alpha_true = 0.1 / np.pi

def create_forward_model():
    """Creates and solves the forward problem to generate reference data."""
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde_forward(x, u):
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        return du_t - alpha_true * du_xx

    bc_left = dde.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 0))
    bc_right = dde.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 1))
    ic = dde.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(geomtime, pde_forward, [bc_left, bc_right, ic], num_domain=2540, num_boundary=80, num_initial=160)
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    print("Solving forward problem to generate data...")
    model.compile("adam", lr=1e-3)
    model.train(iterations=10000, display_every=5000)
    print("Forward problem solved.")
    return model

# Create and solve the forward model
forward_model = create_forward_model()

# --- Step 3: Sample Sparse Data ---
num_measurements = 50
observe_x = np.random.rand(num_measurements, 2)  # Random points in [0,1]x[0,1]
observe_u = forward_model.predict(observe_x)

# Optionally, add some noise to the measurements
# noise_level = 0.01
# observe_u += noise_level * np.std(observe_u) * np.random.randn(*observe_u.shape)

# Create the PointSetBC object for the inverse problem
observe_points = dde.PointSetBC(observe_x, observe_u)


# --- Step 4: Define and Solve the Inverse Problem ---

# Define the unknown parameter alpha as a dde.Variable.
# We initialize it with a guess, e.g., 0.0
alpha_inverse = dde.Variable(0.0)

def pde_inverse(x, u):
    """PDE residual for the inverse problem with a learnable alpha."""
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t - alpha_inverse * du_xx

# Reuse the same geometry and IC/BC definitions as the forward problem
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_left = dde.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 0))
bc_right = dde.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 1))
ic = dde.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# In the inverse problem, the boundary conditions now include the observational data points
data_inverse = dde.data.TimePDE(
    geomtime,
    pde_inverse,
    [bc_left, bc_right, ic, observe_points], # Add the PointSetBC here
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
)

net_inverse = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model_inverse = dde.Model(data_inverse, net_inverse)

# We need to tell the optimizer to train the external variable (alpha) as well
external_trainable_variables = [alpha_inverse]
variable_filename = "variables.dat"
variable_callback = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename=variable_filename
)

print("\nSolving inverse problem...")
model_inverse.compile("adam", lr=1e-3, external_trainable_variables=external_trainable_variables)
losshistory, train_state = model_inverse.train(iterations=20000, callbacks=[variable_callback])

# --- Step 5: Check the Result ---
print(f"\nTrue value of alpha: {alpha_true:.6f}")
print(f"Learned value of alpha: {alpha_inverse.value.numpy()[0]:.6f}")

# Plot the convergence of alpha
lines = open(variable_filename, "r").readlines()
alpha_history = np.array([float(line.split(":")[-1]) for line in lines])
plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(alpha_history)) * 1000, alpha_history, marker='o')
plt.axhline(y=alpha_true, color='r', linestyle='--', label=f'True value ({alpha_true:.4f})')
plt.xlabel("Epoch")
plt.ylabel("Value of alpha")
plt.title("Convergence of the Learned Thermal Diffusivity (alpha)")
plt.legend()
plt.grid(True)
plt.show()
