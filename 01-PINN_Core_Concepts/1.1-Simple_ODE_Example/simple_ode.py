"""
simple_ode.py

This script provides a from-scratch implementation of a Physics-Informed Neural Network (PINN)
to solve a simple first-order Ordinary Differential Equation (ODE). This serves as the first
practical example in the course, demonstrating the core principles without relying on high-level libraries.

Problem Definition:
- ODE: dy/dx + y = 0
- Initial Condition: y(0) = 1
- Domain: x in [0, 5]
- Analytical Solution: y(x) = exp(-x)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better plots and consistent visualization
sns.set_style("whitegrid")

# --- 1. Neural Network Definition ---
# Represents the candidate solution y_theta(x)
class PINN(nn.Module):
    """
    A simple Feed-Forward Neural Network with Tanh activation functions.
    """
    def __init__(self):
        super(PINN, self).__init__()
        # Using nn.Sequential for a straightforward layer stack
        self.net = nn.Sequential(
            nn.Linear(1, 32),  # Input layer: 1 feature (x)
            nn.Tanh(),
            nn.Linear(32, 32), # Hidden layer
            nn.Tanh(),
            nn.Linear(32, 1)   # Output layer: 1 feature (y)
        )

    def forward(self, x):
        """Forward pass: maps input x to output y."""
        return self.net(x)

# --- 2. Loss Function Construction ---
# This is the core of the PINN methodology

def get_losses(model, x_domain, x0, y0):
    """
    Computes the composite loss for the PINN.
    
    Args:
        model (nn.Module): The neural network representing the solution.
        x_domain (torch.Tensor): Collocation points for enforcing the ODE.
        x0 (torch.Tensor): The point for the initial condition.
        y0 (torch.Tensor): The value for the initial condition.
        
    Returns:
        tuple: A tuple containing the total loss, ODE loss, and initial condition loss.
    """
    # --- a. ODE Residual Loss (L_ode) ---
    # Enforce the governing equation: dy/dx + y = 0
    
    # Enable gradient computation for the input tensor
    x_domain.requires_grad_(True)
    
    # Get the network's prediction for y at the collocation points
    y_pred = model(x_domain)
    
    # Use torch.autograd.grad to compute the derivative dy/dx
    # This is the key step where Automatic Differentiation is used.
    # grad_outputs must be specified and have the same shape as y_pred.
    # create_graph=True allows for computing higher-order derivatives if needed.
    dy_dx = torch.autograd.grad(
        y_pred, x_domain, 
        grad_outputs=torch.ones_like(y_pred), 
        create_graph=True
    )[0]
    
    # Calculate the residual of the ODE
    residual = dy_dx + y_pred
    
    # The loss is the Mean Squared Error of the residual
    loss_ode = torch.mean(residual**2)
    
    # --- b. Initial Condition Loss (L_ic) ---
    # Enforce the constraint y(0) = 1
    y0_pred = model(x0)
    loss_ic = torch.mean((y0_pred - y0)**2)
    
    # --- c. Total Loss ---
    total_loss = loss_ode + loss_ic
    
    return total_loss, loss_ode, loss_ic


# --- 3. Training ---

# Instantiate the model
pinn = PINN()
print("Model Architecture:")
print(pinn)

# Setup the optimizer
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# Define the computational domain and initial condition
# Collocation points for the ODE loss
x_domain = torch.linspace(0, 5, 100, requires_grad=True).view(-1, 1)
# Initial condition point
x0 = torch.tensor([[0.0]])
y0 = torch.tensor([[1.0]])

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Set model to training mode
    pinn.train()
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Calculate losses
    total_loss, loss_ode, loss_ic = get_losses(pinn, x_domain, x0, y0)
    
    # Backpropagate the total loss
    total_loss.backward()
    
    # Update the model parameters
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.6f}, '
              f'ODE Loss: {loss_ode.item():.6f}, IC Loss: {loss_ic.item():.6f}')


# --- 4. Validation & Visualization ---

# Set model to evaluation mode
pinn.eval()

# Generate a dense set of points for a smooth plot
x_test = torch.linspace(0, 5, 200).view(-1, 1)

# Get the PINN's prediction
with torch.no_grad():
    y_pinn = pinn(x_test)

# Get the analytical solution
y_analytical = torch.exp(-x_test)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_test.numpy(), y_analytical.numpy(), label='Analytical Solution ($e^{-x}$)', color='green', linewidth=2)
plt.plot(x_test.numpy(), y_pinn.numpy(), label='PINN Solution', color='red', linestyle='--', linewidth=2)
plt.title('PINN vs. Analytical Solution for $\\frac{dy}{dx} + y = 0$', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Final quantitative check at the initial condition
with torch.no_grad():
    y0_final_pred = pinn(x0)
print(f"\nFinal prediction at x=0: {y0_final_pred.item():.6f}")
print(f"True value at x=0: {y0.item():.6f}")
