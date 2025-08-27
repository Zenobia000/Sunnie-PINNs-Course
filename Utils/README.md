# Utilities Module

## 1. Purpose of this Module

This module is intended to hold reusable utility functions that can be shared across the different coding examples in this course. The primary goal is to promote code reusability and keep the main example scripts focused on the core PINN implementation.

## 2. Contents

### `plotters.py`

This file will contain standardized functions for visualizing the results of our PINN models. For example, we could create functions like:

-   `plot_solution_1d(model, x_range, analytical_sol)`: For plotting 1D solutions against their analytical counterparts.
-   `plot_solution_2d(model, x_range, y_range)`: For creating heatmap and quiver plots for 2D solutions.
-   `plot_loss_history(losshistory)`: For plotting the training, validation, and test loss evolution.

By centralizing these plotting functions, we can ensure a consistent visual style for all the examples in the course.

## 3. How to Use

To use functions from this module in an example script (e.g., in `02-Standard_PINNs_for_Forward_Problems/`), you can add the root of the project to the Python path and import the necessary functions.

Example:
```python
import sys
sys.path.append('../..') # Adjust path as necessary
from Utils.plotters import plot_solution_1d
```

Note: The `pyproject.toml` file in `00-Course_Setup` has been configured to include `Utils` as a package, which should simplify imports once the environment is set up with `poetry install`.
