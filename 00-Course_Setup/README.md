# Part 0: Course Setup

This module contains all the necessary setup instructions to create a consistent and reproducible environment for this course. We will be using `Poetry` for dependency management.

## 1. Environment Setup

### 1.1. Python Version

Please ensure you have Python `3.10` or a compatible version installed on your system.

### 1.2. Poetry Installation

Poetry is a modern tool for dependency management and packaging in Python. If you don't have it installed, you can find the official installation instructions [here](https://python-poetry.org/docs/#installation).

### 1.3. Project Initialization

Once Poetry is installed, navigate to the root directory of this project (`PINNs-course/`) in your terminal and run the following command:

```bash
poetry install
```

This command will automatically create a virtual environment, read the `pyproject.toml` file from the `00-Course_Setup` directory, and install all the required libraries (e.g., `torch`, `deepxde`).

## 2. Main Dependencies

This course primarily relies on the following libraries:

-   **PyTorch**: The core deep learning framework.
-   **DeepXDE**: A powerful library for Physics-Informed Neural Networks.
-   **Matplotlib / Seaborn**: For data visualization.

All specific versions are managed in the `pyproject.toml` file.
