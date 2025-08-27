# WBS TODO LIST for PINNs Course Development

This document outlines the Work Breakdown Structure (WBS) for the creation of the Physics-Informed Neural Networks (PINNs) course. The content will be developed following the Body of Knowledge (BoK) framework to ensure a systematic and comprehensive learning path suitable for graduate-level students.

## Phase 1: Project Initialization & Core Concepts

-   [x] **Module 00: Course Setup**
    -   [x] Create `README.md` with setup instructions.
    -   [x] Create `pyproject.toml` with project dependencies.
-   [x] **Module 01: PINN Core Concepts**
    -   [x] Write `README.md` for the module, explaining the BoK for this section.
    -   [x] **Sub-module 1.1: Simple ODE Example**
        -   [x] Write `README.md` explaining the problem from first principles (the ODE itself) and outlining the fundamental implementation steps.
        -   [x] Develop `simple_ode.py` with detailed comments.

## Phase 2: Standard PINNs for Forward Problems

-   [x] **Module 02: Standard PINNs for Forward Problems**
    -   [x] Write `README.md` for the module, introducing the concept of forward problems and the PDEs to be solved.
    -   [x] **Sub-module 2.1: Parabolic PDE - 1D Heat Equation**
        -   [x] Write `README.md` detailing the physics and mathematics (LaTeX) of the heat equation.
        -   [x] Develop `heat_equation.py` using the DeepXDE library.
    -   [x] **Sub-module 2.2: Elliptic PDE - 2D Poisson Equation**
        -   [x] Write `README.md` for the Poisson equation, focusing on boundary conditions.
        -   [x] Develop `poisson_equation.py`.
    -   [x] **Sub-module 2.3: Nonlinear PDE - 1D Burgers' Equation**
        -   [x] Write `README.md` explaining the challenge of the nonlinear term.
        -   [x] Develop `burgers_equation.py`.

## Phase 3: Advanced Topics & Capstone Project

-   [x] **Module 03: Advanced Applications - Inverse Problems**
    -   [x] Write `README.md` for the module, defining inverse problems.
    -   [x] **Sub-module 3.1: Heat Equation Parameter Discovery**
        -   [x] Write `README.md` explaining the setup for parameter discovery.
        -   [x] Develop `heat_inverse.py`.
-   [x] **Module 04: Optimization Challenges and Solutions**
    -   [x] Write `README.md` for the module, introducing the key optimization challenges.
    -   [x] **Sub-module 4.1: Theory - Gradient Pathologies**
        -   [x] Write `README.md` with a detailed theoretical explanation of gradient balancing, spectral bias, etc.
    -   [x] **Sub-module 4.2: Adaptive Weights - SA-PINN**
        -   [x] Write `README.md` explaining the Self-Adaptive PINN concept.
        -   [x] Develop `sa_pinn_example.py`.
    -   [x] **Sub-module 4.3: Variational Formulation - VPINN**
        -   [x] Write `README.md` on the theory of variational forms.
        -   [x] Develop `vpinn_example.py`.
    -   [x] **Sub-module 4.4: Discontinuous Solutions - cPINN**
        -   [x] Write `README.md` on the theory of conservative PINNs for shockwaves.
        -   [x] Develop `cpinn_example.py`.
-   [x] **Module 05: Specialized and Frontier Topics**
    -   [x] Write `README.md` for the module.
    -   [x] **Sub-module 5.1: Theory - Advanced Loss Functions**
        -   [x] Write `README.md` detailing concepts like gPINNs.
    -   [x] **Sub-module 5.2: Example - Gradient Enhanced PINN**
        -   [x] Write `README.md` for the gPINN example.
        -   [x] Develop `gpinn_example.py`.
-   [x] **Module 06: Capstone Project - Navier Stokes**
    -   [x] Write `README.md` introducing the Navier-Stokes equations as a capstone challenge.
    -   [x] **Sub-module 6.1: 2D Lid-Driven Cavity Flow**
        -   [x] Write `README.md` explaining the classic fluid dynamics problem.
        -   [x] Develop `navier_stokes.py`.

## Phase 4: Utilities and Finalization

-   [x] **Module: Utils**
    -   [x] Write `README.md` for the utility module.
    -   [x] Develop `plotters.py` with reusable visualization functions.
-   [ ] **Final Review**: Review all content for consistency, clarity, and accuracy.
