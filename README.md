# NNs-Differential-Equations-Solver
Neural network methods for solving differential equations (ODE/PDE).

This repository implements neural-network-based methods for solving ordinary and partial differential equations, following the ideas of Lagaris et al.

## Implemented Problems
- Problem 1: first-order ODE with initial condition
- Problem 2: first-order ODE
- Problem 3: second-order ODE with initial conditions
- Problem 4: two coupled first-order ODE's system
- Problem 5: linear PDE with Dirichlet boundary conditions
- Problem 6: linear PDE with mixed boundary conditions
- Problem 7: nonlinear PDE with mixed boundary conditions

## Main Features
- Generic training function: `NN_train`
- Unified ODE plotting helper
- Unified PDE plotting helpers
- Problem-specific trial solutions for PDEs

## Repository Structure
- `src/models.py`: neural network architectures
- `src/utils.py`: training, plotting, loss functions
- `notebooks/`: experiments and visualizations

## How to Run
1. Install dependencies
2. Open the notebook
3. Train the desired problem
4. Visualize solution and accuracy