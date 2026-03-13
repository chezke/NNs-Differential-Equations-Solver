import torch
import torch.optim as optim
from models import MLP
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# ==================================================
# 1. Training
# ==================================================
def NN_train(
        loss_function,
        N=None,
        models=None,
        learning_rate=0.0005,
        epochs=10000,
        display_step=1000,
        optimizer_class=optim.Adam,
        **loss_kwargs):
    
    if models is None:
        if N is None:
            N = MLP()
        models = [N]
    else:
        models = list(models)
        if len(models) == 0:
            raise ValueError("`models` must contain at least one model.")
        if N is None and len(models) == 1:
            N = models[0]

    params = []
    for model in models:
        params.extend(list(model.parameters()))

    optimizer = optimizer_class(params, lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(*models, **loss_kwargs)
        loss.backward()
        optimizer.step()

        if epoch % display_step == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    if len(models) == 1:
        return models[0]
    return tuple(models)


# ====================================================================================================
# 2. Visualization
# ====================================================================================================
# ====================================================================================================
# 2.1 Plot: ODE solution and error
# ====================================================================================================
def plot_ode_solution(
    x_min,
    x_max,
    approx_fun,
    exact_fun,
    num_points=100,
    xlabel='x',
    ylabel=r'$\Psi(x)$',
    error_label='Error',
    title=None,
    is_coupled=False,
    approx_fun_2=None,
    exact_fun_2=None,
):
    x_t = torch.linspace(x_min, x_max, num_points, requires_grad=True).unsqueeze(1)

    y_approx = approx_fun(x_t).detach().numpy().reshape(-1)
    y_exact = exact_fun(x_t).detach().numpy().reshape(-1)
    x = x_t.detach().numpy().reshape(-1)

    # --------------------------------------------------
    # Single ODE case
    # --------------------------------------------------
    if not is_coupled:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        ax1.plot(x, y_approx, label='NN Approximation')
        ax1.plot(x, y_exact, label='Exact Solution')
        ax1.legend()
        ax1.set_ylabel(ylabel, fontsize=18)

        ax2.plot(x, y_exact - y_approx, label='Residual')
        ax2.legend()
        ax2.set_xlabel(xlabel, fontsize=18)
        ax2.set_ylabel(error_label, fontsize=18)

    # --------------------------------------------------
    # Coupled ODE case
    # --------------------------------------------------
    else:
        if approx_fun_2 is None or exact_fun_2 is None:
            raise ValueError("For coupled systems you must provide approx_fun_2 and exact_fun_2.")
        y2_approx = approx_fun_2(x_t).detach().numpy().reshape(-1)
        y2_exact = exact_fun_2(x_t).detach().numpy().reshape(-1)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        # ψ1 solution
        axes[0, 0].plot(x, y_approx, label='NN Approximation')
        axes[0, 0].plot(x, y_exact, label='Exact Solution')
        axes[0, 0].set_ylabel(r'$\Psi_1(x)$')
        axes[0, 0].legend()
        # ψ1 error
        axes[1, 0].plot(x, y_exact - y_approx, label='Error')
        axes[1, 0].set_xlabel(xlabel)
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].legend()
        # ψ2 solution
        axes[0, 1].plot(x, y2_approx, label='NN Approximation')
        axes[0, 1].plot(x, y2_exact, label='Exact Solution')
        axes[0, 1].set_ylabel(r'$\Psi_2(x)$')
        axes[0, 1].legend()
        # ψ2 error
        axes[1, 1].plot(x, y2_exact - y2_approx, label='Error')
        axes[1, 1].set_xlabel(xlabel)
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].legend()

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


# ====================================================================================================
# 2.2 Plot: PDE solution and error
# ====================================================================================================
# ==================================================
# Helper: build 2D grid
# ==================================================
def build_pde_grid(x_min=0, x_max=1, y_min=0, y_max=1, num_points=30):
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y


# ==================================================
# Helper: evaluate trial solution on grid
# ==================================================
def evaluate_trial_solution_on_grid(trial_fun, N, X, Y):
    xy_pairs = torch.tensor(
        np.column_stack([X.ravel(), Y.ravel()]),
        dtype=torch.float32
    )

    with torch.no_grad():
        Z_trial = trial_fun(xy_pairs, N).cpu().numpy().reshape(X.shape)

    return Z_trial


# ==================================================
# Plot: exact solution wireframe
# ==================================================
def plot_pde_exact_wireframe(
    exact_fun,
    num_points=30,
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    title='Exact solution',
    elev=18,
    azim=-58
):
    X, Y = build_pde_grid(x_min, x_max, y_min, y_max, num_points)
    Z_exact = exact_fun(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z_exact, color='black', linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Solution')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    plt.show()


# ==================================================
# Plot: error wireframe
# ==================================================
def plot_pde_error_wireframe(
    exact_fun,
    trial_fun,
    N,
    num_points=10,
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    title='Solution Accuracy',
    elev=18,
    azim=-58
):
    X, Y = build_pde_grid(x_min, x_max, y_min, y_max, num_points)

    Z_exact = exact_fun(X, Y)
    Z_trial = evaluate_trial_solution_on_grid(trial_fun, N, X, Y)

    error = Z_trial - Z_exact

    print(f"{title} - max abs error: {np.max(np.abs(error)):.6e}")
    print(f"{title} - Z_trial shape: {Z_trial.shape}")
    print(f"{title} - error shape: {error.shape}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, error, color='black', linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Solution Accuracy')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    plt.show()


# ==================================================
# Plot: exact vs trial surface comparison
# ==================================================
def plot_pde_solution_comparison(
    exact_fun,
    trial_fun,
    N,
    num_points=30,
    x_min=0, x_max=1,
    y_min=0, y_max=1,
    title='Exact vs trial solution'
):
    X, Y = build_pde_grid(x_min, x_max, y_min, y_max, num_points)

    Z_exact = exact_fun(X, Y)
    Z_trial = evaluate_trial_solution_on_grid(trial_fun, N, X, Y)

    print(f"{title} - max abs error: {np.max(np.abs(Z_trial - Z_exact)):.6e}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z_exact, cmap='viridis', alpha=0.8)
    ax.plot_surface(X, Y, Z_trial, color='red', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Solution')
    ax.set_title(title)

    plt.show()

# ====================================================================================================
# 3. Loss Functions
# ====================================================================================================
def first_order_loss_with_ic(neural_network, a, g, ic, domain_lower_bound=0, domain_upper_bound=1, num_points=10):
    """Computes loss for a given NN

    Parameters
    ----------
    nerual_network : torch.nn.Module
        Neural network used in trial solution
    a: function
        Function of coefficient in first-order linear DE
    g : function
        Function on right-hand side of DE
    ic : float
        Initial condition of DE
    domain_lower_bout : float, optional
        Lower limit of the DE domain considered
    domain_upper_bound : float, optional
        Upper limit of the DE domain considered
    num_points : int, optional
        Number of points in which to split the domain
    
    Returns
    -------
    torch.tensor
        resulting loss of trial solution in it's current state
    """

    def trial_solution(x):
        result = ic + x * neural_network(x)

        return result
    
    delta = 1e-5
    x = torch.linspace(domain_lower_bound, domain_upper_bound, num_points, requires_grad=True).unsqueeze(1)
    dtrial_dx = (trial_solution(x+delta) - trial_solution(x)) / delta
    individual_error = (dtrial_dx - (g(x) - a(x)*trial_solution(x)))**2
    # individual_error = torch.abs(dtrial_dx - (g(x) - p(x)*trial_solution(x)))

    return torch.sum(individual_error)


def second_order_loss_with_ic(neural_network, a, b, g, ic, ic_prime, domain_lower_bound=0, domain_upper_bound=1, num_points=10):
    """Computes loss for a given NN

    Parameters
    ----------
    nerual_network : torch.nn.Module
        Neural network used in trial solution
    a: function
        Function of coefficient in first-order linear DE
    b: function
        Function of coefficient in first-order linear DE
    g : function
        Function on right-hand side of DE
    ic : float
        Initial condition of DE solution
    ic_prime : float
        Initial condition on the derivative of DE solution
    domain_lower_bout : float, optional
        Lower limit of the DE domain considered
    domain_upper_bound : float, optional
        Upper limit of the DE domain considered
    num_points : int, optional
        Number of points in which to split the domain
    
    Returns
    -------
    torch.tensor
        resulting loss of trial solution in it's current state
    """

    def trial_solution(x):
        result = ic + x*ic_prime + (x**2) * neural_network(x)
        # print(result.requires_grad)
        return result
    
    delta = 1e-3
    x = torch.linspace(domain_lower_bound, domain_upper_bound, num_points, requires_grad=True).unsqueeze(1)
    dtrial_dx = (trial_solution(x+delta) - trial_solution(x)) / delta
    d2trial_dx2 = (trial_solution(x+delta) - 2*trial_solution(x) + trial_solution(x-delta))/(delta**2)
    individual_error = (d2trial_dx2 - (g(x) - a(x)*dtrial_dx - b(x)*trial_solution(x)))**2
    
    return torch.sum(individual_error)


def coupled_first_order_system_loss_with_ic(
    N1, N2, rhs1, rhs2,
    domain_lower_bound=0.0,
    domain_upper_bound=3.0,
    num_points=100
):
    x = torch.linspace(domain_lower_bound, domain_upper_bound, num_points).unsqueeze(1)
    x.requires_grad_(True)

    # Trial solutions
    psi1_t = x * N1(x)
    psi2_t = 1 + x * N2(x)

    # Derivatives
    dpsi1_dx = torch.autograd.grad(
        psi1_t, x,
        grad_outputs=torch.ones_like(psi1_t),
        create_graph=True
    )[0]

    dpsi2_dx = torch.autograd.grad(
        psi2_t, x,
        grad_outputs=torch.ones_like(psi2_t),
        create_graph=True
    )[0]

    # Residuals
    res1 = dpsi1_dx - rhs1(x, psi1_t, psi2_t)
    res2 = dpsi2_dx - rhs2(x, psi1_t, psi2_t)

    loss = torch.sum(res1**2) + torch.sum(res2**2)
    return loss


def second_order_pde_loss_with_bc(
        neural_network,
        x_domain_lower_bound=0, 
        x_domain_upper_bound=1,
        y_domain_lower_bound=0, 
        y_domain_upper_bound=1, 
        num_points=10
):

    def A(x, y):
        exp_neg_1 = torch.exp(torch.tensor(-1.0))
        result = (1 - x) * y**3 + x * (1 + y**3) * exp_neg_1 + (1 - y) * x * (torch.exp(-x) - exp_neg_1) + y * ((1 + x) * torch.exp(-x) - (1 - x + 2 * x * exp_neg_1))
        return result
        
    def trial_solution(x,y):
        result = A(x,y)+ x *(1-x)*y*(1-y)*neural_network(x,y)
        return result

    h = 1e-2
     # Create a grid of x and y values using torch.meshgrid
    x_values = torch.linspace(x_domain_lower_bound, x_domain_upper_bound, num_points, requires_grad=True)
    y_values = torch.linspace(y_domain_lower_bound, y_domain_upper_bound, num_points, requires_grad=True)
    X, Y = torch.meshgrid(x_values, y_values, indexing='ij')

    # Flatten the grid to pass to the network
    X_flat = X.flatten().unsqueeze(1)
    Y_flat = Y.flatten().unsqueeze(1)

    # Compute the Laplacian using finite differences
    f_xx = (trial_solution(X_flat + h, Y_flat) - 2 * trial_solution(X_flat, Y_flat) + trial_solution(X_flat - h, Y_flat)) / h**2
    f_yy = (trial_solution(X_flat, Y_flat + h) - 2 * trial_solution(X_flat, Y_flat) + trial_solution(X_flat, Y_flat - h)) / h**2
    laplacian = f_xx + f_yy

    # Calculate the individual error at each grid point
    individual_error = (laplacian - torch.exp(-X_flat) * (X_flat - 2 + Y_flat**3 + 6 * Y_flat))**2
    
    # Sum the errors to get the total loss
    return torch.sum(individual_error)


def second_order_pde_loss_with_mixed_bc(
        N,
        x_domain_lower_bound=0, 
        x_domain_upper_bound=1,
        y_domain_lower_bound=0, 
        y_domain_upper_bound=1, 
        num_points=10
):

    h = 1e-2
     # Create a grid of x and y values using torch.meshgrid
    x_values = torch.linspace(x_domain_lower_bound, x_domain_upper_bound, num_points, requires_grad=True)
    y_values = torch.linspace(y_domain_lower_bound, y_domain_upper_bound, num_points, requires_grad=True)
    X, Y = torch.meshgrid(x_values, y_values, indexing='ij')

    # Flatten the grid to pass to the network
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Define the function B(x, y)
    def B(x, y):
        result = 2 * y * torch.sin(torch.pi * x)
        return result

    # Compute the derivative of the neural network with respect to y
    dN_dy = lambda x, y: (N(x, y + 1e-3) - N(x, y)) / 1e-3

    # Define the trial solution using the neural network and B(x, y)
    def trial_solution(x, y):
        result = B(x, y).squeeze() + x * (1 - x) * y * (N(x, y).squeeze() - N(x, torch.ones_like(y)).squeeze() - dN_dy(x, torch.ones_like(y)).squeeze())
        return result

    # Compute the Laplacian using finite differences
    f_xx = (trial_solution(X_flat + h, Y_flat) - 2 * trial_solution(X_flat, Y_flat) + trial_solution(X_flat - h, Y_flat)) / h**2
    f_yy = (trial_solution(X_flat, Y_flat + h) - 2 * trial_solution(X_flat, Y_flat) + trial_solution(X_flat, Y_flat - h)) / h**2
    laplacian = f_xx + f_yy

    # Calculate the individual error at each grid point
    individual_error = (laplacian - torch.sin(torch.pi * X_flat) * (2 - (torch.pi**2) * (Y_flat**2)))**2

    # Sum the errors to get the total loss
    return torch.sum(individual_error)


def second_order_nonlinear_pde_loss_with_mixed_bc(N,x_domain_lower_bound=0, x_domain_upper_bound=1,y_domain_lower_bound=0, y_domain_upper_bound=1, num_points=10):
    """Computes loss for a given N

    Parameters
    ----------
    nerual_network : torch.N.Module
        Neural network used in trial solution
    a: function
        Function of coefficient in first-order linear DE
    g : function
        Function on right-hand side of DE
    ic : float
        Initial condition of DE
    domain_lower_bout : float, optional
        Lower limit of the DE domain considered
    domain_upper_bound : float, optional
        Upper limit of the DE domain considered
    num_points : int, optional
        Number of points in which to split the domain
    
    Returns
    -------
    torch.tensor
        resulting loss of trial solution in it's current state
    """
    h = 1e-2
     # Create a grid of x and y values using torch.meshgrid
    x_values = torch.linspace(x_domain_lower_bound, x_domain_upper_bound, num_points, requires_grad=True)
    y_values = torch.linspace(y_domain_lower_bound, y_domain_upper_bound, num_points, requires_grad=True)
    X, Y = torch.meshgrid(x_values, y_values, indexing='ij')

    # Flatten the grid to pass to the network
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Define the function B(x, y)
    def B(x, y):
        result = 2 * y * torch.sin(torch.pi * x)
        return result

    # Compute the derivative of the neural network with respect to y
    dN_dy = lambda x, y: (N(x, y + 1e-3) - N(x, y)) / 1e-3

    # Define the trial solution using the neural network and B(x, y)
    def trial_solution(x, y):
        result = B(x, y).squeeze() + x * (1 - x) * y * (N(x, y).squeeze() - N(x, torch.ones_like(y)).squeeze() - dN_dy(x, torch.ones_like(y)).squeeze())
        return result

    # Compute the Laplacian using finite differences
    f_xx = (trial_solution(X_flat + h, Y_flat) - 2 * trial_solution(X_flat, Y_flat) + trial_solution(X_flat - h, Y_flat)) / h**2
    f_yy = (trial_solution(X_flat, Y_flat + h) - 2 * trial_solution(X_flat, Y_flat) + trial_solution(X_flat, Y_flat - h)) / h**2
    dtrial_dy=(trial_solution(X_flat,Y_flat+h)-trial_solution(X_flat,Y_flat))/h
    laplacian = f_xx + f_yy

    # Calculate the individual error at each grid point
    individual_error = (laplacian + trial_solution(X_flat, Y_flat)*dtrial_dy - torch.sin(torch.pi*X_flat)*(2-(torch.pi**2)*(Y_flat**2) + 2*Y_flat**3*torch.sin(torch.pi*X_flat)))**2

    # Sum the errors to get the total loss
    return torch.sum(individual_error)