import time
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
#  PINN Network
#  Input  : t (scalar time) OR (x, t) for PDEs
#  Output : y (state vars)   shape (N, n_vars)
# ─────────────────────────────────────────────────────────────

class PINN(nn.Module):
    def __init__(self, n_input: int = 1, n_output: int = 1, hidden: int = 64, layers: int = 4):
        """
        Args:
            n_input  : 1 for ODEs (t), 2 for 1D PDEs (x, t)
            n_output : number of state variables (1 for decay, 2 for SHO/VdP)
            hidden   : neurons per hidden layer
            layers   : number of hidden layers
        """
        super().__init__()

        dims = [n_input] + [hidden] * layers + [n_output]
        net  = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                net.append(nn.Tanh())   # Tanh works well for smooth ODE/PDE solutions

        self.net = nn.Sequential(*net)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        # If multiple inputs are passed (e.g., x and t), concatenate them
        if len(inputs) == 1:
            return self.net(inputs[0])
        else:
            return self.net(torch.cat(inputs, dim=1))


# ─────────────────────────────────────────────────────────────
#  Physics Residuals  (one per problem)
# ─────────────────────────────────────────────────────────────

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx via autograd."""
    return torch.autograd.grad(
        y, x,
        grad_outputs = torch.ones_like(y),
        create_graph = True,
    )[0]


def residual_decay(t, y, params):
    k    = params["k"]
    dydt = _grad(y[:, 0:1], t)
    return dydt + k * y[:, 0:1]


def residual_sho(t, y, params):
    omega = params["omega"]
    pos   = y[:, 0:1]
    vel   = y[:, 1:2]
    dpdt  = _grad(pos, t)
    dvdt  = _grad(vel, t)
    r1    = dpdt - vel
    r2    = dvdt + (omega ** 2) * pos
    return torch.cat([r1, r2], dim=1)


def residual_vdp(t, y, params):
    mu  = params["mu"]
    pos = y[:, 0:1]
    vel = y[:, 1:2]
    dpdt = _grad(pos, t)
    dvdt = _grad(vel, t)
    r1   = dpdt - vel
    r2   = dvdt - mu * (1 - pos ** 2) * vel + pos
    return torch.cat([r1, r2], dim=1)


def residual_heat(x, t, u, params):
    # u_t = alpha * u_xx  →  residual = u_t - alpha * u_xx
    alpha = params["alpha"]
    
    u_t  = _grad(u, t)
    u_x  = _grad(u, x)
    u_xx = _grad(u_x, x) # Second derivative for diffusion
    
    return u_t - (alpha * u_xx)


RESIDUAL_FNS = {
    "decay" : residual_decay,
    "sho"   : residual_sho,
    "vdp"   : residual_vdp,
}


# ─────────────────────────────────────────────────────────────
#  Training (ODEs)
# ─────────────────────────────────────────────────────────────

def train_pinn(
    problem     : dict,
    problem_key : str,
    n_colloc    : int   = 2000,
    epochs      : int   = 5000,
    lr          : float = 1e-3,
    verbose     : bool  = True,
) -> dict:
    
    name    = problem["name"]
    y0      = problem["y0"]
    t_span  = problem["t_span"]
    t_eval  = problem["t_eval"]
    exact   = problem["exact"]
    params  = problem["params"]
    n_vars  = len(y0)

    residual_fn = RESIDUAL_FNS[problem_key]

    print(f"[pinn] Training: {name}  |  vars={n_vars}  epochs={epochs}")

    t_colloc = torch.FloatTensor(n_colloc, 1).uniform_(t_span[0], t_span[1])
    t_colloc.requires_grad_(True)

    t_ic = torch.zeros(1, 1, requires_grad=False)
    y_ic = torch.FloatTensor([y0])

    # n_input defaults to 1 for ODEs
    model     = PINN(n_input=1, n_output=n_vars)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    start        = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        y_colloc = model(t_colloc)
        res      = residual_fn(t_colloc, y_colloc, params)
        loss_phy = torch.mean(res ** 2)

        y_pred_ic  = model(t_ic)
        loss_ic    = torch.mean((y_pred_ic - y_ic) ** 2)

        loss = loss_phy + loss_ic
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and epoch % 500 == 0:
            print(f"  epoch {epoch:>5} | loss {loss.item():.4e} "
                  f"(phy {loss_phy.item():.4e} + ic {loss_ic.item():.4e})")

    duration = time.time() - start

    model.eval()
    with torch.no_grad():
        t_tensor = torch.FloatTensor(t_eval).unsqueeze(1)
        y_tensor = model(t_tensor)

    t_out = t_eval
    y_out = y_tensor.numpy().T

    error = None
    if exact is not None:
        y_exact = exact(t_out)
        error   = float(np.mean(np.abs(y_out[0] - y_exact)))

    print(f"[pinn] Done in {duration:.2f}s", end="")
    print(f"  |  MAE vs exact: {error:.2e}" if error is not None else "  |  no exact solution")

    return {
        "t"            : t_out,
        "y"            : y_out,
        "duration"     : duration,
        "loss_history" : loss_history,
        "error"        : error,
    }


# ─────────────────────────────────────────────────────────────
#  Training (PDEs - 1D Heat Equation)
# ─────────────────────────────────────────────────────────────

def train_pinn_heat(
    problem     : dict,
    n_colloc    : int   = 3000,
    n_bc        : int   = 300,
    n_ic        : int   = 300,
    epochs      : int   = 5000,
    lr          : float = 1e-3,
    verbose     : bool  = True,
) -> dict:
    """
    Dedicated training loop for the 1D Heat Equation.
    """
    name    = problem["name"]
    x_span  = problem["x_span"]
    t_span  = problem["t_span"]
    x_eval  = problem["x_eval"] # For plotting meshgrid
    t_eval  = problem["t_eval"] # For plotting meshgrid
    ic_func = problem["ic_func"] # function returning initial temperature
    bc_func = problem["bc_func"] # function returning boundary temperature
    exact   = problem.get("exact", None)
    params  = problem["params"]

    print(f"[pinn] Training PDE: {name}  |  epochs={epochs}")

    # 1. Collocation Points (Interior of the x-t domain)
    x_colloc = torch.FloatTensor(n_colloc, 1).uniform_(x_span[0], x_span[1]).requires_grad_(True)
    t_colloc = torch.FloatTensor(n_colloc, 1).uniform_(t_span[0], t_span[1]).requires_grad_(True)

    # 2. Initial Condition Points (t = 0, x is random)
    x_ic = torch.FloatTensor(n_ic, 1).uniform_(x_span[0], x_span[1])
    t_ic = torch.zeros(n_ic, 1)
    u_ic = torch.FloatTensor(ic_func(x_ic.numpy()))

    # 3. Boundary Condition Points (x = x_min or x_max, t is random)
    t_bc = torch.FloatTensor(n_bc, 1).uniform_(t_span[0], t_span[1])
    t_bc_full = torch.cat([t_bc, t_bc], dim=0) # Double size for both boundaries
    
    x_bc_0 = torch.full((n_bc, 1), x_span[0], dtype=torch.float32)
    x_bc_L = torch.full((n_bc, 1), x_span[1], dtype=torch.float32)
    x_bc_full = torch.cat([x_bc_0, x_bc_L], dim=0)
    
    u_bc_full = torch.FloatTensor(bc_func(x_bc_full.numpy(), t_bc_full.numpy()))

    # Network needs 2 inputs (x, t) and 1 output (u)
    model     = PINN(n_input=2, n_output=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    start        = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # ── Physics Loss ──
        u_colloc = model(x_colloc, t_colloc)
        res      = residual_heat(x_colloc, t_colloc, u_colloc, params)
        loss_phy = torch.mean(res ** 2)

        # ── IC Loss ──
        u_pred_ic = model(x_ic, t_ic)
        loss_ic   = torch.mean((u_pred_ic - u_ic) ** 2)

        # ── BC Loss ──
        u_pred_bc = model(x_bc_full, t_bc_full)
        loss_bc   = torch.mean((u_pred_bc - u_bc_full) ** 2)

        loss = loss_phy + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and epoch % 500 == 0:
            print(f"  epoch {epoch:>5} | loss {loss.item():.4e} "
                  f"(phy {loss_phy.item():.4e} + ic {loss_ic.item():.4e} + bc {loss_bc.item():.4e})")

    duration = time.time() - start

    # ── Evaluation on Grid ──
    model.eval()
    with torch.no_grad():
        X, T = np.meshgrid(x_eval, t_eval)
        x_flat = torch.FloatTensor(X.flatten()).unsqueeze(1)
        t_flat = torch.FloatTensor(T.flatten()).unsqueeze(1)
        
        u_pred = model(x_flat, t_flat).numpy()
        U_pred = u_pred.reshape(X.shape)

    error = None
    if exact is not None:
        U_exact = exact(X, T)
        error   = float(np.mean(np.abs(U_pred - U_exact)))

    print(f"[pinn] Done in {duration:.2f}s", end="")
    print(f"  |  MAE vs exact: {error:.2e}" if error is not None else "  |  no exact solution")

    return {
        "X"            : X,
        "T"            : T,
        "U"            : U_pred,
        "duration"     : duration,
        "loss_history" : loss_history,
        "error"        : error,
    }


# ── quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from core.problems import PROBLEMS

    for key, problem in PROBLEMS.items():
        if key == "heat":
            result = train_pinn_heat(problem, epochs=3000, verbose=True)
            print(f"  U shape : {result['U'].shape}")
        else:
            result = train_pinn(problem, key, epochs=3000, verbose=True)
            print(f"  t shape : {result['t'].shape}")
            print(f"  y shape : {result['y'].shape}")
        print()