import time
import numpy as np
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────────
#  ODE solver — unchanged from before
#  Uses scipy solve_ivp with RK45
# ─────────────────────────────────────────────

def solve_numerically(problem: dict, method: str = "RK45") -> dict:
    """
    Solves an ODE problem using scipy's solve_ivp.

    Args:
        problem : a problem dict from problems.py
        method  : integration method — RK45 (default) or Radau for stiff problems

    Returns a dict with:
        t        : time points (np.ndarray)
        y        : solution array, shape (n_vars, n_points)
        duration : wall-clock solve time in seconds
        method   : method used
        error    : MAE vs exact solution if available, else None
    """

    t_span  = problem["t_span"]
    t_eval  = problem["t_eval"]
    y0      = problem["y0"]
    ode     = problem["ode"]
    exact   = problem["exact"]
    name    = problem["name"]

    print(f"[numerical] Solving: {name} using {method} ...")

    start = time.time()

    sol = solve_ivp(
        fun      = ode,
        t_span   = t_span,
        y0       = y0,
        method   = method,
        t_eval   = t_eval,
        rtol     = 1e-8,    # relative tolerance
        atol     = 1e-8,    # absolute tolerance
    )

    duration = time.time() - start

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for {name}: {sol.message}")

    # sol.y shape: (n_vars, n_points)
    t = sol.t
    y = sol.y

    # compute MAE against exact solution if available
    error = None
    if exact is not None:
        y_exact = exact(t)          # exact solution for first state var
        y_pred  = y[0]              # first state variable
        error   = float(np.mean(np.abs(y_pred - y_exact)))

    print(f"[numerical] Done in {duration:.4f}s", end="")
    print(f"  |  MAE vs exact: {error:.2e}" if error is not None else "  |  no exact solution")

    return {
        "t"        : t,
        "y"        : y,
        "duration" : duration,
        "method"   : method,
        "error"    : error,
    }


# ─────────────────────────────────────────────
#  PDE solver — Method of Lines (MOL)
#  for the 1D Heat Equation
#
#  ∂u/∂t = α ∂²u/∂x²
#
#  Idea:
#   1. Extract interior grid points from x_eval
#   2. Approximate ∂²u/∂x² at each point using
#      finite differences:
#      u''[i] ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
#   3. This turns the PDE into a system of N ODEs
#      one per grid point — and feeds it to solve_ivp
# ─────────────────────────────────────────────

def solve_heat_mol(problem: dict) -> dict:
    """
    Solves the 1D heat equation using Method of Lines.
    Now utilizes the shared x_eval and t_eval grids to match PINN output.
    """

    name     = problem["name"]
    alpha    = problem["alpha"]
    x_span   = problem["x_span"]
    t_span   = problem["t_span"]
    x_eval   = problem["x_eval"]
    t_eval   = problem["t_eval"]
    ic_func  = problem["ic_func"]
    bc_func  = problem["bc_func"]
    exact    = problem.get("exact", None)

    print(f"[numerical] Solving: {name} using Method of Lines ...")

    # ── spatial grid (interior points only) ───────────────────
    x_interior = x_eval[1:-1]
    dx = x_eval[1] - x_eval[0]

    # ── initial condition on interior grid ────────────────────
    u0 = ic_func(x_interior)

    # ── ODE system: one equation per grid point ───────────────
    def heat_odes(t, u):
        """
        Finite difference approximation of d²u/dx²
        Dynamic boundary conditions are evaluated at time t.
        """
        # bc_func expects arrays, so we wrap the scalars
        bc_left  = bc_func(np.array([x_span[0]]), np.array([t]))[0]
        bc_right = bc_func(np.array([x_span[1]]), np.array([t]))[0]

        dudt   = np.zeros_like(u)
        u_full = np.concatenate([[bc_left], u, [bc_right]])

        for i in range(len(u)):
            d2u_dx2 = (u_full[i] - 2 * u_full[i+1] + u_full[i+2]) / dx**2
            dudt[i] = alpha * d2u_dx2

        return dudt

    # ── solve ─────────────────────────────────────────────────
    start = time.time()

    sol = solve_ivp(
        fun    = heat_odes,
        t_span = t_span,
        y0     = u0,
        method = "RK45",
        t_eval = t_eval,
        rtol   = 1e-6,
        atol   = 1e-6,
    )

    duration = time.time() - start

    if not sol.success:
        raise RuntimeError(f"MOL solver failed: {sol.message}")

    u_interior = sol.y  # shape: (n_interior, n_time)

    # ── Reconstruct Full Grid (including boundaries) ──────────
    # Create an empty array for (n_x, n_t)
    U_full = np.zeros((len(x_eval), len(t_eval)))
    
    # Calculate boundary values across all time steps
    bc_left_all  = bc_func(np.full_like(t_eval, x_span[0]), t_eval)
    bc_right_all = bc_func(np.full_like(t_eval, x_span[1]), t_eval)

    # Stitch it all together
    U_full[0, :]  = bc_left_all
    U_full[-1, :] = bc_right_all
    U_full[1:-1, :] = u_interior

    # Transpose to match the (t, x) meshgrid orientation used by PINN
    U_pred = U_full.T
    X, T = np.meshgrid(x_eval, t_eval)

    # ── MAE vs exact ──────────────────────────────────────────
    error = None
    if exact is not None:
        U_exact = exact(X, T)
        error = float(np.mean(np.abs(U_pred - U_exact)))

    print(f"[numerical] Done in {duration:.4f}s", end="")
    print(f"  |  MAE vs exact: {error:.2e}" if error is not None else "")

    return {
        "X"        : X,
        "T"        : T,
        "U"        : U_pred,
        "duration" : duration,
        "method"   : "Method of Lines (RK45)",
        "error"    : error,
    }


# ── quick test ────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from core.problems import PROBLEMS

    for key in ["decay", "sho", "vdp"]:
        result = solve_numerically(PROBLEMS[key])
        print(f"  t: {result['t'].shape}  y: {result['y'].shape}\n")

    result = solve_heat_mol(PROBLEMS["heat"])
    print(f"  X: {result['X'].shape}  T: {result['T'].shape}  U: {result['U'].shape}")