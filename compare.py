import numpy as np

from core.problems    import PROBLEMS
from core.numerical   import solve_numerically, solve_heat_mol
from core.pinn        import train_pinn, train_pinn_heat


def compare(
    problem_key : str,
    pinn_epochs : int   = 5000,
    pinn_lr     : float = 1e-3,
    pinn_colloc : int   = 2000,
    num_method  : str   = "RK45",
    verbose     : bool  = True,
) -> dict:
    """
    Runs both solvers on the chosen problem and returns a unified results dict.

    Args:
        problem_key : one of 'decay', 'sho', 'vdp', 'heat'
        pinn_epochs : training epochs for PINN
        pinn_lr     : learning rate for PINN
        pinn_colloc : number of collocation points for PINN
        num_method  : RK45 (default) or Radau for stiff problems
        verbose     : print progress

    Returns dict with:
        problem     : the problem dict (name, params, state_vars etc.)
        numerical   : full result dict from numerical solver
        pinn        : full result dict from pinn solver
        comparison  : summary metrics side by side
    """

    if problem_key not in PROBLEMS:
        raise ValueError(f"Unknown problem '{problem_key}'. Choose from: {list(PROBLEMS.keys())}")

    problem = PROBLEMS[problem_key]
    is_pde  = problem.get("type") == "pde"

    print("=" * 60)
    print(f"  Problem  : {problem['name']}")
    print(f"  t_span   : {problem['t_span']}")
    
    # Print appropriate domain/initial parameters based on problem type
    if is_pde:
        print(f"  x_span   : {problem['x_span']}")
    else:
        print(f"  y0       : {problem['y0']}")
    print("=" * 60)

    # ── run solvers ───────────────────────────────────────────
    if is_pde:
        num_result  = solve_heat_mol(problem)
        pinn_result = train_pinn_heat(
            problem  = problem,
            n_colloc = pinn_colloc,
            epochs   = pinn_epochs,
            lr       = pinn_lr,
            verbose  = verbose,
        )
    else:
        num_result  = solve_numerically(problem, method=num_method)
        pinn_result = train_pinn(
            problem     = problem,
            problem_key = problem_key,
            n_colloc    = pinn_colloc,
            epochs      = pinn_epochs,
            lr          = pinn_lr,
            verbose     = verbose,
        )

    # ── build comparison summary ──────────────────────────────
    exact = problem.get("exact")

    if exact is not None:
        # Both ODE and PDE solvers return MAE if exact exists
        num_mae  = num_result["error"]
        pinn_mae = pinn_result["error"]
        ref      = "exact solution"
    else:
        # Use numerical solution as reference for PINN (e.g., Van der Pol)
        pinn_t   = pinn_result["t"]
        num_t    = num_result["t"]
        num_y0   = num_result["y"][0]               # first state var
        pinn_y0  = pinn_result["y"][0]

        num_interp = np.interp(pinn_t, num_t, num_y0)
        pinn_mae   = float(np.mean(np.abs(pinn_y0 - num_interp)))
        num_mae    = None                            # numerical is the reference
        ref        = "numerical solution"

    comparison = {
        "reference"       : ref,
        "num_mae"         : num_mae,
        "pinn_mae"        : pinn_mae,
        "num_solve_time"  : num_result["duration"],
        "pinn_train_time" : pinn_result["duration"],
        "speedup"         : pinn_result["duration"] / num_result["duration"],
    }

    # ── print summary ─────────────────────────────────────────
    print()
    print("── Comparison Summary ──────────────────────────────")
    print(f"  Reference          : {ref}")
    if num_mae is not None:
        print(f"  Numerical MAE      : {num_mae:.2e}")
    print(f"  PINN MAE           : {pinn_mae:.2e}")
    print(f"  Numerical time     : {num_result['duration']:.4f}s")
    print(f"  PINN train time    : {pinn_result['duration']:.2f}s")
    print(f"  PINN / Numerical   : {comparison['speedup']:.1f}x slower")
    print("────────────────────────────────────────────────────")

    return {
        "problem"    : problem,
        "numerical"  : num_result,
        "pinn"       : pinn_result,
        "comparison" : comparison,
    }


# ── run all problems back to back ────────────────────────────
def compare_all(pinn_epochs: int = 5000, verbose: bool = False) -> dict:
    """Runs compare() on all problems. Returns dict keyed by problem_key."""
    results = {}
    for key in PROBLEMS:
        results[key] = compare(key, pinn_epochs=pinn_epochs, verbose=verbose)
        print()
    return results


# ── entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # usage: python compare.py decay
    #        python compare.py sho
    #        python compare.py vdp
    #        python compare.py heat
    #        python compare.py all

    key = sys.argv[1] if len(sys.argv) > 1 else "decay"

    if key == "all":
        compare_all(pinn_epochs=3000, verbose=False)
    else:
        compare(key, pinn_epochs=3000, verbose=True)