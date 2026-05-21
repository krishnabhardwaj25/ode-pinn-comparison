import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from compare import compare, compare_all

# ── output folder ─────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor"  : "#0f0f0f",
    "axes.facecolor"    : "#1a1a1a",
    "axes.edgecolor"    : "#444",
    "axes.labelcolor"   : "#ccc",
    "axes.titlesize"    : 11,
    "axes.titlecolor"   : "#fff",
    "xtick.color"       : "#888",
    "ytick.color"       : "#888",
    "grid.color"        : "#2a2a2a",
    "grid.linestyle"    : "--",
    "text.color"        : "#ccc",
    "legend.framealpha" : 0.2,
    "legend.edgecolor"  : "#444",
    "font.family"       : "monospace",
})

COLORS = {
    "numerical" : "#00d4ff",   # cyan
    "pinn"      : "#ff6b6b",   # coral
    "exact"     : "#a8ff78",   # green
    "error"     : "#ffd700",   # gold
    "loss"      : "#ff9f43",   # orange
}


# ─────────────────────────────────────────────────────────────
#  Plotting: ODEs
# ─────────────────────────────────────────────────────────────

def plot_ode_comparison(results: dict, problem_key: str, save: bool = True):
    problem    = results["problem"]
    num        = results["numerical"]
    pinn       = results["pinn"]
    comp       = results["comparison"]
    name       = problem["name"]
    exact_fn   = problem.get("exact")
    state_vars = problem["state_vars"]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"{name}  —  PINN vs Numerical", fontsize=14, color="#fff", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_sol    = fig.add_subplot(gs[0, 0])
    ax_loss   = fig.add_subplot(gs[0, 1])
    ax_err    = fig.add_subplot(gs[1, 0])
    ax_metric = fig.add_subplot(gs[1, 1])

    # ── [0,0] Solution overlay ──
    ax_sol.set_title(f"Solution  ({state_vars[0]})")
    ax_sol.set_xlabel("t")
    ax_sol.set_ylabel(state_vars[0])

    ax_sol.plot(num["t"], num["y"][0], color=COLORS["numerical"], linewidth=2, 
                label=f"Numerical ({comp['reference'] if exact_fn is None else 'RK45'})", zorder=3)
    ax_sol.plot(pinn["t"], pinn["y"][0], color=COLORS["pinn"], linewidth=1.8, 
                linestyle="--", label="PINN", zorder=4)

    if exact_fn is not None:
        t_fine  = np.linspace(problem["t_span"][0], problem["t_span"][1], 500)
        y_exact = exact_fn(t_fine)
        ax_sol.plot(t_fine, y_exact, color=COLORS["exact"], linewidth=1.2, 
                    linestyle=":", label="Exact", zorder=2)

    ax_sol.legend(fontsize=8)
    ax_sol.grid(True)

    # ── [0,1] Loss ──
    ax_loss.set_title("PINN Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log)")
    ax_loss.semilogy(pinn["loss_history"], color=COLORS["loss"], linewidth=1.5)
    ax_loss.grid(True)

    # ── [1,0] Error ──
    ax_err.set_title("Pointwise Absolute Error  (first state var)")
    ax_err.set_xlabel("t")
    ax_err.set_ylabel("|error|")

    pinn_t  = pinn["t"]
    pinn_y0 = pinn["y"][0]

    if exact_fn is not None:
        y_exact_eval = exact_fn(pinn_t)
        pinn_err     = np.abs(pinn_y0 - y_exact_eval)
        num_y0_interp= np.interp(pinn_t, num["t"], num["y"][0])
        num_err      = np.abs(num_y0_interp - y_exact_eval)

        ax_err.plot(pinn_t, num_err, color=COLORS["numerical"], linewidth=1.5, label="Numerical error")
        ax_err.plot(pinn_t, pinn_err, color=COLORS["pinn"], linewidth=1.5, label="PINN error", linestyle="--")
    else:
        num_y0_interp= np.interp(pinn_t, num["t"], num["y"][0])
        pinn_err     = np.abs(pinn_y0 - num_y0_interp)
        ax_err.plot(pinn_t, pinn_err, color=COLORS["error"], linewidth=1.5, label="PINN vs Numerical")
    
    ax_err.legend(fontsize=8)
    ax_err.grid(True)

    # ── [1,1] Metrics ──
    ax_metric.axis("off")
    ax_metric.set_title("Metrics Summary")

    lines = [
        f"Problem      :  {name}",
        f"t span       :  {problem['t_span']}",
        f"y0           :  {problem['y0']}",
        "",
        f"Reference    :  {comp['reference']}",
        f"Num MAE      :  {comp['num_mae']:.2e}" if comp["num_mae"] is not None else "Numerical MAE:  (is reference)",
        f"PINN MAE     :  {comp['pinn_mae']:.2e}",
        "",
        f"Numerical t  :  {comp['num_solve_time']:.4f}s",
        f"PINN train t :  {comp['pinn_train_time']:.2f}s",
        f"Slowdown     :  {comp['speedup']:.1f}x",
    ]

    ax_metric.text(
        0.05, 0.95, "\n".join(lines), transform=ax_metric.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace", color="#ccc",
        bbox=dict(boxstyle="round", facecolor="#1f1f1f", edgecolor="#444", alpha=0.8),
    )

    if save:
        path = os.path.join(RESULTS_DIR, f"{problem_key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[app] saved → {path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
#  Plotting: PDEs (1D Heat Equation)
# ─────────────────────────────────────────────────────────────

def plot_pde_comparison(results: dict, problem_key: str, save: bool = True):
    problem = results["problem"]
    num     = results["numerical"]
    pinn    = results["pinn"]
    comp    = results["comparison"]
    name    = problem["name"]

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"{name}  —  PINN vs Numerical", fontsize=14, color="#fff", y=0.98)

    # We use a 2x2 grid, but the bottom-right cell is split into two for Loss and Metrics
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    ax_num  = fig.add_subplot(gs[0, 0])
    ax_pinn = fig.add_subplot(gs[0, 1])
    ax_err  = fig.add_subplot(gs[1, 0])
    
    gs_inner  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.5)
    ax_loss   = fig.add_subplot(gs_inner[0, 0])
    ax_metric = fig.add_subplot(gs_inner[1, 0])

    X, T = pinn["X"], pinn["T"]
    U_num, U_pinn = num["U"], pinn["U"]

    # ── [0,0] Numerical Heatmap ──
    c1 = ax_num.contourf(T, X, U_num, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=ax_num)
    ax_num.set_title("Numerical Solution (U)")
    ax_num.set_xlabel("t")
    ax_num.set_ylabel("x")

    # ── [0,1] PINN Heatmap ──
    c2 = ax_pinn.contourf(T, X, U_pinn, levels=50, cmap="viridis")
    fig.colorbar(c2, ax=ax_pinn)
    ax_pinn.set_title("PINN Solution (U)")
    ax_pinn.set_xlabel("t")
    ax_pinn.set_ylabel("x")

    # ── [1,0] Absolute Error Heatmap ──
    exact_fn = problem.get("exact")
    if exact_fn is not None:
        U_exact   = exact_fn(X, T)
        err_val   = np.abs(U_pinn - U_exact)
        err_title = "Absolute Error (|PINN - Exact|)"
    else:
        err_val   = np.abs(U_pinn - U_num)
        err_title = "Absolute Error (|PINN - Numerical|)"

    c3 = ax_err.contourf(T, X, err_val, levels=50, cmap="magma")
    fig.colorbar(c3, ax=ax_err)
    ax_err.set_title(err_title)
    ax_err.set_xlabel("t")
    ax_err.set_ylabel("x")

    # ── [1,1 Top] Loss ──
    ax_loss.set_title("PINN Training Loss", fontsize=10)
    ax_loss.semilogy(pinn["loss_history"], color=COLORS["loss"], linewidth=1.5)
    ax_loss.grid(True)

    # ── [1,1 Bottom] Metrics ──
    ax_metric.axis("off")
    lines = [
        f"Problem      :  {name}",
        f"x span       :  {problem['x_span']}",
        f"t span       :  {problem['t_span']}",
        f"Reference    :  {comp['reference']}",
        f"Num MAE      :  {comp['num_mae']:.2e}" if comp["num_mae"] is not None else "Num MAE      :  (is ref)",
        f"PINN MAE     :  {comp['pinn_mae']:.2e}",
        f"Speedup      :  {comp['speedup']:.1f}x slower",
    ]

    ax_metric.text(
        0.05, 1.0, "\n".join(lines), transform=ax_metric.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace", color="#ccc",
        bbox=dict(boxstyle="round", facecolor="#1f1f1f", edgecolor="#444", alpha=0.8),
    )

    if save:
        path = os.path.join(RESULTS_DIR, f"{problem_key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[app] saved → {path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────────────────────

def plot_comparison(results: dict, problem_key: str, save: bool = True):
    """Routes to the correct plotter based on problem type."""
    if results["problem"].get("type") == "pde":
        plot_pde_comparison(results, problem_key, save)
    else:
        plot_ode_comparison(results, problem_key, save)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # usage:
    #   python app.py decay
    #   python app.py sho
    #   python app.py vdp
    #   python app.py heat
    #   python app.py all

    key = sys.argv[1] if len(sys.argv) > 1 else "decay"

    if key == "all":
        all_results = compare_all(pinn_epochs=5000, verbose=False)
        for k, res in all_results.items():
            plot_comparison(res, k, save=True)
    else:
        results = compare(key, pinn_epochs=5000, verbose=True)
        plot_comparison(results, key, save=True)