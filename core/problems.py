import numpy as np

# ─────────────────────────────────────────────
#  Each ODE problem dict has:
#   ode        : f(t, y) for solve_ivp
#   y0         : initial conditions
#   t_span     : (t_start, t_end)
#   t_eval     : points to evaluate solution at
#   exact      : closed-form solution if it exists, else None
#   name       : display name
#   state_vars : names of each variable (for plotting)
#   type       : "ode" or "pde"
#
#  PDE problem dicts additionally have:
#   x_domain   : (x_start, x_end)
#   alpha      : diffusion coefficient
#   ic         : initial condition function u(x, 0)
#   bc         : boundary condition value (u=0 at both ends)
# ─────────────────────────────────────────────


# ── 1. Exponential Decay ─────────────────────
#   dy/dt = -k*y,  y(0) = 1
#   exact: y(t) = e^(-kt)

K = 1.0

def _decay_ode(t, y):
    return [-K * y[0]]

def _decay_exact(t):
    return np.exp(-K * t)

EXPONENTIAL_DECAY = {
    "name"       : "Exponential Decay",
    "type"       : "ode",
    "ode"        : _decay_ode,
    "y0"         : [1.0],
    "t_span"     : (0.0, 5.0),
    "t_eval"     : np.linspace(0, 5, 200),
    "exact"      : _decay_exact,
    "state_vars" : ["y"],
    "params"     : {"k": K},
}


# ── 2. Simple Harmonic Oscillator ────────────
#   dy/dt =  v
#   dv/dt = -ω²*y
#   y(0) = 1,  v(0) = 0
#   exact: y(t) = cos(ωt)

OMEGA = 2.0

def _sho_ode(t, y):
    dydt = y[1]
    dvdt = -(OMEGA ** 2) * y[0]
    return [dydt, dvdt]

def _sho_exact(t):
    return np.cos(OMEGA * t)

SIMPLE_HARMONIC_OSCILLATOR = {
    "name"       : "Simple Harmonic Oscillator",
    "type"       : "ode",
    "ode"        : _sho_ode,
    "y0"         : [1.0, 0.0],
    "t_span"     : (0.0, 2 * np.pi),
    "t_eval"     : np.linspace(0, 2 * np.pi, 300),
    "exact"      : _sho_exact,
    "state_vars" : ["position", "velocity"],
    "params"     : {"omega": OMEGA},
}


# ── 3. Van der Pol Oscillator ────────────────
#   dy/dt =  v
#   dv/dt =  µ*(1 - y²)*v - y
#   y(0) = 2,  v(0) = 0
#   no closed-form solution

MU = 1.0

def _vdp_ode(t, y):
    dydt = y[1]
    dvdt = MU * (1 - y[0] ** 2) * y[1] - y[0]
    return [dydt, dvdt]

VAN_DER_POL = {
    "name"       : "Van der Pol Oscillator",
    "type"       : "ode",
    "ode"        : _vdp_ode,
    "y0"         : [2.0, 0.0],
    "t_span"     : (0.0, 10.0),
    "t_eval"     : np.linspace(0, 10, 500),
    "exact"      : None,
    "state_vars" : ["position", "velocity"],
    "params"     : {"mu": MU},
}


# ── 4. 1D Heat Equation (PDE) ─────────────────
#  ∂u/∂t = α ∂²u/∂x²
#  Domain : x ∈ [0, 1],  t ∈ [0, 1]
#  IC     : u(x, 0) = sin(πx)
#  BC     : u(0, t) = u(1, t) = 0
#  Exact  : u(x, t) = e^(-α π² t) * sin(πx)

ALPHA = 0.4   # diffusion coefficient

def _heat_ic(x):
    """Initial condition: u(x, 0) = sin(πx)"""
    return np.sin(np.pi * x)

def _heat_bc(x, t):
    """Boundary condition: u(0, t) = u(1, t) = 0"""
    # Returns an array of zeros matching the shape of the boundary points
    return np.zeros_like(x)

def _heat_exact(x, t):
    """Exact solution: u(x, t) = e^(-α π² t) sin(πx)"""
    # Numpy handles the meshgrid arrays automatically here
    return np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)

HEAT_1D = {
    "name"       : "1D Heat Equation",
    "type"       : "pde",
    "x_span"     : (0.0, 1.0),
    "t_span"     : (0.0, 1.0),
    "x_eval"     : np.linspace(0, 1, 100),   # 100 points for spatial plotting
    "t_eval"     : np.linspace(0, 1, 100),   # 100 points for temporal plotting
    "alpha"      : ALPHA,
    "ic_func"    : _heat_ic,
    "bc_func"    : _heat_bc,
    "exact"      : _heat_exact,
    "state_vars" : ["u"],
    "params"     : {"alpha": ALPHA},
}

# ── registry ──────────────────────────────────
PROBLEMS = {
    "decay" : EXPONENTIAL_DECAY,
    "sho"   : SIMPLE_HARMONIC_OSCILLATOR,
    "vdp"   : VAN_DER_POL,
    "heat"  : HEAT_1D,
}