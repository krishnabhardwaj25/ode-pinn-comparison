# Physics-Informed Neural Networks vs Classical Numerical Methods

A comparative study of PINNs against classical numerical solvers across ODEs and PDEs of increasing complexity.

**BTech Mathematics and Computing — Central University of Karnataka**
Supervised by Dr. Sreenivasulu Ballem | May 2026

---

## Overview

Most PINN papers showcase the method on problems where it performs well. This project does the opposite — puts PINNs and classical methods on the same problems, same machine, and lets the numbers speak.

Two approaches compared:
- **Classical** — RK45 (SciPy solve_ivp) for ODEs, Method of Lines for PDEs
- **PINN** — PyTorch neural network trained purely on physics residuals, zero training data

---

## Problems Studied

| Problem | Type | Complexity |
|---|---|---|
| Exponential Decay | ODE | 1st order, linear |
| Simple Harmonic Oscillator | ODE | 2nd order, linear |
| Van der Pol Oscillator | ODE | 2nd order, nonlinear |
| 1D Heat Equation | PDE | Linear, IBVP |

---

## Results

| Problem | Num MAE | PINN MAE | Slowdown |
|---|---|---|---|
| Exponential Decay | 1.14e-09 | 6.01e-05 | 14,939x |
| Simple Harmonic Oscillator | 1.33e-08 | 3.77e-01 | 4,753x |
| Van der Pol | reference | 7.06e-01 | 4,237x |
| 1D Heat Equation | 1.20e-05 | 1.58e-03 | 62x |

**Key findings:**
- RK45 dominates all three ODEs — faster by thousands, more accurate by orders of magnitude
- PINN fails on SHO — Tanh activations cannot represent periodic solutions
- PINN fails on Van der Pol — Adam gets stuck in complex loss landscape
- PINN becomes competitive on the PDE — slowdown drops to 62x
- Mesh-free advantage of PINNs empirically demonstrated

---

## PINN Architecture

- 4 hidden layers, 64 neurons each, Tanh activation
- Adam optimizer, lr=1e-3, 5000 epochs
- 2000 collocation points, zero training data
- Loss = physics residual + initial condition + boundary condition (PDE only)
- PyTorch autograd computes exact derivatives

---

## Project Structure
ode_pinn_comp/

├── core/

│   ├── problems.py

│   ├── numerical.py

│   └── pinn.py

├── compare.py

├── app.py

└── README.md

---

## How to Run

Install dependencies:
pip install torch scipy numpy matplotlib

Run a specific problem:

python app.py decay

python app.py sho

python app.py vdp

python app.py heat


---

## Future Work

- **SIREN networks** — sinusoidal activations to fix SHO failure
- **L-BFGS optimizer** — second order optimization for Van der Pol
- **2D PDEs on irregular domains** — mesh-free advantage becomes decisive
- **Inverse problems** — infer unknown physical parameters from noisy observations

---

## References

- Dormand & Prince (1980)
- Lagaris et al. (1998)
- Raissi et al. (2019)
- Lu et al. (2021)
- Sitzmann et al. (2020)
- Cuomo et al. (2022)
