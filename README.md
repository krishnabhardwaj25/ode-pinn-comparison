Physics-Informed Neural Networks vs Classical Numerical Methods
A comparative study of PINNs against classical numerical solvers across ODEs and PDEs of increasing complexity.
BTech Mathematics and Computing — Central University of Karnataka
Supervised by Dr. Sreenivasulu Ballem | May 2026

Overview
Most PINN papers showcase the method on problems where it performs well. This project does the opposite — puts PINNs and classical methods on the same problems, same machine, and lets the numbers speak.
Two approaches compared:

Classical — RK45 (SciPy solve_ivp) for ODEs, Method of Lines for PDEs
PINN — PyTorch neural network trained purely on physics residuals, zero training data


Problems Studied
ProblemTypeComplexityExponential DecayODE1st order, linearSimple Harmonic OscillatorODE2nd order, linearVan der Pol OscillatorODE2nd order, nonlinear1D Heat EquationPDELinear, IBVP

Results
ProblemNum MAEPINN MAESlowdownExponential Decay1.14e-096.01e-0514,939xSimple Harmonic Oscillator1.33e-083.77e-014,753xVan der Polreference7.06e-014,237x1D Heat Equation1.20e-051.58e-0362x
Key findings:

RK45 dominates all three ODEs — faster by thousands, more accurate by orders of magnitude
PINN fails on SHO due to architectural mismatch — Tanh activations cannot represent periodic solutions
PINN fails on Van der Pol due to optimization failure — Adam gets stuck in complex loss landscape
PINN becomes competitive on the PDE — slowdown drops to 62x as Method of Lines complexity increases
Mesh-free advantage of PINNs empirically demonstrated


PINN Architecture

4 hidden layers, 64 neurons each, Tanh activation
Adam optimizer, lr=1e-3, 5000 epochs
2000 collocation points, zero training data
Loss = physics residual + initial condition + boundary condition (PDE only)
PyTorch autograd computes exact derivatives — no finite difference approximation


Project Structure
ode_pinn_comp/
├── core/
│   ├── problems.py       # Problem definitions and exact solutions
│   ├── numerical.py      # RK45 and Method of Lines solvers
│   └── pinn.py           # PINN architecture and training
├── compare.py            # Comparison framework and plotting
├── app.py                # Main entry point
└── README.md

How to Run
Install dependencies:
pip install torch scipy numpy matplotlib
Run a specific problem:
python app.py decay
python app.py sho
python app.py vdp
python app.py heat

Future Work

SIREN networks — sinusoidal activations to fix oscillatory failure on SHO
L-BFGS optimizer — second order optimization for complex loss landscapes like Van der Pol
2D PDEs on irregular domains — where mesh-free advantage becomes decisive
Inverse problems — infer unknown physical parameters from noisy observations


References

Dormand & Prince (1980) — RK45
Lagaris et al. (1998) — Neural networks for ODEs/PDEs
Raissi et al. (2019) — Modern PINN framework
Lu et al. (2021) — DeepXDE
Sitzmann et al. (2020) — SIREN
Cuomo et al. (2022) — PINN survey
