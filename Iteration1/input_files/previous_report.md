

Iteration 0:
# Research Summary: 3D NS Turbulence Vortex Effective Theory

## 1. Project Status & Methodology
- **Objective**: Characterize vortex interaction potential, motion statistics, and stability in 3D driven turbulence.
- **Data**: 200 snapshots (128³ grid, periodic, Δt=0.05).
- **Methodology**: Q-criterion identification (threshold: mean + 1.5σ), greedy nearest-neighbor tracking (max dist 0.10), and Lagrangian deformation analysis.
- **Key Constraints**: 128³ resolution limits inertial range (E(k) slope -0.625 vs -1.667).

## 2. Key Findings
- **Vortex Motion**: Super-diffusive (H=0.638) and non-Gaussian. Step displacements follow Lévy flight statistics (α=1.52, kurtosis=10.5).
- **Interaction Potential**: Best fit is a power-law potential $V(r) \sim r^{0.98}$ ($|F| \sim r^{0.02}$). Short-range repulsion observed ($g(r) < 1$ for $r < 0.075$).
- **Scaling**: Circulation $\Gamma$ scales linearly with vortex volume ($\beta \approx 1.04$).
- **Lifetime**: Exponential distribution (mean 0.613 time units), suggesting a memoryless Poisson destruction process.
- **Deformation**: Weak correlation (r=-0.032) between integrated strain-to-rotation ratio and vortex lifetime; local strain is not the primary determinant of vortex stability.

## 3. Limitations & Uncertainties
- **Resolution**: Finite-Reynolds-number effects significantly distort the energy spectrum.
- **Model Selection**: While power-law is the best AIC fit, the $R^2$ for the force law is extremely low (0.001), indicating high stochasticity or insufficient model complexity.
- **Tracking**: Greedy matching may misidentify events in high-density regions; merger/split categorization requires further validation.

## 4. Future Directions
- **Refine Interaction Model**: Investigate if the poor $R^2$ is due to the omission of vortex orientation (vorticity vector alignment) in the pairwise force calculation.
- **Stability Analysis**: Shift focus from local strain to global topological invariants or helicity density to explain vortex destruction.
- **Resolution Scaling**: Compare results against higher-resolution (256³+) datasets to confirm if the Lévy-like transport and interaction potential are robust or artifacts of grid-scale dissipation.
        