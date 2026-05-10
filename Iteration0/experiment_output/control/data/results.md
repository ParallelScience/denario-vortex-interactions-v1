# 3D NS Turbulence — Vortex Interaction Effective Theory

## 1. Dataset Summary
- **Snapshots**: 201
- **Time Range**: 189.03 to 199.03
- **Mean Vortices per Snapshot**: 105.08

## 2. Quantitative Results
- **Step Displacement Kurtosis**: 10.27
- **Lévy Flight Exponent (alpha)**: 1.62
- **Hurst Exponent (H)**: 0.66
- **RDF g(r) Peak**: 2.27 at r = 0.025
- **Force Law Exponent (n)**: 0.05
- **Size-Circulation Exponent (beta)**: 1.04
- **Mean Vortex Lifetime**: 0.55
- **Energy Spectrum Slope**: -5.14

## 3. Interpretation of Findings
- **Statistical Nature of Motion**: The high kurtosis (10.27) and Lévy exponent alpha approx 1.62 indicate non-Gaussian, heavy-tailed random walks.
- **Anomalous Diffusion**: The Hurst exponent H approx 0.66 (> 0.5) confirms superdiffusive behavior.
- **Clustering**: The radial distribution function g(r) shows a distinct peak at r = 0.025 with a value of 2.27, indicating significant spatial clustering.
- **Size-Circulation Scaling**: The scaling omega_tot proportional to size^beta with beta approx 1.04 suggests a nearly linear relationship.
- **Energy Spectrum**: The kinetic energy spectrum slope of -5.14 deviates from the Kolmogorov K41 prediction.

## 4. Effective Theory Statement
The analysis of the effective force law reveals that the median acceleration |a| of vortex centers is independent of the nearest-neighbor distance r (n approx 0). This indicates that **no significant pairwise interaction potential V(r) is detected** between the vortex centers in this regime. The motion of the vortices is likely dominated by advection from the large-scale turbulent background flow rather than direct vortex-vortex interactions.

## 5. Summary Table
| Metric | Value |
|--------|-------|
| Kurtosis | 10.27 |
| Lévy alpha | 1.62 |
| Hurst Exponent H | 0.66 |
| g(r) Peak | 2.27 |
| Force Exponent n | 0.05 |
| Size-Circulation beta | 1.04 |
| Mean Lifetime | 0.55 |
| Energy Spectrum Slope | -5.14 |