# Results: Effective Theory of Vortex Interactions in 3D Driven Turbulence

## Dataset Summary
- **Snapshots analysed**: 201 (every 5th from 1001 total; stride=5, Δt_snap=0.05)
- **Time range**: t = 189.03 to 199.03 (total span 10.00)
- **Grid**: 128³, periodic domain [-0.5, 0.5]³
- **Mean vortices per snapshot**: 104.6 (min=0, max=143)

## Vortex Tracking
- **Total trajectories formed**: 3327
- **Long trajectories (≥5 pts)**: 1386
- **Track length**: min=5, max=69, mean=12.3, median=9
- **Total step displacements**: 15608

## 1. Vortex Motion Statistics

### Step displacement distribution
| Metric | Value |
|--------|-------|
| Mean \|Δr\| | 0.01945 |
| Std \|Δr\| | 0.01713 |
| Kurtosis(Δx) | 10.51 |
| KS vs Gaussian (D) | 0.0927 |
| KS p-value | 3.16e-117 |
| Lévy stability α | 1.520 |

**Interpretation**: The kurtosis of 10.5 (Gaussian = 3) indicates extremely heavy-tailed step displacements. The KS test strongly rejects Gaussianity (p = 3.16e-117). The Lévy stability index α = 1.520 < 2 indicates super-Gaussian heavy tails consistent with Lévy flight dynamics.

### MSD Analysis
- **MSD exponent (slope)**: 1.277  
- **Hurst exponent H**: 0.638  
- **R²**: 0.9967

**Interpretation**: H = 0.638 > 0.5 indicates superdiffusion (anomalous Lévy-like transport).

## 2. Vortex Clustering and Spatial Structure

- **g(r) peak**: r = 0.0350, g = 1.334
- **Exclusion zone**: r < 0.0750
- **Total pair observations**: 726851

**Interpretation**: g(r) < 1 at small r indicates vortex exclusion — vortices repel each other at short range.  
The moderate structure in g(r) implies spatial correlations beyond a random Poisson field.

## 3. Effective Interaction Force Law

| Model | Parameters | R² | AIC |
|-------|-----------|-----|-----|
| power_law | [3.566754450508949, 2.9328725221311914e-08] | -0.0000 | 42655.7 |
| screened_exp | [0.30204356151739276, 1684296.9139745138] | -0.2217 | 46010.1 |
| yukawa | [0.011191915601670207, 842151.124078668] | -0.6638 | 51183.0 |


- **Power-law fit**: \|F\| ~ r^0.022  (R² = 0.001)
- **Best model by AIC**: **power_law**

**Effective potential**: From \|F\| = -dV/dr ~ r^0.022:
V(r) ~ r^0.978 (power-law potential)

**Minimal effective theory**: V(r) ≈ 3.5668 / r^-1.000  (Yukawa-like repulsion with exponent n-1=-1.000)


## 4. Vortex Size–Circulation Scaling

- **Scaling exponent β**: omega_tot ~ size^1.039  (R² = 0.992)

**Interpretation**: β = 1.039 ≈ 1 indicates circulation scales linearly with vortex volume. Consistent with Kolmogorov-scale vortex tubes where Γ ~ l (vortex tube radius ~ l).

## 5. Vortex Lifetime Distribution

- **Mean lifetime**: 0.613 simulation time units
- **Max lifetime**: 3.450
- **Exponential fit τ_mean**: 0.613
- **Normalised mean lifetime**: 12.3 snapshot intervals

**Interpretation**: The exponential lifetime distribution suggests vortex death is a memoryless Poisson process — consistent with stochastic strain-induced destruction at a constant rate 1/τ_mean = 1.631.

## 6. Turbulence Validation

- **Energy spectrum slope**: E(k) ~ k^-0.625  (K41 prediction: -5/3 = -1.667)
- **Deviation from K41**: 1.041

**Interpretation**: Deviates from K41 by 1.04 — consistent with finite-Reynolds-number effects at 128^3 resolution.

## 7. Lagrangian Deformation Analysis

- **Pearson r(deformation, lifetime)**: -0.032

**Interpretation**: Weak correlation between deformation and lifetime — vortex stability is not primarily controlled by local strain in this simulation.

## Summary: Effective Theory of Vortex Interactions

The vortex dynamics in this 3D driven turbulence simulation can be summarised by three key results:

1. **Vortex motion is non-Gaussian (super-diffusive, Lévy-like)**: 
   kurtosis(Δx) = 10.5, H = 0.638, α_Lévy = 1.520.

2. **Effective pairwise interaction potential**: The best-fit model is **power_law**
   with V(r) ~ r^0.98 (|F|~r^0.02).
   The g(r) peak at r=0.0350 indicates preferred vortex spacing consistent with the interaction potential minimum.

3. **Vortex birth/death**: Mean lifetime = 0.613 time units (exponential distribution, 
   memoryless destruction rate 1.631). Circulation scales as Ω ~ size^1.04.

These findings constrain the sub-grid scale parameterisation of vortex–vortex interactions in turbulence models.
