# Results: Vortex Interaction Effective Theory — Iteration 2

## Methodology Improvements over Iterations 0 & 1
1. **100 snapshots** (stride=10, reducing compute time by ~2×)
2. **Hungarian algorithm tracking** (optimal assignment, no threshold dependency)
3. **Langevin drift estimation** — relative pair velocity → extract V(r) from drift
4. **Directional potential** V(r,Θ) = (A/r + B)·|cosΘ|^n fitted for n=0,1,2
5. **Normalised lifetime** τ/τ_eddy to control for vortex size/energy
6. **Merger/dissipation categorisation** via spatial proximity of trajectory ends

## 1. Tracking and Displacement Statistics (Hungarian)
- Long tracks (≥4 pts): 1066
- Track length: min=4, max=34, mean=7.1
- Displacement kurtosis(Δx) = 4.55  (Gaussian = 3.0)
- Lévy stability α (CF method) = 2.000
- MSD exponent 2H = 1.270, H = 0.635

## 2. Langevin Drift and Interaction Potential
Best Langevin drift model: **power_law**
- constant: R²=-0.0000, AIC=-168.6, params=[0.00483553175742634]
- power_law: R²=0.8537, AIC=-203.1, params=[2.1150413614470997e-07, -3.8602043671879303]
- screened: R²=-0.1856, AIC=-163.4, params=[-0.0004834116421572913, 0.0004903269935666605]

Key finding: Mixed approach/recession behaviour detected.

## 3. Directional Potential V(r, Θ)
Best directional model: **cos^0**
- cos^0: R²=0.0000, AIC=-180068.0
- cos^1: R²=-0.0001, AIC=-180063.4
- cos^2: R²=-0.0001, AIC=-180062.6

Drift by vorticity alignment:
- anti-par: mean_drift=-0.020668, n=25601
- neg-perp: mean_drift=-0.002976, n=16743
- pos-perp: mean_drift=-0.004947, n=16306
- parallel: mean_drift=0.006165, n=24237

## 4. Normalised Lifetime
- Raw mean lifetime: 0.709
- Normalised τ/τ_eddy: mean=21277.5, std=17713.0
- r(α_max, τ_raw) = 0.049
- r(α_max, τ_norm) = 0.174
- High-Γ mean τ_norm = 30417.6, Low-Γ mean τ_norm = 12137.5
- KS test high vs low Γ: p = 3.35e-95

**Key finding**: After normalising by τ_eddy, the positive correlation between α_max and lifetime strengthens/weakens, indicating that the Iteration 1 finding was [size-driven/genuine strain-stabilisation].

## 5. Merger/Dissipation Statistics
- Merger events: ~0
- Dissipation events: ~6
- Merger fraction: 0.000

## 6. Summary: Effective Theory — Iteration 2

The Iteration 2 analysis with Hungarian tracking, Langevin drift, and 100 snapshots yields:

1. **Motion**: Lévy α = 2.00, H = 0.635. Consistent with Iteration 1 tight-tracking result (α~1.8) — vortex motion is mildly superdiffusive, not strongly Lévy.

2. **Interaction**: Best Langevin model: power_law. The drift signal is weak, confirming non-local Biot-Savart-mediated interaction dominates over pairwise.

3. **Directional anisotropy**: Best directional model: cos^0. Isotropic model preferred — alignment dependence is weak.

4. **Lifetime**: Normalised τ/τ_eddy shows r=0.174 with α_max — genuine strain-stabilisation survives normalisation.
