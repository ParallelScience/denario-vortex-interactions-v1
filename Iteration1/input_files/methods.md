1. **Vortex Identification and Feature Extraction**:
   - Execute the parallelized Q-criterion pipeline, applying a volume-based filter (minimum 27 cells) to ensure structural coherence.
   - Compute the local strain-to-rotation ratio $\alpha(x,y,z) = |S|/|\Omega|$ and the total circulation $\Gamma$ for each identified vortex.
   - Extract the maximum local strain $\alpha_{max}$ within each labeled mask to represent the "weakest link" susceptible to shredding.

2. **Trajectory Construction and Lévy Flight Validation**:
   - Perform greedy nearest-neighbor matching across snapshots with variable `max_matching_distance` (0.05, 0.10, 0.15).
   - Validate "Lévy-like" behavior by comparing the Lévy index $\mu$ against a null model of synthetic Gaussian random walk trajectories (matched for density and duration) using a Kolmogorov-Smirnov test.
   - Flag and manually inspect trajectories exhibiting large jumps to distinguish between physical mergers/splits and tracking artifacts.

3. **Lagrangian Stability and Survival Analysis**:
   - Use a Cox Proportional Hazards model to predict vortex lifetime $\tau$. Include $\alpha_{max}$, the rate of change $\Delta \alpha_{max} / \Delta t$, vortex volume, and circulation $\Gamma$ as covariates to isolate the drivers of vortex termination.
   - Compare the stability of vortices in the driving range versus the inertial range to determine if shredding mechanisms are scale-dependent.

4. **Relative Dynamics and Geometric Exclusion**:
   - Isolate the interaction-induced velocity by subtracting the background flow, calculated via Gaussian kernel smoothing with $\sigma \approx 2-3 \times R_{avg}$.
   - Quantify the "dynamical exclusion zone" by comparing the observed radial distribution function $g(r)$ against a randomized Poisson point process of the same density. Define the physical exclusion radius $r_c$ where $g(r)$ significantly deviates from the random distribution.

5. **Effective Force Law Regression**:
   - Bin the interaction-induced relative velocity $\Delta \mathbf{v}$ against separation distance $r$ for $r > r_c$.
   - Fit the interaction model $F(r) = A r^{-n} \exp(-r/\lambda)$ using Bayesian Information Criterion (BIC) to determine the necessity of the screening term $\lambda$ versus a pure power-law.

6. **Turbulence Regime and Conditional Spectral Analysis**:
   - Compute the 3D energy spectrum $E(k)$ for the total field, the vortex-masked field, and the background flow (vortex-removed).
   - Perform conditional averaging of the velocity field centered on vortex cores to visualize internal structure (e.g., Lamb-Oseen profiles) and compare the core spectrum to the vorticity spectrum to determine if vortices are distinct from background turbulence.

7. **Vortex Size-Circulation Scaling**:
   - Perform robust regression of $\Gamma$ against effective radius $R_{eff} = (3V/4\pi)^{1/3}$.
   - Analyze the residuals of this fit against $\alpha_{max}$ to determine if high-strain environments systematically suppress circulation for a given vortex size.

8. **Model Synthesis**:
   - Synthesize findings into a descriptive model: (1) Birth via driving, (2) Evolution governed by $\alpha_{max}$ and $\Gamma$ stability, and (3) Interaction dynamics governed by the screened potential $V(r)$.
   - Define the model’s validity limits based on the energy spectrum analysis, framing the results as an effective theory for large-scale driven turbulence.