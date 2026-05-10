1. **Data Preprocessing and Vortex Identification**:
   - Implement a parallelized pipeline to process the 200 VTK snapshots.
   - Compute velocity gradient tensor components using `np.gradient` with periodic boundary handling.
   - Calculate the Q-criterion and define vortex cores using the threshold $Q > \text{mean}(Q_{Q>0}) + 1.5 \times \text{std}(Q_{Q>0})$.
   - Apply `scipy.ndimage.label` to identify connected components, enforcing a minimum volume threshold (e.g., 27 cells) to filter numerical noise.

2. **Trajectory Construction and Event Classification**:
   - Calculate vorticity-weighted centroids using the complex exponential trick for periodic domains.
   - Store centroid coordinates, total circulation $\Gamma$, and volume for each structure.
   - Perform greedy nearest-neighbor matching across snapshots (max distance 0.10).
   - Explicitly categorize trajectory terminations: flag "mergers" (two trajectories ending near a new one) and "splits" (one ending near two new ones) to distinguish from simple birth/death/dissipation events.
   - Save the resulting trajectory database to a structured format (e.g., Parquet or HDF5) for efficient downstream analysis.

3. **Lagrangian Deformation Analysis**:
   - For each tracked vortex, extract the velocity gradient tensor $A_{ij}$ as a volume-weighted average over the labeled mask to account for spatial extent.
   - Compute the strain-to-rotation ratio $\alpha = |S|/|\Omega|$.
   - Calculate the average deformation rate $\langle \alpha \rangle$ and the lifetime-normalized integrated deformation $\frac{1}{\tau} \int \alpha(t) \, dt$ to ensure comparability between short-lived and long-lived structures.

4. **Vortex Stability and Scaling Statistics**:
   - Correlate the evolution of circulation $\Gamma(t)$ with the normalized deformation intensity and vortex size.
   - Perform regression to determine the scaling exponent $\beta$ in $\Gamma \sim \text{size}^\beta$.
   - Identify the critical strain threshold $\alpha_c$ at which topological integrity is lost, excluding merger/split events from this specific stability regression to avoid noise contamination.

5. **Statistical Dynamics of Vortex Motion**:
   - Compute the Mean Squared Displacement (MSD) for the ensemble, extracting the Hurst exponent $H$.
   - Apply a Savitzky-Golay filter to centroid trajectories to reduce high-frequency noise before calculating kinematic statistics.
   - Analyze step displacement distributions (kurtosis) and perform hypothesis testing for Lévy flight vs. Gaussian random walk.
   - Fit the survival function of vortex lifetimes, accounting for the categorized merger/split events to refine decay statistics.

6. **Clustering and Interaction Potential Modeling**:
   - Compute the radial distribution function $g(r)$ to quantify spatial clustering.
   - Estimate centroid acceleration $a(t)$ using a second-order finite difference scheme: $a(t) \approx (x_{t+1} - 2x_t + x_{t-1}) / \Delta t^2$.
   - Bin acceleration vs. distance $r$, normalizing by $g(r)$ and the product of interacting circulations $\Gamma_i \Gamma_j$ to account for geometric bias and Biot-Savart dependencies.
   - Compare candidate potentials $V(r)$ (power-law, screened Coulomb, Yukawa) using AIC/BIC.

7. **Turbulence Diagnostics**:
   - Compute the 3D energy spectrum $E(k)$ via 3D FFT and shell-averaging to verify the inertial range scaling, acknowledging the limited resolution of the 128³ grid.
   - Validate the vortex identification method by correlating the number of identified vortices with the dissipation scale $\eta$ and driving parameters.

8. **Empirical Model Synthesis**:
   - Integrate findings into a unified model relating solenoidal driving to vortex-core stability.
   - Synthesize the relationship between circulation, cumulative stretching, and interaction potential to provide a high-fidelity basis for sub-grid scale parameterization in turbulence modeling.