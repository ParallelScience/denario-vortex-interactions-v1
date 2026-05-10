1. **Vortex Identification and Feature Extraction**:
   - Compute the velocity gradient tensor and the Q-criterion for all 200 snapshots.
   - Apply the threshold $Q > \text{mean}(Q_{Q>0}) + 1.5 \times \text{std}(Q_{Q>0})$ to generate binary masks.
   - Extract vortex properties: centroid (periodic-corrected), total circulation $\Gamma$ (sum of $\omega_{mag}$ in mask), volume, and the vorticity vector $\vec{\omega}$.
   - Compute the local strain-to-rotation ratio $\alpha = |S|/|\Omega|$ and the helicity density $H = \vec{v} \cdot \vec{\omega}$ within each identified core.

2. **Robust Vortex Tracking**:
   - Implement a tracking system using the Hungarian algorithm to minimize a cost matrix based on Euclidean distance, penalized by the difference in circulation $\Gamma$ and the alignment of the vorticity vectors $\vec{\omega}_1 \cdot \vec{\omega}_2$.
   - Use a Kalman Filter for each trajectory to predict positions and handle occlusions up to 3 frames.
   - Define "Mergers" as events where two trajectories terminate and a new one begins, satisfying $\Gamma_{new} \approx \sum \Gamma_i$ and spatial proximity.
   - Define "Dissipation" as trajectories shrinking below the volume threshold and "Exit" as periodic boundary crossings.

3. **Langevin Drift Estimation**:
   - Model relative motion as $d\mathbf{r}/dt = -\nabla V(r) + \eta(t)$.
   - Calculate the background flow by applying a Gaussian low-pass filter ($\sigma \approx 4$ grid cells) to the velocity field, while masking out the identified vortex cores to avoid self-advection bias.
   - Estimate the deterministic drift $\mathbf{F}_{int}(r) = \langle \Delta \mathbf{r} | \mathbf{r} \rangle / \Delta t$ by subtracting the filtered background flow from the centroid velocity.
   - Perform non-linear least-squares fitting of $V(r) = A r^{-n} \exp(-r/\lambda)$ using BIC for model selection.

4. **Deformation and Stability Analysis**:
   - Calculate the eddy turnover time $\tau_{eddy} = R_{eff}^2 / \Gamma$ and normalized decay rate $\lambda^* = (\tau_{life} / \tau_{eddy})^{-1}$.
   - Correlate $\lambda^*$ with the time-integrated strain $\alpha$ along the trajectory to identify the critical strain threshold $\alpha_c$ for vortex shredding.
   - Compare the helicity spectrum $H(k)$ within vortex cores versus the global background to validate the structural stability of the cores.

5. **Vortex Population Statistics**:
   - Calculate the radial distribution function $g(r)$ of vortex centroids to quantify clustering.
   - Determine the merger cross-section as a function of $r$ and $\Gamma_1/\Gamma_2$.
   - Compute the survival function of vortex lifetimes and fit to exponential or power-law distributions.

6. **Effective Theory Synthesis and Validation**:
   - Construct a unified model combining the interaction potential $V(r)$ and the empirical merger/dissipation rates.
   - Perform a synthetic Monte Carlo N-body simulation of a "vortex gas" using these derived rules in a periodic box.
   - Validate the effective theory by comparing the synthetic gas's steady-state $g(r)$, vortex density, and lifetime distributions against the original 3D simulation data.

7. **Spectral Analysis**:
   - Compute the 3D energy spectrum $E(k)$ specifically within the masked vortex core regions.
   - Compare core-specific spectral slopes against the global $k^{-5/3}$ Kolmogorov spectrum to characterize internal core dynamics (e.g., $k^{-1}$ or $k^{-2}$ scaling).

8. **Computational Execution**:
   - Utilize 8 parallel workers for VTK processing, ensuring OMP_NUM_THREADS=1 to prevent thread oversubscription.
   - Vectorize cost matrix calculations and ensure the Hungarian algorithm uses a sparse cost matrix (infinity for distances $> 0.15$) to maintain performance and accuracy.