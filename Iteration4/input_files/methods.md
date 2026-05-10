1. **Feature Extraction and Vortex Identification**:
   - Compute Q-criterion, vorticity, and velocity gradient tensors for all 200 snapshots.
   - Identify vortex cores using the Q-criterion threshold (mean + 1.5*std) and `scipy.ndimage.label` with periodic padding.
   - For each core, extract: total circulation ($\Gamma$), core volume ($V_{core}$), and local helicity alignment ($\cos \theta = \frac{\vec{u} \cdot \vec{\omega}}{|\vec{u}| |\vec{\omega}|}$).
   - To isolate intrinsic vortex properties, calculate helicity alignment by subtracting the large-scale velocity field (filtered at $k > 3$) from the local velocity.

2. **Lagrangian Trajectory Construction**:
   - Link centroids across snapshots using the Hungarian algorithm with 3D mask IoU as the cost function.
   - Maintain a registry of vortex IDs, recording birth/death snapshots and parent/child relationships.
   - Filter out transient noise by excluding trajectories with a duration of fewer than 5 snapshots.

3. **Lagrangian Stability Metrics Calculation**:
   - Compute the strain tensor $S$ and rotation tensor $\Omega$ relative to the vortex's own frame of reference (subtracting the vortex's mean velocity/rotation).
   - Define the stability metric $\alpha = \langle |S| \rangle / \langle |\Omega| \rangle$, where $\langle \cdot \rangle$ is the time-average over the entire lifespan of the vortex.
   - Normalize $\alpha$ by the vortex's internal turnover time ($\tau_{vortex} \sim \Gamma / \text{Area}$) to obtain a dimensionless environmental strain measure.
   - Track $V_{core}$ evolution to distinguish between "stretching" (amplification) and "shredding" (topological loss).

4. **Spatial Distribution and Driving Correlation**:
   - Map vortex coordinates and categorize them based on local stretching ($\vec{\omega} \cdot \nabla \vec{u}$).
   - Calculate the radial distribution function $g(r)$ for vortices partitioned by their stretching sign and proximity to energy injection scales (wavenumbers 1–3).

5. **Survival Analysis Modeling**:
   - Implement a Cox Proportional Hazards model to predict vortex survival, accounting for censored data (vortices present at the start/end of the sequence).
   - Use the feature set $\{\Gamma, \alpha, \text{helicity\_alignment}\}$ as predictors.
   - Include an interaction term between $\alpha$ and helicity alignment to quantify the protective "buffer" effect of helicity in high-strain environments.

6. **Generalization and Temporal Invariance**:
   - Split the 200 snapshots into a training set (first 150) and a testing set (final 50).
   - Train the survival model on the first 150 snapshots and evaluate its predictive power on the final 50 to ensure the stability criterion is a generalizable physical property of the flow.

7. **Effective Theory Formulation**:
   - Synthesize the survival model into a stability criterion: $\tau_{life} = f(\Gamma, \alpha, \text{alignment})$.
   - Validate the criterion by comparing predicted survival probabilities against observed decay rates in the hold-out test set.

8. **Data Organization and Final Reporting**:
   - Consolidate all Lagrangian trajectory data, stability metrics, and model coefficients into a structured HDF5 file.
   - Generate diagnostic plots (e.g., survival probability vs. normalized strain, helicity alignment distributions) to support conclusions on vortex stability and sub-grid parameterization.