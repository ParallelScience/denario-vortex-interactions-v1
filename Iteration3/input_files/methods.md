1. **Adaptive Vortex Identification and Sensitivity Analysis**:
   - Perform a sensitivity analysis on the Q-criterion threshold by computing vortex population statistics (count, volume, total circulation) across a range of thresholds (0.5 to 2.0 standard deviations above the mean).
   - Select a threshold that maximizes the detection of coherent structures while ensuring topological connectivity.
   - Use `scipy.ndimage` labeling with periodic boundary padding to extract 3D vortex cores as connected components.

2. **Lagrangian Trajectory Construction with Persistence Filtering**:
   - Implement a tracking algorithm using the Hungarian method, utilizing 3D mask Intersection-over-Union (IoU) as the primary cost function to capture physical identity across snapshots.
   - Explicitly record "parent/child" relationships to map shredding (splitting) and merger (coalescence) events.
   - Apply a persistence filter to exclude trajectories lasting fewer than 3–5 snapshots, removing numerical noise and transient fluctuations.

3. **Vortex Stretching and Shredding Metrics**:
   - For each tracked vortex, compute the Lagrangian evolution of the vorticity vector $\vec{\omega}$ and the velocity gradient tensor $\mathbf{A} = \nabla \vec{u}$.
   - Calculate the strain-to-rotation ratio $\alpha = |S|/|\Omega|$ and the vortex stretching term $\vec{\omega} \cdot \nabla \vec{u}$.
   - Define the shredding metric by tracking the evolution of peak vorticity $\omega_{max}$, core volume $V_{core}$, and the circulation $\Gamma$. Normalize $d\Gamma/dt$ by $dV_{core}/dt$ to distinguish between physical dissipation and geometric fragmentation.

4. **Conditional Averaging of the Velocity Gradient**:
   - Compute the average strain $\langle \mathbf{S} \rangle$ and rotation $\langle \mathbf{\Omega} \rangle$ tensors in shells around vortex centroids.
   - Normalize the radial distance $r$ by the local effective vortex radius $R_{eff}$ to ensure scale-invariance.
   - Bin the results by the local strain-to-rotation ratio $\alpha$ of the vortex core to characterize how "strong" (rotation-dominated) versus "weak" (strain-dominated) vortices influence their local environment.

5. **Helicity and Lifetime Correlation**:
   - Calculate the internal helicity $H = \vec{v} \cdot \vec{\omega}$ for each identified core.
   - Perform statistical analysis to determine if high-helicity cores exhibit longer lifetimes or higher resistance to strain-induced shredding.
   - Fit the survival function of vortex lifetimes to determine if the decay process follows a characteristic timescale related to the local eddy turnover time $\tau_{eddy}$.

6. **Lagrangian Deformation Analysis**:
   - Compute the time-integrated deformation rate by integrating the strain tensor $\mathbf{S}$ along the Lagrangian trajectory.
   - Map the relationship between cumulative strain and the loss of circulation $\Gamma$ to identify the critical strain threshold $\alpha_c$ at which topological integrity is lost.
   - Generate a 2D histogram of $\alpha$ vs. $d\Gamma/dt$ to visually demonstrate the stability boundary of vortex cores.

7. **Data Management and Storage**:
   - Store all Lagrangian data (snapshot, vortex_id, centroid, $\Gamma$, $V_{core}$, $\alpha$, $H$, parent/child IDs) in a structured HDF5 file using a "long-format" table.
   - Ensure the structure supports variable-length trajectories (ragged arrays) to accommodate the diverse lifespans of vortices.

8. **Stability Map and Effective Theory Formulation**:
   - Construct a 2D histogram of $\alpha$ (strain-to-rotation) vs. $\Gamma$ (circulation) for all vortices across all time steps.
   - Color-code the map by the subsequent "survival probability" or "lifetime" of the vortices.
   - Use this map as the empirical "effective theory" to predict the stability and fate of a vortex based on its current state and local environment.