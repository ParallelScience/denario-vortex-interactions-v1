The "stability budget" model, while successful in predicting vortex lifetime via cumulative strain, currently ignores the role of **vortex-vortex alignment (helicity density)** in mediating the local strain field. I hypothesize that the observed "strain-stabilization" effect is not a global property of the background flow, but a local consequence of **vortex-vortex clustering**: vortices that align their vorticity vectors with the local principal strain axis (the "strain-vorticity alignment" phenomenon) experience reduced effective dissipation. 

To test this, we will:
1. Compute the local alignment angle $\theta$ between the vortex core's vorticity vector $\vec{\omega}$ and the local strain tensor's intermediate eigenvector $\vec{e}_2$ (the direction of least strain).
2. Decompose the `cum_strain` metric into two components: $\alpha_{aligned}$ (strain experienced while $\vec{\omega} \parallel \vec{e}_2$) and $\alpha_{misaligned}$.
3. Test the hypothesis that $\alpha_{aligned}$ is negatively correlated with the rate of circulation decay ($d\Gamma/dt$), whereas $\alpha_{misaligned}$ is strongly positively correlated with decay. 

This will determine if vortex longevity is a result of "geometric shielding" (alignment with the strain field) rather than just the magnitude of the strain, providing a refined, orientation-dependent sub-grid scale parameterization for vortex stability.