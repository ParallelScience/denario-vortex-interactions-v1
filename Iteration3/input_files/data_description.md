
# 3D NS Turbulence — Vortex Interaction Effective Theory

## Scientific Goal

Identify coherent vortex structures in a 3D driven turbulence simulation, track their centres of vorticity over time, and discover the **effective theory of vortex interactions** — specifically:
1. The pairwise interaction potential V(r) between vortex centres (power-law, screened Coulomb, Yukawa?)
2. The statistical nature of vortex centre motion (Gaussian random walk vs. Lévy flight)
3. The vortex lifetime distribution and decay/merger statistics
4. How vortex circulation (total vorticity) scales with vortex size
5. The relationship between vortex clustering (radial distribution function g(r)) and the interaction potential

## Data

**200 evenly-spaced VTK snapshots** (every 5th file from the full 1001-snapshot sequence):

- **Files**: `/home/node/work/projects/ns_turbulence_vortex/data/Turb.hydro_w.NNNNN.vtk`
  where NNNNN ∈ {18903, 18908, 18913, ..., 19898} (stride=5, giving 200 files)
- **Format**: Legacy VTK binary, DATASET STRUCTURED_POINTS
- **Grid**: 128×128×128 cells, domain [-0.5, 0.5]³, periodic, dx = 1/128 ≈ 0.0078125
- **Time cadence**: Δt = 0.05 between selected snapshots (0.01 × stride=5)
- **Time range**: t = 189.03 to ~199.0 (total span ~10 simulation time units)

**How to list the 200 files in Python:**
```python
import glob, numpy as np
all_files = sorted(glob.glob('/home/node/work/projects/ns_turbulence_vortex/data/Turb.hydro_w.*.vtk'))
files = all_files[::5]   # every 5th = 200 files
print(len(files))  # should be 200 (or 201)
```

**How to read a single file:**
```python
import pyvista as pv
import numpy as np
mesh = pv.read('/home/node/work/projects/ns_turbulence_vortex/data/Turb.hydro_w.18903.vtk')
velx = mesh['velx'].reshape(128, 128, 128)
vely = mesh['vely'].reshape(128, 128, 128)
velz = mesh['velz'].reshape(128, 128, 128)
dens = mesh['dens'].reshape(128, 128, 128)
```

**Fields per file (cell-centred scalars, float32):**
| Field | Range | Description |
|-------|-------|-------------|
| dens  | [0.985, 1.007] | Mass density (near-uniform, isothermal) |
| velx  | [-0.87, +0.69] | Velocity x-component |
| vely  | [-0.69, +0.72] | Velocity y-component |
| velz  | [-0.80, +0.85] | Velocity z-component |

## Simulation Parameters

- Code: AthenaK, 3D driven turbulence
- EOS: Isothermal, c_s = 5.0 (subsonic: Mach ≈ 0.16)
- Driving: solenoidal (div-free), dedt = 1×10⁻⁴, τ_corr = 5.0, wavenumbers n=1–3
- Boundary: periodic on all faces
- Time integrator: RK2, CFL=0.3, Riemann: HLLE, reconstruction: PLM

## Vorticity and Q-criterion

Vorticity: ω = ∇×v. Components using np.gradient with dx=1/128:
```python
dx = 1.0/128
omega_x = np.gradient(velz, dx, axis=1) - np.gradient(vely, dx, axis=2)
omega_y = np.gradient(velx, dx, axis=2) - np.gradient(velz, dx, axis=0)
omega_z = np.gradient(vely, dx, axis=0) - np.gradient(velx, dx, axis=1)
omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
# Observed: mean ≈ 6.4, max ≈ 28.4
```

Q-criterion (Q > 0 = rotation-dominated = vortex core):
```python
# velocity gradient tensor A_ij = dv_i/dx_j
# S = symmetric part (strain), Omega = antisymmetric part (rotation)
# Q = 0.5*(|Omega|^2 - |S|^2)
Sxx=dvx_dx; Syy=dvy_dy; Szz=dvz_dz
Sxy=0.5*(dvx_dy+dvy_dx); Sxz=0.5*(dvx_dz+dvz_dx); Syz=0.5*(dvy_dz+dvz_dy)
S2 = Sxx**2+Syy**2+Szz**2 + 2*(Sxy**2+Sxz**2+Syz**2)
Omxy=0.5*(dvx_dy-dvy_dx); Omxz=0.5*(dvx_dz-dvz_dx); Omyz=0.5*(dvy_dz-dvz_dy)
Om2 = 2*(Omxy**2+Omxz**2+Omyz**2)
Q = 0.5*(Om2 - S2)
# Threshold: Q > mean(Q[Q>0]) + 1.5*std(Q[Q>0])
# Typical vortices per snapshot: ~100–150
```

**Vortex centroid** (vorticity-weighted, with periodic BC correction using complex exponential trick):
```python
from scipy import ndimage
mask = Q > threshold
padded = np.pad(mask, 2, mode='wrap')
labeled_pad, n = ndimage.label(padded)
labeled = labeled_pad[2:-2, 2:-2, 2:-2]
for lb in range(1, n+1):
    idx = np.argwhere(labeled == lb)
    w = omega_mag[idx[:,0], idx[:,1], idx[:,2]]
    # periodic centroid per axis:
    for dim, N in zip([0,1,2],[128,128,128]):
        angles = 2*np.pi*idx[:,dim]/N
        z = np.sum(w * np.exp(1j*angles))/w.sum()
        centroid_coord = np.angle(z)/(2*np.pi)*1.0 - 0.5 + dx/2
```

## Periodic Boundary Conditions

Minimum-image distance between two vortex positions a, b in [-0.5, 0.5]³:
```python
d = a - b
d -= np.round(d)  # wrap to [-0.5, 0.5]
r = np.sqrt(np.sum(d**2))
```

## Vortex Tracking

Link vortex centroids across consecutive snapshots via greedy nearest-neighbour:
- max_matching_distance = 0.10 (in domain units)
- Handle vortex birth/death: unmatched vortices start/end trajectories

## Key Analyses to Perform

1. **Vortex identification**: Q-criterion labeling → ~100–150 vortices per snapshot
2. **Trajectory building**: track centroids across 200 snapshots
3. **Step displacement statistics**: kurtosis (expect >>3 if non-Gaussian), Lévy fit
4. **MSD analysis**: fit MSD(τ) ~ τ^(2H), extract Hurst exponent H
5. **RDF g(r)**: radial distribution function of vortex centroids → clustering signature
6. **Effective force law**: bin centroid acceleration |a| vs nearest-neighbour distance r → fit power law or screened potential
7. **Vortex size-circulation scaling**: omega_tot ~ size^β
8. **Lifetime distribution**: fit exponential or power-law survival function
9. **Energy spectrum**: E(k) ~ k^(-5/3) Kolmogorov validation
10. **Interaction model selection**: compare power-law V(r)~r^(-n), screened-exp, Yukawa via AIC/BIC

## Memory/Performance Notes

- Use 8 parallel workers (multiprocessing.Pool) for reading + Q-criterion computation
- Each file: ~40 MB memory during processing, released after centroid extraction
- With 200 files × 8 workers: completes in ~30–60 seconds
- Total centroid data: 200 snapshots × ~120 vortices × few floats ≈ negligible RAM
- Set OMP_NUM_THREADS=1 to avoid OpenBLAS thread oversubscription
