# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '1'
import pyvista as pv
import numpy as np
from scipy import ndimage
import glob
import time
import multiprocessing
def process_snapshot(fpath):
    basename = os.path.basename(fpath)
    file_idx = int(basename.split('.')[-2])
    sim_time = file_idx / 100.0
    mesh = pv.read(fpath)
    velx = mesh['velx'].reshape(128, 128, 128)
    vely = mesh['vely'].reshape(128, 128, 128)
    velz = mesh['velz'].reshape(128, 128, 128)
    dx = 1.0 / 128.0
    dvx_dx = np.gradient(velx, dx, axis=0)
    dvx_dy = np.gradient(velx, dx, axis=1)
    dvx_dz = np.gradient(velx, dx, axis=2)
    dvy_dx = np.gradient(vely, dx, axis=0)
    dvy_dy = np.gradient(vely, dx, axis=1)
    dvy_dz = np.gradient(vely, dx, axis=2)
    dvz_dx = np.gradient(velz, dx, axis=0)
    dvz_dy = np.gradient(velz, dx, axis=1)
    dvz_dz = np.gradient(velz, dx, axis=2)
    omega_x = dvz_dy - dvy_dz
    omega_y = dvx_dz - dvz_dx
    omega_z = dvy_dx - dvx_dy
    omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    Sxx = dvx_dx
    Syy = dvy_dy
    Szz = dvz_dz
    Sxy = 0.5 * (dvx_dy + dvy_dx)
    Sxz = 0.5 * (dvx_dz + dvz_dx)
    Syz = 0.5 * (dvy_dz + dvy_dz)
    S2 = Sxx**2 + Syy**2 + Szz**2 + 2.0 * (Sxy**2 + Sxz**2 + Syz**2)
    Omxy = 0.5 * (dvx_dy - dvy_dx)
    Omxz = 0.5 * (dvx_dz - dvz_dx)
    Omyz = 0.5 * (dvy_dz - dvy_dz)
    Om2 = 2.0 * (Omxy**2 + Omxz**2 + Omyz**2)
    Q = 0.5 * (Om2 - S2)
    Q_pos = Q[Q > 0]
    if len(Q_pos) == 0:
        return file_idx, sim_time, []
    threshold = np.mean(Q_pos) + 1.5 * np.std(Q_pos)
    mask = Q > threshold
    padded = np.pad(mask, 2, mode='wrap')
    labeled_pad, n = ndimage.label(padded)
    labeled = labeled_pad[2:-2, 2:-2, 2:-2]
    labels, counts = np.unique(labeled, return_counts=True)
    valid_labels = labels[(counts >= 8) & (labels > 0)]
    vortices = []
    for lb in valid_labels:
        idx = np.argwhere(labeled == lb)
        w = omega_mag[idx[:, 0], idx[:, 1], idx[:, 2]]
        w_sum = w.sum()
        if w_sum == 0:
            continue
        centroid = []
        for dim in [0, 1, 2]:
            angles = 2.0 * np.pi * idx[:, dim] / 128.0
            z_cplx = np.sum(w * np.exp(1j * angles)) / w_sum
            coord = (np.angle(z_cplx) / (2.0 * np.pi)) % 1.0 - 0.5 + dx / 2.0
            centroid.append(coord)
        vortices.append({'x': centroid[0], 'y': centroid[1], 'z': centroid[2], 'size': len(idx), 'omega_tot': w_sum, 'omega_max': np.max(w)})
    return file_idx, sim_time, vortices
if __name__ == '__main__':
    DATA_PATH = '/home/node/work/projects/ns_turbulence_vortex/data'
    all_files = sorted(glob.glob(DATA_PATH + '/Turb.hydro_w.*.vtk'))
    files = all_files[::5]
    results = []
    start_time = time.time()
    with multiprocessing.Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(process_snapshot, files)):
            results.append(res)
            if (i + 1) % 25 == 0:
                print('Processed ' + str(i + 1) + ' / ' + str(len(files)) + ' files.')
    results.sort(key=lambda x: x[0])
    file_indices = np.array([r[0] for r in results])
    sim_times = np.array([r[1] for r in results])
    all_vortices = [r[2] for r in results]
    np.save('data/all_vortices.npy', np.array(all_vortices, dtype=object), allow_pickle=True)
    np.save('data/sim_times.npy', sim_times)
    np.save('data/file_indices.npy', file_indices)