# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import time
import multiprocessing
import numpy as np
import pickle
import json
from scipy import stats
from scipy import ndimage
import pyvista as pv

def compute_alpha_for_snapshot(fpath):
    basename = os.path.basename(fpath)
    file_idx = int(basename.split('.')[-2])
    try:
        mesh = pv.read(fpath)
        array_names = mesh.array_names
        if 'velx' in array_names and 'vely' in array_names and 'velz' in array_names:
            velx = mesh['velx'].reshape(128, 128, 128)
            vely = mesh['vely'].reshape(128, 128, 128)
            velz = mesh['velz'].reshape(128, 128, 128)
        elif 'vel' in array_names:
            vel = mesh['vel']
            velx = vel[:, 0].reshape(128, 128, 128)
            vely = vel[:, 1].reshape(128, 128, 128)
            velz = vel[:, 2].reshape(128, 128, 128)
        elif 'velocity' in array_names:
            vel = mesh['velocity']
            velx = vel[:, 0].reshape(128, 128, 128)
            vely = vel[:, 1].reshape(128, 128, 128)
            velz = vel[:, 2].reshape(128, 128, 128)
        else:
            return file_idx, []
    except Exception:
        return file_idx, []
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
    Syz = 0.5 * (dvy_dz + dvy_dy)
    S2 = Sxx**2 + Syy**2 + Szz**2 + 2.0 * (Sxy**2 + Sxz**2 + Syz**2)
    Omxy = 0.5 * (dvx_dy - dvy_dx)
    Omxz = 0.5 * (dvx_dz - dvz_dx)
    Omyz = 0.5 * (dvy_dz - dvy_dy)
    Om2 = 2.0 * (Omxy**2 + Omxz**2 + Omyz**2)
    Q = 0.5 * (Om2 - S2)
    Q_pos = Q[Q > 0]
    if len(Q_pos) == 0:
        return file_idx, []
    threshold = np.mean(Q_pos) + 1.5 * np.std(Q_pos)
    mask = Q > threshold
    padded = np.pad(mask, 2, mode='wrap')
    labeled_pad, n = ndimage.label(padded)
    labeled = labeled_pad[2:-2, 2:-2, 2:-2]
    labels, counts = np.unique(labeled, return_counts=True)
    valid_labels = labels[(counts >= 8) & (labels > 0)]
    S_mag = np.sqrt(np.maximum(S2, 0))
    Om_mag = np.sqrt(np.maximum(Om2, 0))
    ratio = S_mag / (Om_mag + 1e-12)
    alphas = []
    for lb in valid_labels:
        idx = np.argwhere(labeled == lb)
        w = omega_mag[idx[:, 0], idx[:, 1], idx[:, 2]]
        w_sum = w.sum()
        if w_sum == 0:
            continue
        mean_alpha = np.mean(ratio[idx[:, 0], idx[:, 1], idx[:, 2]])
        alphas.append(mean_alpha)
    return file_idx, alphas

if __name__ == '__main__':
    data_dir = 'data/'
    with open(os.path.join(data_dir, 'unwrapped_trajs.pkl'), 'rb') as f:
        unwrapped_trajs = pickle.load(f)
    dt_snap = 0.05
    lifetimes = np.array([len(traj['t']) * dt_snap for traj in unwrapped_trajs.values()])
    loc_exp, scale_exp = stats.expon.fit(lifetimes)
    ks_stat_exp, ks_p_exp = stats.kstest(lifetimes, 'expon', args=(loc_exp, scale_exp))
    logL_exp = np.sum(stats.expon.logpdf(lifetimes, loc_exp, scale_exp))
    AIC_exp = 2*2 - 2*logL_exp
    b_par, loc_par, scale_par = stats.pareto.fit(lifetimes)
    ks_stat_par, ks_p_par = stats.kstest(lifetimes, 'pareto', args=(b_par, loc_par, scale_par))
    logL_par = np.sum(stats.pareto.logpdf(lifetimes, b_par, loc_par, scale_par))
    AIC_par = 2*3 - 2*logL_par
    print("Mean lifetime: " + str(np.mean(lifetimes)))
    print("Exponential fit: scale = " + str(scale_exp) + ", KS stat = " + str(ks_stat_exp) + ", p-value = " + str(ks_p_exp) + ", AIC = " + str(AIC_exp))
    print("Pareto fit: b = " + str(b_par) + ", scale = " + str(scale_par) + ", KS stat = " + str(ks_stat_par) + ", p-value = " + str(ks_p_par) + ", AIC = " + str(AIC_par))
    if AIC_exp < AIC_par: print("Exponential model is preferred by AIC.")
    else: print("Pareto (power-law) model is preferred by AIC.")
    np.save(os.path.join(data_dir, 'lifetimes.npy'), lifetimes)
    all_vortices = np.load(os.path.join(data_dir, 'all_vortices.npy'), allow_pickle=True)
    sizes = np.array([v['size'] for vorts in all_vortices for v in vorts])
    omegas = np.array([v['omega_tot'] for vorts in all_vortices for v in vorts])
    mask_size = sizes >= 4
    beta, _, r_value_size, _, _ = stats.linregress(np.log(sizes[mask_size]), np.log(omegas[mask_size]))
    b_size, _, _ = stats.pareto.fit(sizes)
    print("Log-log regression (size >= 4): beta = " + str(beta) + ", R^2 = " + str(r_value_size**2))
    print("Power law fit to sizes: exponent b = " + str(b_size))
    with open(os.path.join(data_dir, 'size_omega_scaling.json'), 'w') as f:
        json.dump({'beta': float(beta), 'R2': float(r_value_size**2), 'pareto_b': float(b_size)}, f, indent=4)
    all_files = sorted(glob.glob('/home/node/work/projects/ns_turbulence_vortex/data/Turb.hydro_w.*.vtk'))[::5]
    valid_files = [f for f in all_files if '19903' not in f]
    fpath_mid = valid_files[len(valid_files)//2]
    try:
        mesh_mid = pv.read(fpath_mid)
        array_names = mesh_mid.array_names
        if 'velx' in array_names and 'vely' in array_names and 'velz' in array_names:
            velx = mesh_mid['velx'].reshape(128, 128, 128)
            vely = mesh_mid['vely'].reshape(128, 128, 128)
            velz = mesh_mid['velz'].reshape(128, 128, 128)
        elif 'vel' in array_names:
            vel = mesh_mid['vel']
            velx = vel[:, 0].reshape(128, 128, 128)
            vely = vel[:, 1].reshape(128, 128, 128)
            velz = vel[:, 2].reshape(128, 128, 128)
        elif 'velocity' in array_names:
            vel = mesh_mid['velocity']
            velx = vel[:, 0].reshape(128, 128, 128)
            vely = vel[:, 1].reshape(128, 128, 128)
            velz = vel[:, 2].reshape(128, 128, 128)
        vel_stack = np.stack([velx, vely, velz])
        v_k = np.fft.fftn(vel_stack, axes=(1, 2, 3))
        E_3D = 0.5 * np.sum(np.abs(v_k)**2, axis=0) / (128**3)**2
        kx = np.fft.fftfreq(128, d=1.0/128.0)
        Kx, Ky, Kz = np.meshgrid(kx, kx, kx, indexing='ij')
        K_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        k_bins = np.arange(0.5, 65, 1.0)
        k_mids = 0.5 * (k_bins[1:] + k_bins[:-1])
        E_k = np.array([np.sum(E_3D[(K_mag >= k_bins[i]) & (K_mag < k_bins[i+1])]) for i in range(len(k_mids))])
        mask_fit = (k_mids >= 4) & (k_mids <= 20)
        s, _, r_value_E, _, _ = stats.linregress(np.log(k_mids[mask_fit]), np.log(E_k[mask_fit]))
        print("Energy spectrum fit (k=4..20): slope s = " + str(s) + ", R^2 = " + str(r_value_E**2))
        print("Comparison with K41 (-5/3 = -1.667): diff = " + str(abs(s - (-5.0/3.0))))
        np.save(os.path.join(data_dir, 'energy_spectrum.npy'), np.column_stack((k_mids, E_k)))
    except Exception as e: print('Failed to compute energy spectrum: ' + str(e))
    with multiprocessing.Pool(8) as pool:
        results = sorted(pool.map(compute_alpha_for_snapshot, valid_files), key=lambda x: x[0])
    snapshot_alphas = [r[1] for r in results]
    sim_times = np.load(os.path.join(data_dir, 'sim_times.npy'))
    mean_alphas, valid_lifetimes = [], []
    for tid, traj in unwrapped_trajs.items():
        traj_alphas = []
        for i in range(len(traj['t'])):
            t = traj['t'][i]
            snap_idx = np.argmin(np.abs(sim_times - t))
            if snap_idx >= len(snapshot_alphas): continue
            vorts = all_vortices[snap_idx]
            size = traj['size'][i]
            omega_tot = traj['omega_tot'][i]
            best_j = -1
            for j, v in enumerate(vorts):
                if v['size'] == size and np.isclose(v['omega_tot'], omega_tot, rtol=1e-5):
                    best_j = j
                    break
            if best_j != -1 and best_j < len(snapshot_alphas[snap_idx]):
                traj_alphas.append(snapshot_alphas[snap_idx][best_j])
        if traj_alphas:
            mean_alphas.append(np.mean(traj_alphas))
            valid_lifetimes.append(len(traj['t']) * dt_snap)
    pearson_r, p_val_alpha = stats.pearsonr(mean_alphas, valid_lifetimes)
    print("Pearson r between mean_alpha and lifetime: " + str(pearson_r) + " (p-value: " + str(p_val_alpha) + ")")
    with open(os.path.join(data_dir, 'deformation_vs_lifetime.json'), 'w') as f:
        json.dump({'pearson_r': float(pearson_r), 'p_value': float(p_val_alpha)}, f, indent=4)