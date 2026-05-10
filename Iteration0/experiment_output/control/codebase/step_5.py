# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pickle
import json
from scipy import stats
from scipy.optimize import curve_fit

def power_law(r, A, n):
    return A * r**(-n)

def screened(r, A, lam):
    return A * np.exp(-r / lam) / r

def yukawa(r, A, lam):
    return A * np.exp(-r / lam) / r**2

if __name__ == '__main__':
    data_dir = 'data/'
    with open(os.path.join(data_dir, 'unwrapped_trajs.pkl'), 'rb') as f:
        unwrapped_trajs = pickle.load(f)
    time_to_vorts = {}
    for tid, traj in unwrapped_trajs.items():
        ts = traj['t']
        xs = traj['x']
        ys = traj['y']
        zs = traj['z']
        sizes = traj['size']
        omegas = traj['omega_tot']
        if len(ts) >= 3:
            vx = np.gradient(xs, ts)
            vy = np.gradient(ys, ts)
            vz = np.gradient(zs, ts)
            ax = np.gradient(vx, ts)
            ay = np.gradient(vy, ts)
            az = np.gradient(vz, ts)
            a_mag = np.sqrt(ax**2 + ay**2 + az**2)
            for i in range(len(ts)):
                t = ts[i]
                if t not in time_to_vorts:
                    time_to_vorts[t] = {}
                time_to_vorts[t][tid] = {'x': xs[i], 'y': ys[i], 'z': zs[i], 'a_mag': a_mag[i], 'omega_tot': omegas[i], 'size': sizes[i]}
    accel_data_list = []
    for t, vorts in time_to_vorts.items():
        tids = list(vorts.keys())
        n_v = len(tids)
        if n_v < 2:
            continue
        pos = np.array([[vorts[tid]['x'], vorts[tid]['y'], vorts[tid]['z']] for tid in tids])
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        diff -= np.round(diff)
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_dist_idx = np.argmin(dist, axis=1)
        min_dists = dist[np.arange(n_v), min_dist_idx]
        for i, tid in enumerate(tids):
            nearest_idx = min_dist_idx[i]
            nearest_tid = tids[nearest_idx]
            r_nearest = min_dists[i]
            a_mag = vorts[tid]['a_mag']
            omega_tot = vorts[tid]['omega_tot']
            omega_nearest = vorts[nearest_tid]['omega_tot']
            size = vorts[tid]['size']
            accel_data_list.append([r_nearest, a_mag, omega_tot, omega_nearest, size])
    accel_data = np.array(accel_data_list)
    r_bins = np.linspace(0.02, 0.45, 21)
    r_mids = (r_bins[:-1] + r_bins[1:]) / 2.0
    bin_indices = np.digitize(accel_data[:, 0], r_bins)
    median_a_mag = np.zeros(20)
    for i in range(1, 21):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            median_a_mag[i-1] = np.median(accel_data[mask, 1])
        else:
            median_a_mag[i-1] = np.nan
    valid_mask = ~np.isnan(median_a_mag)
    log_r = np.log(r_mids[valid_mask])
    log_a = np.log(median_a_mag[valid_mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_a)
    print('Linear regression on binned medians:')
    print('Slope n: ' + str(slope))
    print('Intercept: ' + str(intercept))
    print('R^2: ' + str(r_value**2))
    r_raw = accel_data[:, 0]
    a_raw = accel_data[:, 1]
    mask_raw = (r_raw > 0) & np.isfinite(r_raw) & np.isfinite(a_raw)
    r_fit = r_raw[mask_raw]
    a_fit = a_raw[mask_raw]
    models = {'power_law': {'func': power_law, 'p0': [0.001, 1.5]}, 'screened': {'func': screened, 'p0': [0.001, 0.1]}, 'yukawa': {'func': yukawa, 'p0': [0.001, 0.1]}}
    results = {}
    best_aic = np.inf
    best_model = None
    N = len(r_fit)
    for name, m in models.items():
        try:
            popt, pcov = curve_fit(m['func'], r_fit, a_fit, p0=m['p0'], bounds=(0, np.inf), maxfev=10000)
            a_pred = m['func'](r_fit, *popt)
            ssr = np.sum((a_fit - a_pred)**2)
            k = len(popt)
            sigma2 = ssr / N
            log_L = -N / 2.0 * np.log(2.0 * np.pi * sigma2) - ssr / (2.0 * sigma2)
            aic = 2.0 * k - 2.0 * log_L
            sst = np.sum((a_fit - np.mean(a_fit))**2)
            r2 = 1.0 - ssr / sst
            results[name] = {'popt': popt.tolist(), 'aic': float(aic), 'r2': float(r2)}
            if aic < best_aic:
                best_aic = aic
                best_model = name
        except Exception as e:
            print('Fit failed for ' + name + ': ' + str(e))
    print('\nModel Comparison:')
    for name, res in results.items():
        print('Model: ' + name)
        print('  Parameters: ' + str(res['popt']))
        print('  AIC: ' + str(res['aic']))
        print('  R^2: ' + str(res['r2']))
    print('\nBest model by AIC: ' + str(best_model))
    np.save(os.path.join(data_dir, 'accel_data.npy'), accel_data)
    force_vs_r = np.column_stack((r_mids, median_a_mag))
    np.save(os.path.join(data_dir, 'force_vs_r.npy'), force_vs_r)
    with open(os.path.join(data_dir, 'model_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)