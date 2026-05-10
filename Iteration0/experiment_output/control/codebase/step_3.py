# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pickle
import json
from scipy import stats
import os

if __name__ == '__main__':
    data_dir = 'data/'
    with open(os.path.join(data_dir, 'unwrapped_trajs.pkl'), 'rb') as f:
        unwrapped_trajs = pickle.load(f)
    step_dx_list = []
    step_dy_list = []
    step_dz_list = []
    step_dr_list = []
    max_traj_len = 0
    for tid, traj in unwrapped_trajs.items():
        xs = traj['x']
        ys = traj['y']
        zs = traj['z']
        n_pts = len(xs)
        if n_pts > max_traj_len:
            max_traj_len = n_pts
        if n_pts > 1:
            dxs = np.diff(xs)
            dys = np.diff(ys)
            dzs = np.diff(zs)
            drs = np.sqrt(dxs**2 + dys**2 + dzs**2)
            step_dx_list.append(dxs)
            step_dy_list.append(dys)
            step_dz_list.append(dzs)
            step_dr_list.append(drs)
    step_dx = np.concatenate(step_dx_list)
    step_dy = np.concatenate(step_dy_list)
    step_dz = np.concatenate(step_dz_list)
    step_dr = np.concatenate(step_dr_list)
    print('Total step displacements: ' + str(len(step_dr)))
    print('Mean |dr|: ' + str(np.mean(step_dr)))
    print('Std |dr|: ' + str(np.std(step_dr)))
    print('Max |dr|: ' + str(np.max(step_dr)))
    kurt_dx = stats.kurtosis(step_dx, fisher=False)
    print('Kurtosis of dx (Fisher=False): ' + str(kurt_dx))
    dx_std = (step_dx - np.mean(step_dx)) / np.std(step_dx)
    ks_stat, ks_p = stats.kstest(dx_std, 'norm')
    print('KS test of dx vs Gaussian: stat = ' + str(ks_stat) + ', p-value = ' + str(ks_p))
    median_dx = np.median(step_dx)
    try:
        alpha_levy, beta_levy, loc_levy, scale_levy = stats.levy_stable.fit(step_dx, floc=median_dx)
        print('Levy stable fit: alpha = ' + str(alpha_levy) + ', scale = ' + str(scale_levy))
    except Exception as e:
        print('Levy stable fit failed: ' + str(e))
        alpha_levy, beta_levy, loc_levy, scale_levy = None, None, None, None
    max_lag = min(50, max_traj_len // 2)
    if max_lag < 1:
        max_lag = 1
    msd_mean = np.zeros(max_lag)
    lag_times = np.arange(1, max_lag + 1)
    for lag in range(1, max_lag + 1):
        dr2_list = []
        for tid, traj in unwrapped_trajs.items():
            xs = traj['x']
            ys = traj['y']
            zs = traj['z']
            if len(xs) > lag:
                dx = xs[lag:] - xs[:-lag]
                dy = ys[lag:] - ys[:-lag]
                dz = zs[lag:] - zs[:-lag]
                dr2 = dx**2 + dy**2 + dz**2
                dr2_list.append(dr2)
        if dr2_list:
            dr2_all = np.concatenate(dr2_list)
            msd_mean[lag-1] = np.mean(dr2_all)
    log_tau = np.log(lag_times)
    log_msd = np.log(msd_mean)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_msd)
    H = slope / 2.0
    R2 = r_value**2
    print('MSD fit (all lags): slope (2H) = ' + str(slope) + ', H = ' + str(H) + ', R^2 = ' + str(R2))
    short_mask = lag_times <= 5
    if np.sum(short_mask) > 1:
        log_tau_short = log_tau[short_mask]
        log_msd_short = log_msd[short_mask]
        slope_s, intercept_s, r_value_s, p_value_s, std_err_s = stats.linregress(log_tau_short, log_msd_short)
        H_s = slope_s / 2.0
        R2_s = r_value_s**2
        print('MSD fit (tau <= 5): slope (2H) = ' + str(slope_s) + ', H = ' + str(H_s) + ', R^2 = ' + str(R2_s))
    else:
        slope_s, H_s, R2_s = None, None, None
    np.save(os.path.join(data_dir, 'step_dx.npy'), step_dx)
    np.save(os.path.join(data_dir, 'step_dy.npy'), step_dy)
    np.save(os.path.join(data_dir, 'step_dz.npy'), step_dz)
    np.save(os.path.join(data_dir, 'step_dr.npy'), step_dr)
    msd_data = np.column_stack((lag_times, msd_mean))
    np.save(os.path.join(data_dir, 'msd.npy'), msd_data)
    fit_results = {'kurtosis_dx': float(kurt_dx), 'ks_stat_gaussian': float(ks_stat), 'ks_p_gaussian': float(ks_p), 'levy_alpha': float(alpha_levy) if alpha_levy is not None else None, 'levy_beta': float(beta_levy) if beta_levy is not None else None, 'levy_loc': float(loc_levy) if loc_levy is not None else None, 'levy_scale': float(scale_levy) if scale_levy is not None else None, 'msd_all_slope': float(slope), 'msd_all_H': float(H), 'msd_all_R2': float(R2), 'msd_short_slope': float(slope_s) if slope_s is not None else None, 'msd_short_H': float(H_s) if H_s is not None else None, 'msd_short_R2': float(R2_s) if R2_s is not None else None}
    with open(os.path.join(data_dir, 'msd_fit_results.json'), 'w') as f:
        json.dump(fit_results, f, indent=4)
    print('Saved step arrays, msd.npy, and msd_fit_results.json to data/')