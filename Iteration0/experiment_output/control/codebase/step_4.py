# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pickle

if __name__ == '__main__':
    data_dir = 'data/'
    
    with open(os.path.join(data_dir, 'unwrapped_trajs.pkl'), 'rb') as f:
        unwrapped_trajs = pickle.load(f)
    sim_times = np.load(os.path.join(data_dir, 'sim_times.npy'))
    
    time_to_vorts = {i: {} for i in range(len(sim_times))}
    for tid, traj in unwrapped_trajs.items():
        for i in range(len(traj['t'])):
            t = traj['t'][i]
            idx = np.argmin(np.abs(sim_times - t))
            time_to_vorts[idx][tid] = {
                'x': traj['x'][i],
                'y': traj['y'][i],
                'z': traj['z'][i],
                'size': traj['size'][i],
                'omega_tot': traj['omega_tot'][i]
            }
            
    pair_dist_list = []
    total_pairs = 0
    for idx, vorts in time_to_vorts.items():
        tids = list(vorts.keys())
        n_v = len(tids)
        if n_v >= 2:
            pos = np.array([[vorts[tid]['x'], vorts[tid]['y'], vorts[tid]['z']] for tid in tids])
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            diff -= np.round(diff)
            dist = np.sqrt(np.sum(diff**2, axis=2))
            i_upper, j_upper = np.triu_indices(n_v, k=1)
            dists_upper = dist[i_upper, j_upper]
            pair_dist_list.append(dists_upper)
            total_pairs += len(dists_upper)
            
    pair_dist_all = np.concatenate(pair_dist_list)
    print('Total pairs computed: ' + str(total_pairs))
    
    n_bins = 50
    r_edges = np.linspace(0, 0.5, n_bins + 1)
    r_mid = (r_edges[:-1] + r_edges[1:]) / 2.0
    hist, _ = np.histogram(pair_dist_all, bins=r_edges)
    
    V_shell = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    
    expected_pairs_per_bin = np.zeros(n_bins)
    for idx, vorts in time_to_vorts.items():
        N_k = len(vorts)
        if N_k >= 2:
            expected_pairs_per_bin += 0.5 * N_k * (N_k - 1) * V_shell
            
    g_r = np.zeros_like(hist, dtype=float)
    mask = expected_pairs_per_bin > 0
    g_r[mask] = hist[mask] / expected_pairs_per_bin[mask]
    
    peak_idx = np.argmax(g_r)
    peak_r = r_mid[peak_idx]
    peak_val = g_r[peak_idx]
    print('g(r) peak location: r = ' + str(round(peak_r, 4)))
    print('g(r) peak height: ' + str(round(peak_val, 4)))
    
    exc_zone_max = 0.0
    for i in range(n_bins):
        if g_r[i] < 0.5:
            exc_zone_max = r_edges[i+1]
        else:
            break
    print('Exclusion zone (g < 0.5): r < ' + str(round(exc_zone_max, 4)))
    
    if np.any(g_r > 1.0):
        print('Verdict: Clustering observed (g(r) > 1 for some r).')
    elif np.any(g_r < 1.0):
        print('Verdict: Exclusion observed (g(r) < 1 for all r).')
    else:
        print('Verdict: No spatial correlation (g(r) ~ 1).')
        
    np.save(os.path.join(data_dir, 'pair_dist_all.npy'), pair_dist_all)
    rdf_data = np.column_stack((r_mid, g_r))
    np.save(os.path.join(data_dir, 'rdf.npy'), rdf_data)
    print('Saved pair_dist_all.npy and rdf.npy to data/')