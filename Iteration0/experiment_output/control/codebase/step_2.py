# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pickle
import os

def unwrap_coord(c):
    c_unw = np.copy(c)
    offset = 0.0
    for k in range(1, len(c_unw)):
        c_unw[k] += offset
        diff = c_unw[k] - c_unw[k-1]
        if diff > 0.5:
            offset -= 1.0
            c_unw[k] -= 1.0
        elif diff < -0.5:
            offset += 1.0
            c_unw[k] += 1.0
    return c_unw

if __name__ == '__main__':
    all_vortices = np.load('data/all_vortices.npy', allow_pickle=True)
    sim_times = np.load('data/sim_times.npy')
    dt_snap = sim_times[1] - sim_times[0]
    print('Time step between snapshots (dt_snap): ' + str(round(dt_snap, 4)))
    trajectories = {}
    active_tracks = {}
    next_track_id = 0
    for snap_idx, vortices in enumerate(all_vortices):
        t = sim_times[snap_idx]
        if not active_tracks:
            for v in vortices:
                v_record = {'t': t, 'x': v['x'], 'y': v['y'], 'z': v['z'], 'size': v['size'], 'omega_tot': v['omega_tot']}
                trajectories[next_track_id] = [v_record]
                active_tracks[next_track_id] = v_record
                next_track_id += 1
            continue
        active_tids = list(active_tracks.keys())
        if len(vortices) > 0 and len(active_tids) > 0:
            pos_active = np.array([[active_tracks[tid]['x'], active_tracks[tid]['y'], active_tracks[tid]['z']] for tid in active_tids])
            pos_current = np.array([[v['x'], v['y'], v['z']] for v in vortices])
            diff = pos_active[:, np.newaxis, :] - pos_current[np.newaxis, :, :]
            diff -= np.round(diff)
            dist = np.sqrt(np.sum(diff**2, axis=2))
            matched_active = set()
            matched_current = set()
            flat_dist = dist.flatten()
            sorted_indices = np.argsort(flat_dist)
            for idx in sorted_indices:
                d = flat_dist[idx]
                if d > 0.10:
                    break
                i = idx // len(vortices)
                j = idx % len(vortices)
                tid = active_tids[i]
                if tid not in matched_active and j not in matched_current:
                    matched_active.add(tid)
                    matched_current.add(j)
                    v = vortices[j]
                    v_record = {'t': t, 'x': v['x'], 'y': v['y'], 'z': v['z'], 'size': v['size'], 'omega_tot': v['omega_tot']}
                    trajectories[tid].append(v_record)
                    active_tracks[tid] = v_record
            for tid in active_tids:
                if tid not in matched_active:
                    del active_tracks[tid]
            for j, v in enumerate(vortices):
                if j not in matched_current:
                    v_record = {'t': t, 'x': v['x'], 'y': v['y'], 'z': v['z'], 'size': v['size'], 'omega_tot': v['omega_tot']}
                    trajectories[next_track_id] = [v_record]
                    active_tracks[next_track_id] = v_record
                    next_track_id += 1
        else:
            active_tracks = {}
            for v in vortices:
                v_record = {'t': t, 'x': v['x'], 'y': v['y'], 'z': v['z'], 'size': v['size'], 'omega_tot': v['omega_tot']}
                trajectories[next_track_id] = [v_record]
                active_tracks[next_track_id] = v_record
                next_track_id += 1
    long_tracks = {tid: traj for tid, traj in trajectories.items() if len(traj) >= 5}
    total_trajs = len(trajectories)
    long_trajs_count = len(long_tracks)
    lengths = [len(traj) for traj in long_tracks.values()]
    print('Total trajectories formed: ' + str(total_trajs))
    print('Long trajectories (>=5 points): ' + str(long_trajs_count))
    if lengths:
        print('Track length stats - Min: ' + str(np.min(lengths)) + ', Max: ' + str(np.max(lengths)) + ', Mean: ' + str(round(np.mean(lengths), 2)) + ', Median: ' + str(np.median(lengths)))
    unwrapped_trajs = {}
    traj_data = []
    for tid, traj in long_tracks.items():
        t_arr = np.array([p['t'] for p in traj])
        x_arr = np.array([p['x'] for p in traj])
        y_arr = np.array([p['y'] for p in traj])
        z_arr = np.array([p['z'] for p in traj])
        size_arr = np.array([p['size'] for p in traj])
        omega_arr = np.array([p['omega_tot'] for p in traj])
        x_unw = unwrap_coord(x_arr)
        y_unw = unwrap_coord(y_arr)
        z_unw = unwrap_coord(z_arr)
        unwrapped_trajs[tid] = {'t': t_arr, 'x': x_unw, 'y': y_unw, 'z': z_unw, 'size': size_arr, 'omega_tot': omega_arr}
        for i in range(len(traj)):
            traj_data.append([tid, t_arr[i], x_arr[i], y_arr[i], z_arr[i], size_arr[i], omega_arr[i]])
    traj_array = np.array(traj_data)
    np.save('data/trajectories.npy', traj_array)
    with open('data/unwrapped_trajs.pkl', 'wb') as f:
        pickle.dump(unwrapped_trajs, f)
    print('Saved data/trajectories.npy and data/unwrapped_trajs.pkl')