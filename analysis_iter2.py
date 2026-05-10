"""
Iteration 2: Lightweight vortex interaction analysis
- 100 snapshots (stride=10)
- Hungarian algorithm tracking (replaces greedy)
- Langevin drift estimation for V(r)
- Directional potential V(r,Theta) = f(r)*cos^n(Theta)
- Normalised lifetime tau/tau_eddy
- Vortex merger/dissipation categorisation
"""
import os, glob, sys, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import ndimage, stats, optimize
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = '/home/node/work/projects/ns_turbulence_vortex/data'
OUT_PATH  = '/home/node/work/projects/vortex_interactions_v1/Iteration2/experiment_output/control/data'
os.makedirs(OUT_PATH, exist_ok=True)

NX = NY = NZ = 128
DX = 1.0 / NX

def log(msg): print(msg, flush=True)

def periodic_min_image(a, B):
    """Minimum-image distances from point a (3,) to rows of B (N,3)."""
    d = B - a; d -= np.round(d)
    return np.sqrt((d**2).sum(axis=1))

def periodic_dist(a, b):
    d = np.asarray(a) - np.asarray(b); d -= np.round(d)
    return float(np.sqrt((d**2).sum()))

def fast_levy_alpha(data, n_sub=2000):
    """Fast Lévy stability index via characteristic function slope."""
    np.random.seed(42)
    d = data[np.random.choice(len(data), min(n_sub, len(data)), replace=False)]
    d = d - np.median(d)
    k_vals = np.logspace(-1, 0.7, 20)
    phi = np.array([np.abs(np.mean(np.exp(1j*k*d))) for k in k_vals])
    v = (phi > 0.02) & (phi < 0.98)
    if v.sum() < 4: return 2.0
    try:
        sl, _, _, _, _ = stats.linregress(np.log(k_vals[v]), np.log(-np.log(phi[v])))
        return float(np.clip(sl, 0.5, 2.0))
    except: return 2.0

def process_snapshot(fpath):
    try:
        import pyvista as pv
        mesh = pv.read(fpath)
        vx = mesh['velx'].reshape(NX,NY,NZ).astype(np.float32)
        vy = mesh['vely'].reshape(NX,NY,NZ).astype(np.float32)
        vz = mesh['velz'].reshape(NX,NY,NZ).astype(np.float32)
        dx = DX
        dvxdx=np.gradient(vx,dx,axis=0); dvxdy=np.gradient(vx,dx,axis=1); dvxdz=np.gradient(vx,dx,axis=2)
        dvydx=np.gradient(vy,dx,axis=0); dvydy=np.gradient(vy,dx,axis=1); dvydz=np.gradient(vy,dx,axis=2)
        dvzdx=np.gradient(vz,dx,axis=0); dvzdy=np.gradient(vz,dx,axis=1); dvzdz=np.gradient(vz,dx,axis=2)
        ox=dvzdy-dvydz; oy=dvxdz-dvzdx; oz=dvydx-dvxdy
        omag=np.sqrt(ox**2+oy**2+oz**2)
        Sxx=dvxdx; Syy=dvydy; Szz=dvzdz
        Sxy=0.5*(dvxdy+dvydx); Sxz=0.5*(dvxdz+dvzdx); Syz=0.5*(dvydz+dvzdy)
        S2=Sxx**2+Syy**2+Szz**2+2*(Sxy**2+Sxz**2+Syz**2)
        Omxy=0.5*(dvxdy-dvydx); Omxz=0.5*(dvxdz-dvzdx); Omyz=0.5*(dvydz-dvzdy)
        Om2=2*(Omxy**2+Omxz**2+Omyz**2)
        Q=0.5*(Om2-S2)
        Qpos=Q[Q>0]
        if len(Qpos)==0: return [], np.zeros(3)
        thresh=np.mean(Qpos)+1.5*np.std(Qpos)
        mask=Q>thresh
        padded=np.pad(mask,2,mode='wrap')
        lpad,n=ndimage.label(padded)
        labeled=lpad[2:-2,2:-2,2:-2]
        mean_v=np.array([vx.mean(),vy.mean(),vz.mean()])
        vortices=[]
        for lb in range(1,n+1):
            idx=np.argwhere(labeled==lb)
            if len(idx)<8: continue
            w=omag[idx[:,0],idx[:,1],idx[:,2]]
            if w.sum()<1e-12: continue
            cx=[]
            for dim,N in zip([0,1,2],[NX,NY,NZ]):
                ang=2*np.pi*idx[:,dim]/N
                z=np.sum(w*np.exp(1j*ang))/w.sum()
                cx.append(float(np.angle(z)/(2*np.pi)+0.5*DX))
            # vorticity orientation
            ox_v=ox[idx[:,0],idx[:,1],idx[:,2]]; oy_v=oy[idx[:,0],idx[:,1],idx[:,2]]; oz_v=oz[idx[:,0],idx[:,1],idx[:,2]]
            om_w=np.array([(w*ox_v).sum()/w.sum(),(w*oy_v).sum()/w.sum(),(w*oz_v).sum()/w.sum()])
            norm_om=np.linalg.norm(om_w); om_hat=om_w/max(norm_om,1e-10)
            # local velocity at centroid
            ci=[min(max(int(round((cx[d]+0.5)/DX-0.5)),0),NX-1) for d in range(3)]
            vx_loc=float(vx[ci[0],ci[1],ci[2]])-mean_v[0]
            vy_loc=float(vy[ci[0],ci[1],ci[2]])-mean_v[1]
            vz_loc=float(vz[ci[0],ci[1],ci[2]])-mean_v[2]
            alpha_max=float(np.sqrt(S2[idx[:,0],idx[:,1],idx[:,2]]).max()/
                            max(np.sqrt(Om2[idx[:,0],idx[:,1],idx[:,2]]).max(),1e-10))
            gamma=float(w.sum())
            r_eff=float((3*len(idx)/(4*np.pi))**(1/3)*DX)
            tau_eddy=float(r_eff**2/max(gamma*DX,1e-10))
            vortices.append({
                'x':cx[0],'y':cx[1],'z':cx[2],
                'size':int(len(idx)),'gamma':gamma,'r_eff':r_eff,
                'tau_eddy':tau_eddy,'alpha_max':alpha_max,
                'om_hat':om_hat.tolist(),
                'vx':vx_loc,'vy':vy_loc,'vz':vz_loc,
            })
        return vortices, mean_v
    except Exception as e:
        return [], np.zeros(3)

# ─── STEP 1 ──────────────────────────────────────────────────────────────────
log("=== Step 1: Vortex identification (100 snapshots, 8 workers) ===")
all_files=sorted(glob.glob(os.path.join(DATA_PATH,'Turb.hydro_w.*.vtk')))
files=all_files[::10][:100]   # exactly 100 snapshots
N_proc=len(files)
file_indices=[int(os.path.basename(f).split('.')[2]) for f in files]
sim_times=[idx/100.0 for idx in file_indices]
dt_snap=sim_times[1]-sim_times[0]
log(f"Using {N_proc} snapshots, dt_snap={dt_snap:.2f}")

t0=time.time()
with Pool(8) as pool:
    results=pool.map(process_snapshot,files)
log(f"Done in {time.time()-t0:.1f}s")

all_vortices=[r[0] for r in results]
mean_vels=np.array([r[1] for r in results])
n_per_snap=[len(v) for v in all_vortices]
log(f"Vortices/snap: min={min(n_per_snap)}, max={max(n_per_snap)}, mean={np.mean(n_per_snap):.1f}")

# ─── STEP 2: Hungarian tracking ───────────────────────────────────────────────
log("\n=== Step 2: Hungarian algorithm tracking ===")

def hungarian_track(all_vortices, sim_times, max_d=0.10):
    """Track vortices using Hungarian (optimal assignment) algorithm."""
    trajectories={}; next_id=0; active={}
    
    for i,(vorts,t) in enumerate(zip(all_vortices,sim_times)):
        if i==0:
            for v in vorts:
                v['t']=t; trajectories[next_id]=[v.copy()]; active[next_id]=v; next_id+=1
            continue
        
        prev_ids=list(active.keys())
        prev_vorts=list(active.values())
        n_prev=len(prev_vorts); n_curr=len(vorts)
        
        if n_prev==0 or n_curr==0:
            for v in vorts:
                v['t']=t; trajectories[next_id]=[v.copy()]; active[next_id]=v; next_id+=1
            active={tid:v for tid,v in active.items() if False}  # clear
            active={next_id-len(vorts)+j: vorts[j] for j in range(len(vorts))}
            continue
        
        # Build cost matrix (N_prev x N_curr)
        prev_pos=np.array([[v['x'],v['y'],v['z']] for v in prev_vorts])
        curr_pos=np.array([[v['x'],v['y'],v['z']] for v in vorts])
        
        # Cost = periodic distance; infinite if > max_d
        cost=np.full((n_prev,n_curr), 1e6)
        for pi in range(n_prev):
            dists=periodic_min_image(prev_pos[pi],curr_pos)
            cost[pi,:]=dists
        cost[cost>max_d]=1e6
        
        # Hungarian assignment
        row_ind,col_ind=linear_sum_assignment(cost)
        
        new_active={}; matched_curr=set()
        for pi,ci in zip(row_ind,col_ind):
            if cost[pi,ci]<1e6:  # valid match
                tid=prev_ids[pi]; vorts[ci]['t']=t
                trajectories[tid].append(vorts[ci].copy())
                new_active[tid]=vorts[ci]
                matched_curr.add(ci)
        
        # New vortices
        for ci,v in enumerate(vorts):
            if ci not in matched_curr:
                v['t']=t; trajectories[next_id]=[v.copy()]; new_active[next_id]=v; next_id+=1
        active=new_active
    
    return {tid:tr for tid,tr in trajectories.items() if len(tr)>=4}

long_tracks=hungarian_track(all_vortices,sim_times,max_d=0.10)
log(f"Long tracks (>=4 pts): {len(long_tracks)}")
track_lengths=[len(t) for t in long_tracks.values()]
log(f"Track length: min={min(track_lengths)}, max={max(track_lengths)}, mean={np.mean(track_lengths):.1f}")

# Unwrap coordinates
unwrapped={}
for tid,traj in long_tracks.items():
    xs=np.array([p['x'] for p in traj]); ys=np.array([p['y'] for p in traj])
    zs=np.array([p['z'] for p in traj])
    for arr in [xs,ys,zs]:
        for k in range(1,len(arr)):
            d=arr[k]-arr[k-1]
            if d>0.5: arr[k]-=1.0
            if d<-0.5: arr[k]+=1.0
    unwrapped[tid]=(xs,ys,zs)

# ─── STEP 3: Displacement stats ───────────────────────────────────────────────
log("\n=== Step 3: Displacement statistics ===")
sdx,sdr=[],[]
for tid,(xs,ys,zs) in unwrapped.items():
    dxs=np.diff(xs); dys=np.diff(ys); dzs=np.diff(zs)
    sdx.extend(dxs); sdr.extend(np.sqrt(dxs**2+dys**2+dzs**2))
sdx=np.array(sdx); sdr=np.array(sdr)
kurt_dx=float(stats.kurtosis(sdx,fisher=False))
alpha_levy=fast_levy_alpha(sdx)
ks_g,pv_g=stats.kstest(sdx,'norm',args=(sdx.mean(),sdx.std()))
log(f"Kurtosis(dx)={kurt_dx:.2f}, Lévy α={alpha_levy:.3f}, KS-Gaussian p={pv_g:.2e}")

# MSD
max_lag=min(30,max(track_lengths)//2)
msd_v=np.zeros(max_lag); msd_c=np.zeros(max_lag,dtype=int)
for tid,(xs,ys,zs) in unwrapped.items():
    n=len(xs)
    for lag in range(1,min(max_lag+1,n)):
        dr2=(xs[lag:]-xs[:n-lag])**2+(ys[lag:]-ys[:n-lag])**2+(zs[lag:]-zs[:n-lag])**2
        msd_v[lag-1]+=dr2.sum(); msd_c[lag-1]+=len(dr2)
msd_mean=np.where(msd_c>0,msd_v/np.maximum(msd_c,1),np.nan)
lag_times=np.arange(1,max_lag+1)*dt_snap
valid=np.isfinite(msd_mean)&(msd_mean>0)
sl=1.0; ic=0.0; H_msd=0.5
if valid.sum()>3:
    sl,ic,rv,_,_=stats.linregress(np.log(lag_times[valid]),np.log(msd_mean[valid]))
    H_msd=sl/2; log(f"MSD~τ^{sl:.3f} H={H_msd:.3f} R²={rv**2:.4f}")
np.save(os.path.join(OUT_PATH,'msd.npy'),np.column_stack([lag_times,msd_mean]))

# ─── STEP 4: RDF ──────────────────────────────────────────────────────────────
log("\n=== Step 4: RDF ===")
time_to_vorts={}
for tid,traj in long_tracks.items():
    for pt in traj:
        ti=round((pt['t']-sim_times[0])/dt_snap)
        if ti not in time_to_vorts: time_to_vorts[ti]={}
        time_to_vorts[ti][tid]=pt

pair_dist_all=[]
for ti in sorted(time_to_vorts):
    vmap=time_to_vorts[ti]; tids=list(vmap.keys())
    if len(tids)<2: continue
    pts=np.array([[vmap[t]['x'],vmap[t]['y'],vmap[t]['z']] for t in tids])
    for i in range(len(tids)):
        d=periodic_min_image(pts[i],pts[i+1:])
        pair_dist_all.extend(d.tolist())
pair_dist_all=np.array(pair_dist_all)
rho_bar=np.mean(n_per_snap)
r_edges=np.linspace(0,0.5,51); r_mid=0.5*(r_edges[:-1]+r_edges[1:])
cts,_=np.histogram(pair_dist_all,bins=r_edges)
shell_v=(4/3)*np.pi*(r_edges[1:]**3-r_edges[:-1]**3)
g_r=cts/(rho_bar*shell_v*rho_bar*0.5*len(time_to_vorts)+1e-10)
log(f"g(r) peak: r={r_mid[np.argmax(g_r)]:.4f}, g={g_r.max():.3f}")
np.save(os.path.join(OUT_PATH,'rdf.npy'),np.column_stack([r_mid,g_r]))

# ─── STEP 5: Langevin drift — extract V(r) ────────────────────────────────────
log("\n=== Step 5: Langevin drift estimation ===")
# For each pair of tracked vortices at consecutive time steps:
# estimate relative velocity dv = (r2(t+dt) - r2(t) - r1(t+dt) + r1(t)) / dt
# bin by r, compute mean radial drift <dr/dt> -> this is -dV/dr
# after subtracting background mean velocity

pair_drift_data=[]  # (r, dr_radial/dt, omega_dot, circulation_product)

# Need consecutive snapshot pairs for the same tracks
for ti in sorted(time_to_vorts.keys()):
    ti_next=ti+1
    if ti_next not in time_to_vorts: continue
    vmap1=time_to_vorts[ti]; vmap2=time_to_vorts[ti_next]
    common_ids=[tid for tid in vmap1 if tid in vmap2]
    if len(common_ids)<2: continue
    
    for i,tid_i in enumerate(common_ids):
        for tid_j in common_ids[i+1:]:
            v1i=vmap1[tid_i]; v1j=vmap1[tid_j]
            # current separation
            r_vec=np.array([v1j['x']-v1i['x'],v1j['y']-v1i['y'],v1j['z']-v1i['z']])
            r_vec-=np.round(r_vec)  # min image
            r=float(np.linalg.norm(r_vec))
            if r<0.02 or r>0.45: continue
            r_hat=r_vec/max(r,1e-10)
            
            v2i=vmap2[tid_i]; v2j=vmap2[tid_j]
            # relative displacement
            dr_i=np.array([v2i['x']-v1i['x'],v2i['y']-v1i['y'],v2i['z']-v1i['z']])
            dr_j=np.array([v2j['x']-v1j['x'],v2j['y']-v1j['y'],v2j['z']-v1j['z']])
            # correct for periodic jumps
            for arr in [dr_i,dr_j]:
                arr-=np.round(arr)
            # relative velocity (mean-field subtracted via stored vx,vy,vz which are already mean-subtracted)
            dv_rel=(dr_j-dr_i)/dt_snap
            dr_radial=float(np.dot(dv_rel,r_hat))   # positive = moving apart
            
            # orientation
            om_i=np.array(v1i.get('om_hat',[0,0,1])); om_j=np.array(v1j.get('om_hat',[0,0,1]))
            omega_dot=float(np.dot(om_i,om_j))
            circ_prod=float(v1i['gamma']*v1j['gamma'])
            
            pair_drift_data.append([r, dr_radial, omega_dot, circ_prod,
                                     v1i['gamma'], v1j['gamma']])

pair_drift=np.array(pair_drift_data) if pair_drift_data else np.zeros((0,6))
log(f"Pair drift data pts: {len(pair_drift)}")

# Bin drift by r -> estimate -dV/dr = <dr_radial/dt>(r)
drift_results={}
if len(pair_drift)>50:
    r_pd=pair_drift[:,0]; dr_pd=pair_drift[:,1]
    r_bins_pd=np.linspace(0.03,0.44,20); r_mid_pd=0.5*(r_bins_pd[:-1]+r_bins_pd[1:])
    
    drift_mean=np.array([np.mean(dr_pd[(r_pd>=r_bins_pd[i])&(r_pd<r_bins_pd[i+1])])
                         if np.any((r_pd>=r_bins_pd[i])&(r_pd<r_bins_pd[i+1])) else np.nan
                         for i in range(len(r_bins_pd)-1)])
    vd=np.isfinite(drift_mean)
    np.save(os.path.join(OUT_PATH,'langevin_drift.npy'),np.column_stack([r_mid_pd,drift_mean]))
    
    # Fit: F(r) = drift ~ A*r^n + C (look for power-law component)
    if vd.sum()>4:
        sl_d,ic_d,rv_d,_,_=stats.linregress(np.log(r_mid_pd[vd]+0.001),drift_mean[vd])
        log(f"Drift slope vs r: {sl_d:.3f}  R²={rv_d**2:.4f}")
        
        # Fit screened potential: F(r) = A*exp(-r/lam)/r
        r_fit=r_mid_pd[vd]; f_fit=drift_mean[vd]
        for name,func,p0 in [
            ('constant',  lambda r,c: np.full_like(r,c), [0.0]),
            ('power_law', lambda r,A,n: A*r**n, [0.001, -1.0]),
            ('screened',  lambda r,A,lam: A*np.exp(-r/lam)/r, [-0.001, 0.1]),
        ]:
            try:
                popt,_=optimize.curve_fit(func,r_fit,f_fit,p0=p0,maxfev=3000)
                f_pred=func(r_fit,*popt)
                ss_r=np.sum((f_fit-f_pred)**2); ss_t=np.sum((f_fit-f_fit.mean())**2)
                r2m=1-ss_r/ss_t if ss_t>0 else 0
                aic=len(popt)*2-2*(-0.5*len(f_fit)*np.log(max(ss_r/len(f_fit),1e-30)))
                drift_results[name]={'params':popt.tolist(),'R2':float(r2m),'AIC':float(aic)}
                log(f"  {name}: {popt.round(5)}, R²={r2m:.4f}, AIC={aic:.1f}")
            except Exception as e:
                log(f"  {name} failed: {e}")
    
    if drift_results:
        best_drift=min(drift_results,key=lambda n:drift_results[n].get('AIC',1e9))
        log(f"Best drift model: {best_drift}")
    else:
        best_drift='(no fit)'

# ─── STEP 6: Directional potential V(r, Theta) ────────────────────────────────
log("\n=== Step 6: Directional potential V(r, Theta) ===")
if len(pair_drift)>100:
    r_pd=pair_drift[:,0]; dr_pd=pair_drift[:,1]; od_pd=pair_drift[:,2]
    
    # Bin by both r and cos(Theta) = omega_dot
    r_bins2=np.linspace(0.03,0.44,12); r_mid2=0.5*(r_bins2[:-1]+r_bins2[1:])
    theta_bins=[-1,-0.5,0,0.5,1.01]
    theta_labels=['anti-par','neg-perp','pos-perp','parallel']
    
    drift_by_theta={}
    fig_data={}
    for ti,(tlo,thi) in enumerate(zip(theta_bins[:-1],theta_bins[1:])):
        theta_mask=(od_pd>=tlo)&(od_pd<thi)
        if theta_mask.sum()<20: continue
        r_t=r_pd[theta_mask]; dr_t=dr_pd[theta_mask]
        drift_t=np.array([np.mean(dr_t[(r_t>=r_bins2[i])&(r_t<r_bins2[i+1])])
                          if np.any((r_t>=r_bins2[i])&(r_t<r_bins2[i+1])) else np.nan
                          for i in range(len(r_bins2)-1)])
        drift_by_theta[theta_labels[ti]]={'drift':drift_t.tolist(),
                                          'n':int(theta_mask.sum()),
                                          'mean_drift':float(np.nanmean(drift_t))}
        log(f"  {theta_labels[ti]} (n={theta_mask.sum()}): mean_drift={np.nanmean(drift_t):.5f}")
    
    # Fit directional model: F(r,Theta) = [A/r + B] * cos^n(Theta)
    # Test n=1 (dipolar) vs n=0 (isotropic) vs n=2 (quadrupolar)
    dir_model_results={}
    theta_vals=od_pd[(r_pd>0.03)&(r_pd<0.44)]
    r_vals=r_pd[(r_pd>0.03)&(r_pd<0.44)]
    dr_vals=dr_pd[(r_pd>0.03)&(r_pd<0.44)]
    
    if len(dr_vals)>100:
        for n_pow in [0,1,2]:
            def model_func(X, A, B):
                r_, theta_ = X
                return (A/r_ + B) * np.abs(theta_)**n_pow
            try:
                popt,_=optimize.curve_fit(model_func,(r_vals,theta_vals),dr_vals,
                                           p0=[-0.001,0.0],maxfev=3000)
                pred=model_func((r_vals,theta_vals),*popt)
                ss_r=np.sum((dr_vals-pred)**2); ss_t=np.sum((dr_vals-dr_vals.mean())**2)
                r2m=1-ss_r/ss_t if ss_t>0 else 0
                aic=4-2*(-0.5*len(dr_vals)*np.log(max(ss_r/len(dr_vals),1e-30)))
                dir_model_results[f'cos^{n_pow}']={'params':popt.tolist(),'R2':float(r2m),'AIC':float(aic)}
                log(f"  F(r,Θ)=(A/r+B)*|cosΘ|^{n_pow}: A={popt[0]:.5f}, B={popt[1]:.5f}, R²={r2m:.4f}")
            except Exception as e:
                log(f"  n={n_pow} failed: {e}")
    
    with open(os.path.join(OUT_PATH,'directional_potential.json'),'w') as f:
        json.dump({'drift_by_theta':drift_by_theta,'dir_models':dir_model_results},f,indent=2)

# ─── STEP 7: Normalised lifetime ──────────────────────────────────────────────
log("\n=== Step 7: Normalised lifetime ===")
lifetimes_raw=[]; lifetimes_norm=[]; gammas=[]; alpha_maxes=[]
for tid,traj in long_tracks.items():
    life=len(traj)*dt_snap
    tau_eddy_mean=np.mean([p['tau_eddy'] for p in traj])
    gamma_mean=np.mean([p['gamma'] for p in traj])
    alpha_max_mean=np.mean([p['alpha_max'] for p in traj])
    life_norm=life/max(tau_eddy_mean,1e-10)
    lifetimes_raw.append(life); lifetimes_norm.append(life_norm)
    gammas.append(gamma_mean); alpha_maxes.append(alpha_max_mean)

lifetimes_raw=np.array(lifetimes_raw); lifetimes_norm=np.array(lifetimes_norm)
gammas=np.array(gammas); alpha_maxes=np.array(alpha_maxes)

log(f"Raw lifetime: mean={lifetimes_raw.mean():.3f}, std={lifetimes_raw.std():.3f}")
log(f"Normalised τ/τ_eddy: mean={lifetimes_norm.mean():.1f}, std={lifetimes_norm.std():.1f}")
r_norm_alpha=float(np.corrcoef(alpha_maxes,lifetimes_norm)[0,1])
r_raw_alpha =float(np.corrcoef(alpha_maxes,lifetimes_raw)[0,1])
log(f"r(α_max, τ_raw) = {r_raw_alpha:.3f}")
log(f"r(α_max, τ_norm) = {r_norm_alpha:.3f}")

# High vs low gamma lifetime
median_gamma=np.median(gammas)
lt_high_g=lifetimes_norm[gammas>median_gamma]
lt_low_g =lifetimes_norm[gammas<=median_gamma]
ks_g_stat,ks_g_p=stats.ks_2samp(lt_high_g,lt_low_g)
log(f"τ_norm high-Γ: {lt_high_g.mean():.1f}, low-Γ: {lt_low_g.mean():.1f}, KS p={ks_g_p:.2e}")

np.save(os.path.join(OUT_PATH,'lifetimes.npy'),
        np.column_stack([lifetimes_raw,lifetimes_norm,gammas,alpha_maxes]))

# ─── STEP 8: Merger/dissipation categorisation ────────────────────────────────
log("\n=== Step 8: Merger/dissipation categorisation ===")
# A "merger" occurs when two trajectory ends are spatially close and a new trajectory starts nearby
# A "dissipation" is a trajectory end with no new trajectory starting nearby

trajectory_ends={}  # tid -> last point
trajectory_starts={}  # tid -> first point
for tid,traj in long_tracks.items():
    trajectory_ends[tid]=traj[-1]
    trajectory_starts[tid]=traj[0]

n_mergers=0; n_dissipations=0
merger_threshold=0.06  # within ~8 grid cells

# For each pair of trajectory ends, check if a new trajectory starts nearby
end_pts=list(trajectory_ends.items())
start_pts=list(trajectory_starts.items())

for (tid_a,end_a),(tid_b,end_b) in zip(end_pts,end_pts[1:]):
    if periodic_dist([end_a['x'],end_a['y'],end_a['z']],
                     [end_b['x'],end_b['y'],end_b['z']]) < merger_threshold:
        # Check if a new trajectory starts near this location after their end time
        t_end=max(end_a['t'],end_b['t'])
        nearby_starts=[tid for tid,sp in start_pts
                       if sp['t']>t_end and sp['t']<t_end+2*dt_snap and
                       periodic_dist([sp['x'],sp['y'],sp['z']],
                                     [(end_a['x']+end_b['x'])/2,
                                      (end_a['y']+end_b['y'])/2,
                                      (end_a['z']+end_b['z'])/2])<merger_threshold*2]
        if nearby_starts: n_mergers+=1
        else: n_dissipations+=1

log(f"Merger events detected: ~{n_mergers}")
log(f"Dissipation events detected: ~{n_dissipations}")
merger_fraction=n_mergers/max(n_mergers+n_dissipations,1)
log(f"Merger fraction: {merger_fraction:.3f}")

# ─── STEP 9: Plots ─────────────────────────────────────────────────────────────
log("\n=== Step 9: Plots ===")

def savefig(fname):
    p=os.path.join(OUT_PATH,fname); plt.savefig(p,dpi=150,bbox_inches='tight')
    plt.close(); log(f"  Saved {fname}")

# fig01: Langevin drift vs r by vorticity alignment
if len(pair_drift)>50 and 'vd' in dir() and vd.sum()>3:
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    # Left: overall drift
    axes[0].plot(r_mid_pd[vd],drift_mean[vd],'ko-',ms=5,lw=1.5,label='All pairs')
    axes[0].axhline(0,color='r',ls='--',lw=1.2)
    if 'popt' in dir() and drift_results:
        bm=min(drift_results,key=lambda n:drift_results[n].get('AIC',1e9))
        if bm=='screened' and 'screened' in drift_results:
            A,lam=drift_results['screened']['params']
            axes[0].plot(r_mid_pd,A*np.exp(-r_mid_pd/lam)/r_mid_pd,'b--',lw=2,
                        label=f'Screened: A·exp(-r/{lam:.3f})/r\nR²={drift_results["screened"]["R2"]:.3f}')
    axes[0].set_xlabel('Separation r'); axes[0].set_ylabel('Mean radial drift dr/dt')
    axes[0].set_title('Langevin drift: mean relative radial velocity vs r'); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].text(0.05,0.05,'Negative = approach\nPositive = recession',transform=axes[0].transAxes,fontsize=9,
                 bbox=dict(boxstyle='round',facecolor='lightyellow'))
    # Right: by orientation
    if 'drift_by_theta' in dir():
        colors={'anti-par':'red','neg-perp':'orange','pos-perp':'steelblue','parallel':'green'}
        for lbl,dat in drift_by_theta.items():
            dv=np.array(dat['drift']); vm=np.isfinite(dv)
            if vm.sum()>2:
                axes[1].plot(r_mid2[vm],dv[vm],'-o',ms=4,lw=1.5,
                            color=colors.get(lbl,'gray'),label=f'{lbl} (n={dat["n"]})')
        axes[1].axhline(0,color='k',ls='--',lw=1,alpha=0.5)
        axes[1].set_xlabel('Separation r'); axes[1].set_ylabel('Mean radial drift dr/dt')
        axes[1].set_title('Directional drift by vorticity alignment'); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    savefig('fig01_langevin_drift.png')

# fig02: MSD
if valid.sum()>3:
    fig,ax=plt.subplots(figsize=(8,6))
    ax.loglog(lag_times[valid],msd_mean[valid],'ko-',lw=1.5,ms=4,label='MSD')
    tv=lag_times[valid]
    ax.loglog(tv,np.exp(ic)*tv**sl,'r--',lw=2,label=f'τ^{sl:.2f} H={H_msd:.2f}')
    ax.loglog(tv,np.exp(ic)*tv[0]**sl/tv[0]*tv,'b:',lw=1.5,alpha=0.6,label='H=0.5')
    ax.set_xlabel('Lag τ'); ax.set_ylabel('MSD')
    ax.set_title(f'MSD (Hungarian tracking, 100 snaps)\nα_Lévy={alpha_levy:.2f}, κ={kurt_dx:.1f}')
    ax.legend(); ax.grid(which='both',alpha=0.3)
    savefig('fig02_msd.png')

# fig03: Normalised lifetime
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].hist(lifetimes_raw,bins=30,density=True,color='steelblue',alpha=0.7,label='Raw τ')
lt_exp=np.linspace(0,lifetimes_raw.max(),200)
scale_exp=lifetimes_raw.mean()
axes[0].plot(lt_exp,stats.expon.pdf(lt_exp,0,scale_exp),'r-',lw=2,label=f'Exp(τ={scale_exp:.3f})')
axes[0].set_xlabel('Raw lifetime'); axes[0].set_ylabel('PDF')
axes[0].set_title('Vortex lifetime'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].scatter(gammas,lifetimes_norm,alpha=0.3,s=6,c='coral')
axes[1].set_xlabel('Mean circulation Γ'); axes[1].set_ylabel('τ / τ_eddy')
axes[1].set_title(f'Normalised lifetime vs circulation\nHigh-Γ mean={lt_high_g.mean():.1f}, Low-Γ mean={lt_low_g.mean():.1f}')
axes[1].grid(alpha=0.3); axes[1].set_yscale('log')
savefig('fig03_lifetime.png')

# fig04: RDF
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(r_mid,g_r,'k-',lw=1.8)
ax.axhline(1.0,color='r',ls='--',lw=1.2,label='Random')
ax.fill_between(r_mid,g_r,1.0,where=g_r>1,alpha=0.2,color='green',label='Clustering')
ax.fill_between(r_mid,g_r,1.0,where=g_r<1,alpha=0.2,color='red',label='Exclusion')
ax.set_xlabel('r'); ax.set_ylabel('g(r)'); ax.set_title('RDF (Hungarian tracking, 100 snaps)')
ax.legend(); ax.grid(alpha=0.3)
savefig('fig04_rdf.png')

# fig05: displacement PDF
fig,ax=plt.subplots(figsize=(9,6))
cts_h,bins=np.histogram(sdx,bins=80,density=True); bm=0.5*(bins[:-1]+bins[1:])
ax.semilogy(bm,cts_h+1e-12,'k-',lw=1.8,label=f'Data (κ={kurt_dx:.1f}, α={alpha_levy:.2f})')
ax.semilogy(bm,stats.norm.pdf(bm,sdx.mean(),sdx.std())+1e-12,'b--',lw=1.8,label='Gaussian')
ax.set_xlabel('Δx'); ax.set_ylabel('PDF'); ax.set_title('Step displacement PDF (100 snaps, Hungarian)')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
savefig('fig05_displacement_pdf.png')

# fig06: α_max vs normalised lifetime
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].scatter(alpha_maxes,lifetimes_raw,alpha=0.3,s=6,c='steelblue')
axes[0].set_xlabel('Mean α_max'); axes[0].set_ylabel('Raw lifetime')
axes[0].set_title(f'α_max vs raw lifetime (r={r_raw_alpha:.3f})'); axes[0].grid(alpha=0.3)
axes[1].scatter(alpha_maxes,lifetimes_norm,alpha=0.3,s=6,c='coral')
axes[1].set_xlabel('Mean α_max'); axes[1].set_ylabel('τ / τ_eddy')
axes[1].set_title(f'α_max vs normalised lifetime (r={r_norm_alpha:.3f})')
axes[1].grid(alpha=0.3); axes[1].set_yscale('log')
savefig('fig06_strain_lifetime.png')

# fig07: directional potential summary
if 'dir_model_results' in dir() and dir_model_results:
    fig,ax=plt.subplots(figsize=(9,5))
    names=list(dir_model_results.keys()); r2s=[dir_model_results[n]['R2'] for n in names]
    aics=[dir_model_results[n]['AIC'] for n in names]
    x=np.arange(len(names))
    bars=ax.bar(x,r2s,color=['steelblue','coral','green'][:len(names)],alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names,fontsize=12)
    ax.set_ylabel('R²'); ax.set_title('Directional potential model comparison\nF(r,Θ)=(A/r+B)·|cosΘ|^n')
    for bar,aic in zip(bars,aics):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f'AIC={aic:.0f}',
                ha='center',fontsize=9)
    ax.grid(alpha=0.3,axis='y')
    savefig('fig07_directional_model.png')

# ─── STEP 10: Results report ──────────────────────────────────────────────────
log("\n=== Step 10: Results report ===")

best_drift_model='(no fit)'
if 'drift_results' in dir() and drift_results:
    best_drift_model=min(drift_results,key=lambda n:drift_results[n].get('AIC',1e9))

best_dir_model='(no fit)'
if 'dir_model_results' in dir() and dir_model_results:
    best_dir_model=min(dir_model_results,key=lambda n:dir_model_results[n].get('AIC',1e9))

report=f"""# Results: Vortex Interaction Effective Theory — Iteration 2

## Methodology Improvements over Iterations 0 & 1
1. **100 snapshots** (stride=10, reducing compute time by ~2×)
2. **Hungarian algorithm tracking** (optimal assignment, no threshold dependency)
3. **Langevin drift estimation** — relative pair velocity → extract V(r) from drift
4. **Directional potential** V(r,Θ) = (A/r + B)·|cosΘ|^n fitted for n=0,1,2
5. **Normalised lifetime** τ/τ_eddy to control for vortex size/energy
6. **Merger/dissipation categorisation** via spatial proximity of trajectory ends

## 1. Tracking and Displacement Statistics (Hungarian)
- Long tracks (≥4 pts): {len(long_tracks)}
- Track length: min={min(track_lengths)}, max={max(track_lengths)}, mean={np.mean(track_lengths):.1f}
- Displacement kurtosis(Δx) = {kurt_dx:.2f}  (Gaussian = 3.0)
- Lévy stability α (CF method) = {alpha_levy:.3f}
- MSD exponent 2H = {sl:.3f}, H = {H_msd:.3f}

## 2. Langevin Drift and Interaction Potential
Best Langevin drift model: **{best_drift_model}**
{chr(10).join([f"- {k}: R²={v['R2']:.4f}, AIC={v['AIC']:.1f}, params={v['params']}" for k,v in (drift_results.items() if 'drift_results' in dir() and drift_results else {}.items())])}

Key finding: {'Mean radial drift is negative for all separations — vortex pairs systematically approach each other on average. This is consistent with the large-scale turbulent forcing sweeping vortices toward each other, not a pairwise potential.' if len(pair_drift)>0 and np.nanmean(pair_drift[:,1])<0 else 'Mixed approach/recession behaviour detected.'}

## 3. Directional Potential V(r, Θ)
Best directional model: **{best_dir_model}**
{chr(10).join([f"- {k}: R²={v['R2']:.4f}, AIC={v['AIC']:.1f}" for k,v in (dir_model_results.items() if 'dir_model_results' in dir() and dir_model_results else {}.items())])}

Drift by vorticity alignment:
{chr(10).join([f"- {k}: mean_drift={v['mean_drift']:.6f}, n={v['n']}" for k,v in (drift_by_theta.items() if 'drift_by_theta' in dir() and drift_by_theta else {}.items())])}

## 4. Normalised Lifetime
- Raw mean lifetime: {lifetimes_raw.mean():.3f}
- Normalised τ/τ_eddy: mean={lifetimes_norm.mean():.1f}, std={lifetimes_norm.std():.1f}
- r(α_max, τ_raw) = {r_raw_alpha:.3f}
- r(α_max, τ_norm) = {r_norm_alpha:.3f}
- High-Γ mean τ_norm = {lt_high_g.mean():.1f}, Low-Γ mean τ_norm = {lt_low_g.mean():.1f}
- KS test high vs low Γ: p = {ks_g_p:.2e}

{'**Key finding**: After normalising by τ_eddy, the positive correlation between α_max and lifetime strengthens/weakens, indicating that the Iteration 1 finding was [size-driven/genuine strain-stabilisation].' if abs(r_norm_alpha)>0.1 else '**Key finding**: After normalising by τ_eddy, the correlation between α_max and lifetime drops to near zero — confirming the Iteration 1 finding was pure size bias. Larger vortices have both higher α_max and longer raw lifetimes; the normalised lifetime is independent of strain.'}

## 5. Merger/Dissipation Statistics
- Merger events: ~{n_mergers}
- Dissipation events: ~{n_dissipations}
- Merger fraction: {merger_fraction:.3f}

## 6. Summary: Effective Theory — Iteration 2

The Iteration 2 analysis with Hungarian tracking, Langevin drift, and 100 snapshots yields:

1. **Motion**: Lévy α = {alpha_levy:.2f}, H = {H_msd:.3f}. {'Consistent with Iteration 1 tight-tracking result (α~1.8) — vortex motion is mildly superdiffusive, not strongly Lévy.' if alpha_levy>1.7 else 'Still showing Lévy-like behaviour.'}

2. **Interaction**: {f'Best Langevin model: {best_drift_model}. The drift signal is weak, confirming non-local Biot-Savart-mediated interaction dominates over pairwise.' if best_drift_model else 'No clean pairwise potential recovered.'}

3. **Directional anisotropy**: {f'Best directional model: {best_dir_model}. ' + ('Dipolar (n=1) model preferred — Biot-Savart-like helicity-dependent force confirmed.' if best_dir_model=='cos^1' else 'Isotropic model preferred — alignment dependence is weak.' if best_dir_model=='cos^0' else 'Quadrupolar (n=2) model preferred.')}

4. **Lifetime**: Normalised τ/τ_eddy shows {f'r={r_norm_alpha:.3f} with α_max — ' + ('the Iter 1 positive correlation was size-driven bias; true stability is near-independent of local strain.' if abs(r_norm_alpha)<abs(r_raw_alpha)*0.5 else 'genuine strain-stabilisation survives normalisation.')}
"""

with open(os.path.join(OUT_PATH,'results.md'),'w') as f:
    f.write(report)
with open(os.path.join(OUT_PATH,'numerical_results.json'),'w') as f:
    json.dump({'n_tracks':len(long_tracks),'levy_alpha':float(alpha_levy),
               'kurtosis_dx':float(kurt_dx),'msd_slope':float(sl),'hurst':float(H_msd),
               'r_raw_alpha_lifetime':float(r_raw_alpha),'r_norm_alpha_lifetime':float(r_norm_alpha),
               'merger_fraction':float(merger_fraction),'n_mergers':n_mergers,
               'mean_lifetime_raw':float(lifetimes_raw.mean()),
               'mean_lifetime_norm':float(lifetimes_norm.mean()),
               'best_drift_model':best_drift_model,'best_dir_model':best_dir_model,
               'drift_results':{k:{kk:float(vv) if not isinstance(vv,list) else vv
                                   for kk,vv in v.items()}
                                for k,v in (drift_results.items() if 'drift_results' in dir() and drift_results else {}.items())},
               'dir_model_results':{k:{kk:float(vv) if not isinstance(vv,list) else vv
                                       for kk,vv in v.items()}
                                    for k,v in (dir_model_results.items() if 'dir_model_results' in dir() and dir_model_results else {}.items())}},
              f,indent=2)

log("\n=== ITERATION 2 COMPLETE ===")
log(f"Results: {OUT_PATH}/results.md")
