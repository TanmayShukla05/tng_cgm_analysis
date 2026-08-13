"""
Main galaxy processing pipeline.
"""

import os
import gc
import math
import pickle
import numpy as np
from scipy.spatial import cKDTree
import h5py

import scipy.stats as _scipy_stats

from . import config
from .data_loader import get_galaxy_parameters, load_gas_data
from .thermodynamics import gas_temperature, electron_number_density, two_phase_ne_correction
from .geometry import rotation_matrix_to_z
from .los import construct_diskless_LOS
from .statistics import aggregate

def process_single_galaxy(halo_id, orig_idx, mw_catalog, hvc_catalog, h,
                         latitudes, longitudes, snap=99,
                         n_pts=500, apply_two_phase=True, max_dist=None,
                         n_bins_ne=60, spacing='log'):
    """
    Process a single galaxy to compute DM along all lines of sight.
    
    Parameters
    ----------
    halo_id : int
        Subhalo ID
    orig_idx : int
        Index in original catalog
    mw_catalog : h5py.File
        MW catalog
    hvc_catalog : h5py.File
        HVC catalog
    h : float
        Hubble parameter
    latitudes : array_like
        Latitude grid
    longitudes : array_like
        Longitude grid
    snap : int
        Snapshot number
    n_pts : int
        Number of LOS points
    apply_two_phase : bool
        Whether to apply two-phase correction
    max_dist : float, optional
        Maximum distance for nearest neighbor
    n_bins_ne : int
        Number of bins for electron density profile
    spacing : str
        LOS spacing ('log' or 'linear')
        
    Returns
    -------
    result : dict or None
        Processing results, or None if failed
    """
    mwpath = config.MWPATH
    cutout_path = f"{mwpath}/cutouts/snap_0{snap}/{halo_id}.hdf5"
    
    if not os.path.exists(cutout_path):
        return None
    
    # Get galaxy parameters
    params = get_galaxy_parameters(mw_catalog, hvc_catalog, halo_id, orig_idx, h)
    
    # Load gas data
    gas_data = load_gas_data(cutout_path)
    if gas_data is None:
        return None
    
    gas_coords = gas_data['coords']
    gas_density = gas_data['density']
    gas_xe = gas_data['xe']
    gas_u = gas_data['u']
    gas_sfr = gas_data['sfr']
    gas_temp = gas_temperature(gas_xe, gas_u)
    
    # Build KD-tree
    gas_tree = cKDTree(gas_coords)
    
    # Construct all lines of sight
    all_LOS, all_r, key_order = [], [], []
    
    for lat in latitudes:
        theta = math.radians(90.0 - lat)
        for lon in longitudes:
            phi = math.radians(lon)
            LOS, r = construct_diskless_LOS(
                theta, phi,
                params['disk_radius_cu'],
                params['disk_height_cu'],
                r_max=params['r_max_cu'],
                n_pts=n_pts,
                spacing=spacing
            )
            all_LOS.append(LOS)
            all_r.append(r)
            key_order.append((lat, lon))
    
    # Query all points at once
    flat_pts = np.vstack(all_LOS)
    distances, gas_idx_flat = gas_tree.query(flat_pts, k=1, workers=-1)
    valid_flat = (distances <= max_dist if max_dist is not None 
                  else np.ones(len(flat_pts), dtype=bool))
    
    del gas_tree
    
    # Process each LOS
    LOSes_dict = {}
    cursor = 0
    
    for i, key in enumerate(key_order):
        sl = slice(cursor, cursor + n_pts)
        cursor += n_pts
        
        cidx = gas_idx_flat[sl]
        valid = valid_flat[sl]
        r_v = all_r[i][valid]
        cidx_v = cidx[valid]
        
        if len(cidx_v) == 0:
            LOSes_dict[key] = {
                'dm': 0.0,
                'temp': np.array([]),
                'dm_arr': np.array([])
            }
            continue
        
        dl = np.diff(np.concatenate([[0.0], r_v]))
        density_v = gas_density[cidx_v]
        xe_v = gas_xe[cidx_v]
        u_v = gas_u[cidx_v]
        sfr_v = gas_sfr[cidx_v]
        temp_v = gas_temp[cidx_v]
        
        ne_v = electron_number_density(density_v, xe_v, h)
        if apply_two_phase:
            ne_v = two_phase_ne_correction(ne_v, u_v, xe_v, sfr_v)
        
        sfr_filt = sfr_v <= 0.0
        dm_arr = ne_v[sfr_filt] * dl[sfr_filt] / h * 1000.0
        raw_dm = float(np.sum(dm_arr))
        
        LOSes_dict[key] = {
            'dm': raw_dm,
            'temp': temp_v[sfr_filt],
            'dm_arr': dm_arr
        }
    
    # Aggregate by latitude and longitude
    dms_per_lat = {}
    for lat in latitudes:
        dms = [LOSes_dict[(lat, lon)]['dm'] 
               for lon in longitudes if (lat, lon) in LOSes_dict]
        if dms:
            dms_per_lat[lat] = np.array(dms)
    
    dms_per_lon = {}
    for lon in longitudes:
        dms = [LOSes_dict[(lat, lon)]['dm'] 
               for lat in latitudes if (lat, lon) in LOSes_dict]
        if dms:
            dms_per_lon[lon] = np.array(dms)
    
    # Temperature binning
    from . import config as cfg
    T_BINS = cfg.T_BINS
    
    temp_all = np.concatenate([LOSes_dict[k]['temp'] 
                               for k in LOSes_dict if len(LOSes_dict[k]['dm_arr']) > 0])
    dm_arr_all = np.concatenate([LOSes_dict[k]['dm_arr'] 
                                 for k in LOSes_dict if len(LOSes_dict[k]['dm_arr']) > 0])
    
    temp_stat = None
    total_DM_gal = None
    
    if len(temp_all) > 0 and len(dm_arr_all) > 0:
        import scipy.stats
        total_DM_gal = np.sum(dm_arr_all)
        temp_stat, _, _ = scipy.stats.binned_statistic(
            temp_all, dm_arr_all, statistic='sum', bins=T_BINS
        )
    
    # Electron density profile
    gal_r_norm, gal_ne = [], []
    cursor2 = 0
    
    for i in range(len(all_LOS)):
        sl = slice(cursor2, cursor2 + n_pts)
        cursor2 += n_pts
        cidx = gas_idx_flat[sl]
        r_v = all_r[i]
        
        if len(cidx) == 0:
            continue
        
        density_v = gas_density[cidx]
        xe_v = gas_xe[cidx]
        u_v = gas_u[cidx]
        sfr_v = gas_sfr[cidx]
        
        ne_v = electron_number_density(density_v, xe_v, h)
        if apply_two_phase:
            ne_v = two_phase_ne_correction(ne_v, u_v, xe_v, sfr_v)
        
        filt = sfr_v <= 0.0
        gal_r_norm.extend(r_v[filt] / params['r_max_cu'])
        gal_ne.extend(ne_v[filt])
    
    gal_r_norm = np.array(gal_r_norm)
    gal_ne = np.array(gal_ne)
    
    interp_prof = None
    if len(gal_ne) > 0:
        x_common = np.logspace(-2, 0.5, n_bins_ne)
        r_min_fit = max(gal_r_norm.min(), 1e-4)
        r_max_fit = min(gal_r_norm.max(), 3.0)
        bin_edges = np.logspace(np.log10(r_min_fit), np.log10(r_max_fit), 
                               n_bins_ne + 1)
        bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        idx_digit = np.digitize(gal_r_norm, bin_edges) - 1
        ne_med_prof = np.full(n_bins_ne, np.nan)
        
        for b in range(n_bins_ne):
            mask = idx_digit == b
            if np.sum(mask) >= 3:
                ne_med_prof[b] = np.median(gal_ne[mask])
        
        valid_ne = np.isfinite(ne_med_prof) & (ne_med_prof > 0)
        if valid_ne.sum() >= 3:
            interp_prof = np.interp(x_common, bin_ctrs[valid_ne], 
                                   ne_med_prof[valid_ne],
                                   left=np.nan, right=np.nan)
    
    # Clean up
    del gas_coords, gas_density, gas_xe, gas_u, gas_sfr, gas_temp
    del LOSes_dict, all_LOS, all_r, flat_pts
    gc.collect()
    
    return {
        'dms_per_lat': dms_per_lat,
        'dms_per_lon': dms_per_lon,
        'temp_binned': temp_stat,
        'total_DM': total_DM_gal,
        'ne_profile': interp_prof,
        'R200c_kpc': params['R200c_kpc'],
    }

def process_all_galaxies(selected_ids, selected_orig, mw_catalog, hvc_catalog, h,
                        latitudes, longitudes, cache_file=None, **kwargs):
    """
    Process all galaxies in the sample.
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached results from '{cache_file}'...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (ModuleNotFoundError, AttributeError, ImportError) as e:
            print(f"⚠ Warning: Cache file incompatible ({e})")
            print(f"  Deleting cache and reprocessing...")
            os.remove(cache_file)
    
    print(f"Processing {len(selected_ids)} galaxies...")    
    all_galaxy_dms = {lat: [] for lat in latitudes}
    all_galaxy_dms_lon = {lon: [] for lon in longitudes}
    all_temp_binned = []
    all_total_DM = []
    ne_med_interp_all = []
    
    for mw_idx, halo_id in enumerate(selected_ids):
        print(f'\rProcessing galaxy {mw_idx+1}/{len(selected_ids)} '
              f'(SubfindID {halo_id})', end='', flush=True)
        
        orig_idx = selected_orig[mw_idx]
        
        result = process_single_galaxy(
            halo_id, orig_idx, mw_catalog, hvc_catalog, h,
            latitudes, longitudes, **kwargs
        )
        
        if result is None:
            continue
        
        # Aggregate latitude data
        for lat in latitudes:
            if lat in result['dms_per_lat']:
                all_galaxy_dms[lat].append(result['dms_per_lat'][lat])
        
        # Aggregate longitude data
        for lon in longitudes:
            if lon in result['dms_per_lon']:
                all_galaxy_dms_lon[lon].append(result['dms_per_lon'][lon])
        
        # Temperature data
        if result['temp_binned'] is not None:
            all_temp_binned.append(result['temp_binned'])
            all_total_DM.append(result['total_DM'])
        
        # Electron density profile
        if result['ne_profile'] is not None:
            ne_med_interp_all.append(result['ne_profile'])
    
    print(f'\n Completed processing {len(selected_ids)} galaxies.')
    
    results = {
        'all_galaxy_dms': all_galaxy_dms,
        'all_galaxy_dms_lon': all_galaxy_dms_lon,
        'all_temp_binned': np.array(all_temp_binned),
        'all_total_DM': np.array(all_total_DM),
        'ne_med_interp_all': np.array(ne_med_interp_all),
    }
    
    if cache_file:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved to '{cache_file}'.")
    
    return results


def process_single_galaxy_fitted(il, basepath, snap, sid, geom, r200_kpc, sub_pos, h,
                                  latitudes, longitudes, n_pts=500, apply_two_phase=True,
                                  max_dist=None, n_bins_ne=60, spacing='log'):
    """
    Same as process_single_galaxy, but built from raw snapshot gas
    (il.snapshot.loadSubhalo) rotated into THIS galaxy's fitted
    disk-aligned frame (geom['normal']), instead of reading pre-rotated
    cutout files.
    """
    gas = il.snapshot.loadSubhalo(basepath, snap, int(sid), 'gas',
                                   fields=['Coordinates', 'Density', 'ElectronAbundance',
                                           'InternalEnergy', 'StarFormationRate'])
    if gas['count'] == 0:
        return None

    # galactocentric offsets, code units (ckpc/h) - same convention as
    # disk_radius_cu / disk_height_cu / r_max_cu below, so NO h division
    dx_cu = gas['Coordinates'] - sub_pos
    R = rotation_matrix_to_z(geom['normal'])
    gas_coords = dx_cu @ R.T  # rotated into OUR fitted disk-aligned frame

    gas_density = gas['Density']
    gas_xe = gas['ElectronAbundance']
    gas_u = gas['InternalEnergy']
    gas_sfr = gas['StarFormationRate']
    gas_temp = gas_temperature(gas_xe, gas_u)

    disk_radius_cu = geom['radius_kpc'] * h
    disk_height_cu = geom['height_kpc'] * h
    r_max_cu = r200_kpc * h

    gas_tree = cKDTree(gas_coords)

    all_LOS, all_r, key_order = [], [], []
    for lat in latitudes:
        theta = math.radians(90.0 - lat)
        for lon in longitudes:
            phi = math.radians(lon)
            LOS, r = construct_diskless_LOS(
                theta, phi, disk_radius_cu, disk_height_cu,
                r_max=r_max_cu, n_pts=n_pts, spacing=spacing)
            all_LOS.append(LOS)
            all_r.append(r)
            key_order.append((lat, lon))

    flat_pts = np.vstack(all_LOS)
    distances, gas_idx_flat = gas_tree.query(flat_pts, k=1, workers=-1)
    valid_flat = (distances <= max_dist if max_dist is not None
                  else np.ones(len(flat_pts), dtype=bool))
    del gas_tree

    LOSes_dict = {}
    cursor = 0
    for i, key in enumerate(key_order):
        sl = slice(cursor, cursor + n_pts)
        cursor += n_pts
        cidx = gas_idx_flat[sl]
        valid = valid_flat[sl]
        r_v = all_r[i][valid]
        cidx_v = cidx[valid]

        if len(cidx_v) == 0:
            LOSes_dict[key] = {'dm': 0.0, 'temp': np.array([]), 'dm_arr': np.array([])}
            continue

        dl = np.diff(np.concatenate([[0.0], r_v]))
        density_v = gas_density[cidx_v]
        xe_v = gas_xe[cidx_v]
        u_v = gas_u[cidx_v]
        sfr_v = gas_sfr[cidx_v]
        temp_v = gas_temp[cidx_v]

        ne_v = electron_number_density(density_v, xe_v, h)
        if apply_two_phase:
            ne_v = two_phase_ne_correction(ne_v, u_v, xe_v, sfr_v)

        sfr_filt = sfr_v <= 0.0
        dm_arr = ne_v[sfr_filt] * dl[sfr_filt] / h * 1000.0
        raw_dm = float(np.sum(dm_arr))

        LOSes_dict[key] = {'dm': raw_dm, 'temp': temp_v[sfr_filt], 'dm_arr': dm_arr}

    dms_per_lat = {}
    for lat in latitudes:
        dms = [LOSes_dict[(lat, lon)]['dm'] for lon in longitudes if (lat, lon) in LOSes_dict]
        if dms:
            dms_per_lat[lat] = np.array(dms)

    dms_per_lon = {}
    for lon in longitudes:
        dms = [LOSes_dict[(lat, lon)]['dm'] for lat in latitudes if (lat, lon) in LOSes_dict]
        if dms:
            dms_per_lon[lon] = np.array(dms)

    T_BINS = config.T_BINS
    nonempty = [k for k in LOSes_dict if len(LOSes_dict[k]['dm_arr']) > 0]
    temp_all = np.concatenate([LOSes_dict[k]['temp'] for k in nonempty]) if nonempty else np.array([])
    dm_arr_all = np.concatenate([LOSes_dict[k]['dm_arr'] for k in nonempty]) if nonempty else np.array([])

    temp_stat = None
    total_DM_gal = None
    if len(temp_all) > 0 and len(dm_arr_all) > 0:
        total_DM_gal = np.sum(dm_arr_all)
        temp_stat, _, _ = _scipy_stats.binned_statistic(temp_all, dm_arr_all, statistic='sum', bins=T_BINS)

    gal_r_norm, gal_ne, gal_temp = [], [], []
    cursor2 = 0
    for i in range(len(all_LOS)):
        sl = slice(cursor2, cursor2 + n_pts)
        cursor2 += n_pts
        cidx = gas_idx_flat[sl]
        r_v = all_r[i]
        if len(cidx) == 0:
            continue
        density_v = gas_density[cidx]
        xe_v = gas_xe[cidx]
        u_v = gas_u[cidx]
        sfr_v = gas_sfr[cidx]
        temp_v_ne = gas_temp[cidx]

        ne_v = electron_number_density(density_v, xe_v, h)
        if apply_two_phase:
            ne_v = two_phase_ne_correction(ne_v, u_v, xe_v, sfr_v)

        filt = sfr_v <= 0.0
        gal_r_norm.extend(r_v[filt] / r_max_cu)
        gal_ne.extend(ne_v[filt])
        gal_temp.extend(temp_v_ne[filt])

    gal_r_norm = np.array(gal_r_norm)
    gal_ne = np.array(gal_ne)
    gal_temp = np.array(gal_temp)

    x_common = np.logspace(-2, 0.5, n_bins_ne)

    def _interp_ne_profile(r_norm, ne):
        """Median n_e(r/R200) profile on the shared x_common grid, log-spaced
        bins built from this subset's own r-range."""
        if len(ne) == 0:
            return None
        r_min_fit = max(r_norm.min(), 1e-4)
        r_max_fit = min(r_norm.max(), 3.0)
        bin_edges = np.logspace(np.log10(r_min_fit), np.log10(r_max_fit), n_bins_ne + 1)
        bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        idx_digit = np.digitize(r_norm, bin_edges) - 1
        ne_med_prof = np.full(n_bins_ne, np.nan)
        for b in range(n_bins_ne):
            mask = idx_digit == b
            if np.sum(mask) >= 3:
                ne_med_prof[b] = np.median(ne[mask])
        valid_ne = np.isfinite(ne_med_prof) & (ne_med_prof > 0)
        if valid_ne.sum() < 3:
            return None
        return np.interp(x_common, bin_ctrs[valid_ne], ne_med_prof[valid_ne],
                          left=np.nan, right=np.nan)

    interp_prof = _interp_ne_profile(gal_r_norm, gal_ne)

    # n_e profile split by temperature phase (config.T_BINS)
    n_phases = len(config.T_BINS) - 1
    interp_prof_by_tbin = np.full((n_phases, n_bins_ne), np.nan)
    if len(gal_temp) > 0:
        phase_idx = np.digitize(gal_temp, config.T_BINS) - 1
        for p in range(n_phases):
            pm = phase_idx == p
            prof_p = _interp_ne_profile(gal_r_norm[pm], gal_ne[pm])
            if prof_p is not None:
                interp_prof_by_tbin[p] = prof_p

    del gas_density, gas_xe, gas_u, gas_sfr, gas_temp, gas_coords
    del LOSes_dict, all_LOS, all_r, flat_pts
    gc.collect()

    return {
        'dms_per_lat': dms_per_lat,
        'dms_per_lon': dms_per_lon,
        'temp_binned': temp_stat,
        'total_DM': total_DM_gal,
        'ne_profile': interp_prof,
        'ne_profile_by_tbin': interp_prof_by_tbin,
        'R200c_kpc': r200_kpc,
    }


def process_all_galaxies_fitted(il, basepath, snap, selected_ids, selected_orig, rvir_vals,
                                 fitted_geom, h, latitudes, longitudes, **kwargs):
    """
    Same aggregation as process_all_galaxies, calling
    process_single_galaxy_fitted (fitted-normal version) per galaxy.
    """
    all_galaxy_dms = {lat: [] for lat in latitudes}
    all_galaxy_dms_lon = {lon: [] for lon in longitudes}
    all_temp_binned = []
    all_total_DM = []
    ne_med_interp_all = []
    ne_by_tbin_interp_all = []

    for mw_idx, sid in enumerate(selected_ids):
        print(f'\rProcessing galaxy {mw_idx+1}/{len(selected_ids)} (SubfindID {sid})',
              end='', flush=True)

        orig_idx = int(selected_orig[mw_idx])
        r200_kpc = rvir_vals[mw_idx]
        if r200_kpc <= 0:
            continue
        geom = fitted_geom[orig_idx]
        sub_pos = il.groupcat.loadSingle(basepath, snap, subhaloID=int(sid))['SubhaloPos']

        result = process_single_galaxy_fitted(il, basepath, snap, sid, geom, r200_kpc, sub_pos, h,
                                               latitudes, longitudes, **kwargs)
        if result is None:
            continue

        for lat in latitudes:
            if lat in result['dms_per_lat']:
                all_galaxy_dms[lat].append(result['dms_per_lat'][lat])
        for lon in longitudes:
            if lon in result['dms_per_lon']:
                all_galaxy_dms_lon[lon].append(result['dms_per_lon'][lon])

        if result['temp_binned'] is not None:
            all_temp_binned.append(result['temp_binned'])
            all_total_DM.append(result['total_DM'])
        if result['ne_profile'] is not None:
            ne_med_interp_all.append(result['ne_profile'])
        ne_by_tbin_interp_all.append(result['ne_profile_by_tbin'])

    print(f'\n Completed processing {len(selected_ids)} galaxies.')

    return {
        'all_galaxy_dms': all_galaxy_dms,
        'all_galaxy_dms_lon': all_galaxy_dms_lon,
        'all_temp_binned': np.array(all_temp_binned),
        'all_total_DM': np.array(all_total_DM),
        'ne_med_interp_all': np.array(ne_med_interp_all),
        'ne_by_tbin_interp_all': np.array(ne_by_tbin_interp_all),
    }


def compute_gas_kinematics(il, basepath, snap, selected_ids, selected_orig, rvir_vals,
                            fitted_geom, h, redshift, sub_vel_scale=True, n_rbins=16):
    """
    Bin radial velocity of CGM gas (outside each galaxy's fitted disk
    boundary, out to R_vir) in spherical shells of r/R_vir. Returns
    (r_centers, shell_vr, net_vr): shell_vr is per-galaxy median outflow
    (+) / inflow (-) velocity per shell, net_vr is the per-galaxy median
    within R_vir.
    """
    from .geometry import outside_disk_mask_fit

    a_scale = 1.0 / (1.0 + redshift)

    r_bins = np.linspace(0, 1, n_rbins)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    shell_vr = np.full((len(selected_ids), len(r_centers)), np.nan)
    net_vr = np.full(len(selected_ids), np.nan)

    for i, sid in enumerate(selected_ids):
        r200_kpc = rvir_vals[i]
        if r200_kpc <= 0:
            continue

        gas = il.snapshot.loadSubhalo(basepath, snap, int(sid), 'gas', fields=['Coordinates', 'Velocities'])
        if gas['count'] == 0:
            continue

        sub = il.groupcat.loadSingle(basepath, snap, subhaloID=int(sid))
        sub_pos = sub['SubhaloPos']
        sub_vel = sub['SubhaloVel']

        dx_kpc = (gas['Coordinates'] - sub_pos) / h
        r_kpc_all = np.linalg.norm(dx_kpc, axis=1)

        geom = fitted_geom[int(selected_orig[i])]
        outside_disk = outside_disk_mask_fit(dx_kpc, geom['normal'], geom['radius_kpc'], geom['height_kpc'])
        in_cgm = (r_kpc_all < r200_kpc) & outside_disk
        if in_cgm.sum() == 0:
            continue

        dx = dx_kpc[in_cgm]
        vel = gas['Velocities'][in_cgm]

        r_kpc = np.linalg.norm(dx, axis=1)
        valid = r_kpc > 0
        dx, vel, r_kpc = dx[valid], vel[valid], r_kpc[valid]
        if len(r_kpc) == 0:
            continue

        dv = vel * np.sqrt(a_scale) - sub_vel if sub_vel_scale else vel - sub_vel
        r_hat = dx / r_kpc[:, None]
        v_r = np.sum(dv * r_hat, axis=1)
        r_norm = r_kpc / r200_kpc

        for b in range(len(r_centers)):
            mask = (r_norm >= r_bins[b]) & (r_norm < r_bins[b + 1])
            if np.any(mask):
                shell_vr[i, b] = np.median(v_r[mask])

        within = r_norm <= 1.0
        if np.any(within):
            net_vr[i] = np.median(v_r[within])

    return r_centers, shell_vr, net_vr
