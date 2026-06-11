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

from . import config
from .data_loader import get_galaxy_parameters, load_gas_data
from .thermodynamics import gas_temperature, electron_number_density, two_phase_ne_correction
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
    
    Parameters
    ----------
    selected_ids : array_like
        Subhalo IDs to process
    selected_orig : array_like
        Original catalog indices
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
    cache_file : str, optional
        Cache file path
    **kwargs : dict
        Additional arguments for process_single_galaxy
        
    Returns
    -------
    results : dict
        Processing results for all galaxies
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached results from '{cache_file}'...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
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