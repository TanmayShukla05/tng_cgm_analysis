"""
Data loading utilities for TNG simulations.
"""

import os
import h5py
import numpy as np
import illustris_python as il

def load_subhalo_ids(ids_path):
    """
    Load subhalo IDs from HDF5 file.
    
    Parameters
    ----------
    ids_path : str
        Path to IDs file
        
    Returns
    -------
    selected_ids : ndarray
        Array of subhalo IDs
    """
    with h5py.File(ids_path, 'r') as f:
        selected_ids = f['ids_MW'][:]
    return selected_ids

def open_catalogs(mwpath):
    """
    Open MW and HVC catalogs.
    
    Parameters
    ----------
    mwpath : str
        Path to MW catalogs
        
    Returns
    -------
    mw_catalog : h5py.File
        MW catalog file
    hvc_catalog : h5py.File
        HVC catalog file
    """
    mw_catalog = h5py.File(f"{mwpath}/mwm31s_hostcatalog.hdf5", "r")
    hvc_catalog = h5py.File(f"{mwpath}/cloud_catalog.hdf5", "r")
    return mw_catalog, hvc_catalog

def get_id_mapping(mw_catalog, selected_ids):
    """
    Create mapping from subhalo IDs to catalog indices.
    
    Parameters
    ----------
    mw_catalog : h5py.File
        MW catalog
    selected_ids : ndarray
        Selected subhalo IDs
        
    Returns
    -------
    selected_orig : ndarray
        Indices in original catalog
    """
    all_ids = mw_catalog['SubfindID'][:]
    id_to_idx = {int(sid): i for i, sid in enumerate(all_ids)}
    selected_orig = np.array([id_to_idx[int(sid)] for sid in selected_ids])
    return selected_orig

def load_header(basepath, snap):
    """
    Load simulation header.
    
    Parameters
    ----------
    basepath : str
        Path to simulation output
    snap : int
        Snapshot number
        
    Returns
    -------
    header : dict
        Simulation header
    """
    return il.groupcat.loadHeader(basepath, snap)

def find_catalog_key(catalog, hints):
    """
    Find catalog key matching hints.
    
    Parameters
    ----------
    catalog : h5py.File
        Catalog file
    hints : list
        List of hint strings
        
    Returns
    -------
    key : str or None
        Matching key name
    """
    for k in catalog.keys():
        for hint in hints:
            if hint.lower() in k.lower():
                return k
    return None

def get_galaxy_parameters(mw_catalog, hvc_catalog, halo_id, orig_idx, h):
    """
    Get galaxy disk and virial parameters.
    
    Parameters
    ----------
    mw_catalog : h5py.File
        MW catalog
    hvc_catalog : h5py.File
        HVC catalog
    halo_id : int
        Subhalo ID
    orig_idx : int
        Index in catalog
    h : float
        Hubble parameter
        
    Returns
    -------
    params : dict
        Dictionary with disk_height_cu, disk_radius_cu, r_max_cu, R200c_kpc
    """
    scale_height_kpc = float(np.array(
        mw_catalog['DiskScaleHeightThin_8kpc'])[orig_idx]) / 1000.0
    scale_length_kpc = float(np.array(
        mw_catalog['DiskScaleLength'])[orig_idx])
    
    disk_height_cu = (1.0 * scale_height_kpc) * h
    disk_radius_cu = (5.0 * scale_length_kpc) * h
    
    hvc_ids = np.array(hvc_catalog['SubhaloIdOfHostGalaxy'])
    hvc_match = np.where(hvc_ids == halo_id)[0]
    
    if len(hvc_match) > 0:
        R200c_kpc = float(hvc_catalog['R200cOfHostHalo'][int(hvc_match[0])])
    else:
        R200c_kpc = float(np.array(
            mw_catalog['HaloVirialRadius_R200c'])[orig_idx])
    
    r_max_cu = R200c_kpc * h
    
    return {
        'disk_height_cu': disk_height_cu,
        'disk_radius_cu': disk_radius_cu,
        'r_max_cu': r_max_cu,
        'R200c_kpc': R200c_kpc,
    }

def load_gas_data(cutout_path):
    """
    Load gas particle data from cutout.
    
    Parameters
    ----------
    cutout_path : str
        Path to cutout file
        
    Returns
    -------
    gas_data : dict or None
        Dictionary with gas properties, or None if no gas
    """
    if not os.path.exists(cutout_path):
        return None
    
    with h5py.File(cutout_path, "r") as f:
        if 'PartType0' not in f:
            return None
        
        gas_data = {
            'coords': np.array(f['PartType0']['RotatedCoordinates']),
            'density': np.array(f['PartType0']['Density']),
            'xe': np.array(f['PartType0']['ElectronAbundance']),
            'u': np.array(f['PartType0']['InternalEnergy']),
            'sfr': np.array(f['PartType0']['StarFormationRate']),
        }
    
    return gas_data

def find_catalog_key(catalog, hints):
    """
    Return the first key in *catalog* whose name contains any of the
    case-insensitive *hints*.

    Parameters
    ----------
    catalog : h5py.File or dict-like
    hints : list of str

    Returns
    -------
    str or None
    """
    for k in catalog.keys():
        for hint in hints:
            if hint.lower() in k.lower():
                return k
    return None


def stack_dm_arrays(all_galaxy_dms, all_galaxy_dms_lon):
    """
    Convert the dictionary-of-lists format returned by
    ``process_all_galaxies`` into contiguous NumPy arrays suitable for
    boolean-mask indexing.

    Parameters
    ----------
    all_galaxy_dms : dict  {latitude: list_of_arrays}
    all_galaxy_dms_lon : dict  {longitude: list_of_arrays}

    Returns
    -------
    all_dms_arr : ndarray, shape (N_gal, N_lat, N_lon)
    all_dms_lon_arr : ndarray, shape (N_gal, N_lon_keys, N_lat)
    lat_keys : list (sorted)
    lon_keys : list (sorted)
    """
    import numpy as np

    lat_keys = sorted(all_galaxy_dms.keys())
    lon_keys = sorted(all_galaxy_dms_lon.keys())

    all_dms_arr = np.stack(
        [all_galaxy_dms[lat] for lat in lat_keys], axis=1
    )
    all_dms_lon_arr = np.stack(
        [all_galaxy_dms_lon[lon] for lon in lon_keys], axis=1
    )
    return all_dms_arr, all_dms_lon_arr, lat_keys, lon_keys