"""
HEALPix-based analysis of DM anisotropy.
"""

import os
import gc
import math
import pickle
import numpy as np
from scipy.spatial import cKDTree
import healpy as hp
import h5py

from . import config
from .data_loader import get_galaxy_parameters, load_gas_data
from .thermodynamics import electron_number_density, two_phase_ne_correction
from .los import construct_diskless_LOS

def process_galaxy_healpix(halo_id, orig_idx, mw_catalog, hvc_catalog, h,
                           nside=64, snap=99, n_pts=500, apply_two_phase=True,
                           chunk_pix=4096):
    """
    Process a single galaxy to generate HEALPix DM map.
    
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
    nside : int
        HEALPix nside parameter
    snap : int
        Snapshot number
    n_pts : int
        Number of LOS points
    apply_two_phase : bool
        Whether to apply two-phase correction
    chunk_pix : int
        Chunk size for processing
        
    Returns
    -------
    dm_map : ndarray or None
        HEALPix DM map, or None if failed
    """
    mwpath = config.MWPATH
    cutout_path = f"{mwpath}/cutouts/snap_0{snap}/{halo_id}.hdf5"
    
    if not os.path.exists(cutout_path):
        return None
    
    params = get_galaxy_parameters(mw_catalog, hvc_catalog, halo_id, orig_idx, h)
    gas_data = load_gas_data(cutout_path)
    
    if gas_data is None:
        return None
    
    gas_coords = gas_data['coords']
    gas_density = gas_data['density']
    gas_xe = gas_data['xe']
    gas_u = gas_data['u']
    gas_sfr = gas_data['sfr']
    
    tree = cKDTree(gas_coords)
    
    npix = hp.nside2npix(nside)
    theta_hp, phi_hp = hp.pix2ang(nside, np.arange(npix))
    dm_map = np.full(npix, np.nan)
    
    for chunk_start in range(0, npix, chunk_pix):
        chunk_end = min(chunk_start + chunk_pix, npix)
        chunk_LOS, chunk_r = [], []
        
        for i in range(chunk_start, chunk_end):
            LOS, r = construct_diskless_LOS(
                theta_hp[i], phi_hp[i],
                params['disk_radius_cu'],
                params['disk_height_cu'],
                r_max=params['r_max_cu'],
                n_pts=n_pts,
                spacing='log'
            )
            chunk_LOS.append(LOS)
            chunk_r.append(r)
        
        flat_pts = np.vstack(chunk_LOS)
        _, cidx_flat = tree.query(flat_pts, k=1, workers=-1)
        
        cursor = 0
        for j in range(chunk_start, chunk_end):
            sl = slice(cursor, cursor + n_pts)
            cursor += n_pts
            
            cidx = cidx_flat[sl]
            r_v = chunk_r[j - chunk_start]
            
            dl = np.diff(np.concatenate([[0.0], r_v]))
            density_v = gas_density[cidx]
            xe_v = gas_xe[cidx]
            u_v = gas_u[cidx]
            sfr_v = gas_sfr[cidx]
            
            ne_v = electron_number_density(density_v, xe_v, h)
            if apply_two_phase:
                ne_v = two_phase_ne_correction(ne_v, u_v, xe_v, sfr_v)
            
            sfr_filt = sfr_v <= 0.0
            dm_arr = ne_v[sfr_filt] * dl[sfr_filt] / h * 1000.0
            dm_map[j] = np.sum(dm_arr)
        
        del flat_pts, chunk_LOS, chunk_r, cidx_flat
        gc.collect()
    
    del tree, gas_coords, gas_density, gas_xe, gas_u, gas_sfr
    gc.collect()
    
    return dm_map

def process_all_healpix(selected_ids, selected_orig, mw_catalog, hvc_catalog, h,
                       nside=64, cache_file=None, **kwargs):
    """
    Process all galaxies to generate HEALPix maps.
    
    Parameters
    ----------
    selected_ids : array_like
        Subhalo IDs
    selected_orig : array_like
        Original catalog indices
    mw_catalog : h5py.File
        MW catalog
    hvc_catalog : h5py.File
        HVC catalog
    h : float
        Hubble parameter
    nside : int
        HEALPix nside
    cache_file : str, optional
        Cache file path
    **kwargs : dict
        Additional arguments for process_galaxy_healpix
        
    Returns
    -------
    results : dict
        HEALPix processing results
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading HEALPix cache from '{cache_file}'...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Processing {len(selected_ids)} galaxies for HEALPix...")
    
    all_healpix_maps = []
    all_healpix_idx = []
    
    for mw_idx, halo_id in enumerate(selected_ids):
        print(f'\r  [{mw_idx+1}/{len(selected_ids)}] SubfindID {halo_id}',
              end='', flush=True)
        
        orig_idx = selected_orig[mw_idx]
        
        dm_map = process_galaxy_healpix(
            halo_id, orig_idx, mw_catalog, hvc_catalog, h,
            nside=nside, **kwargs
        )
        
        if dm_map is not None:
            all_healpix_maps.append(dm_map)
            all_healpix_idx.append(mw_idx)
    
    print(f'\n  Completed {len(all_healpix_maps)} maps.')
    
    results = {
        'all_healpix_maps': all_healpix_maps,
        'all_healpix_idx': np.array(all_healpix_idx),
        'nside': nside,
    }
    
    if cache_file:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved to '{cache_file}'.")
    
    return results

def compute_power_spectra(all_healpix_maps, lmax=None):
    """
    Compute angular power spectra for all maps.
    
    Parameters
    ----------
    all_healpix_maps : list
        List of HEALPix maps
    lmax : int, optional
        Maximum multipole
        
    Returns
    -------
    all_cl : list
        List of power spectra
    valid_cl : ndarray
        Array of valid power spectra
    ell : ndarray
        Multipole values
    """
    if lmax is None:
        nside = hp.npix2nside(len(all_healpix_maps[0]))
        lmax = 3 * nside - 1
    
    all_cl = []
    for dm_map in all_healpix_maps:
        m = dm_map.copy()
        nan_mask = np.isnan(m)
        if nan_mask.all():
            all_cl.append(None)
            continue
        m[nan_mask] = np.nanmean(m)
        all_cl.append(hp.anafast(m, lmax=lmax))
    
    valid_cl = np.array([c for c in all_cl if c is not None])
    ell = np.arange(valid_cl.shape[1])
    
    return all_cl, valid_cl, ell

def plot_healpix_map(dm_map, title='HEALPix DM Map', vmin=None, vmax=None,
                    unit=r'DM [pc cm$^{-3}$]', cmap='magma'):
    """
    Plot a single HEALPix map.
    
    Parameters
    ----------
    dm_map : ndarray
        HEALPix map
    title : str
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    unit : str
        Colorbar unit label
    cmap : str
        Colormap
        
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(8, 5))
    hp.mollview(dm_map, title=title, unit=unit, cmap=cmap,
               min=vmin, max=vmax, fig=fig)
    return fig

def plot_power_spectrum(ell, cl_median, cl_p16, cl_p84,
                       special_cl=None, title='Angular Power Spectrum'):
    """
    Plot angular power spectrum.
    
    Parameters
    ----------
    ell : array_like
        Multipole values
    cl_median : array_like
        Median power spectrum
    cl_p16, cl_p84 : array_like
        Percentile spectra
    special_cl : dict, optional
        Special galaxy spectra to overlay
    title : str
        Plot title
        
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(ell[1:], cl_median[1:], color='black', lw=2.5,
               label=f'Median')
    ax.fill_between(ell[1:], cl_p16[1:], cl_p84[1:],
                   color='steelblue', alpha=0.3,
                   label='16th–84th percentile')
    
    if special_cl:
        colors_hp = plt.cm.viridis(np.linspace(0, 1, len(special_cl)))
        for i, (label, cl) in enumerate(special_cl.items()):
            if cl is not None:
                ax.semilogy(ell[1:], cl[1:], color=colors_hp[i],
                           lw=1.0, alpha=0.7, ls='--', label=label)
    
    ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
    ax.set_ylabel(r'$C_\ell$ [pc$^2$ cm$^{-6}$]', fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig