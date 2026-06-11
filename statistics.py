"""
Statistical analysis utilities.
"""

import math
import numpy as np
from . import config

def aggregate(data, method=None):
    """
    Aggregate data using specified method.
    
    Parameters
    ----------
    data : array_like
        Data to aggregate
    method : str, optional
        'median' or 'mean'. If None, uses config default.
        
    Returns
    -------
    result : float
        Aggregated value
    """
    if method is None:
        method = config.get_aggregation_method()
    
    if method == 'median':
        return np.median(data)
    elif method == 'mean':
        return np.mean(data)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def percentile(data, p):
    """
    Calculate percentile of data.
    
    Parameters
    ----------
    data : array_like
        Input data
    p : float or array_like
        Percentile(s) to compute
        
    Returns
    -------
    result : float or ndarray
        Percentile value(s)
    """
    return np.percentile(data, p)

def compute_dm_stats(dm_dict, latitudes, sin_correct=False, method=None):
    """
    Compute DM statistics over latitudes.
    
    Parameters
    ----------
    dm_dict : dict
        Dictionary mapping latitudes to DM arrays
    latitudes : array_like
        Latitude values
    sin_correct : bool
        Whether to apply sin(b) correction
    method : str, optional
        Aggregation method ('median' or 'mean')
        
    Returns
    -------
    central : ndarray
        Central values (median or mean)
    p16 : ndarray
        16th percentile
    p84 : ndarray
        84th percentile
    """
    if method is None:
        method = config.get_aggregation_method()
    
    central, p16, p84 = [], [], []
    sin_b = np.array([abs(math.sin(math.radians(lat))) for lat in latitudes])
    
    for i, lat in enumerate(latitudes):
        arrays = dm_dict[lat]
        if not arrays:
            central.append(np.nan)
            p16.append(np.nan)
            p84.append(np.nan)
            continue
        
        scale = sin_b[i] if sin_correct else 1.0
        
        gal_central = [aggregate(a * scale, method) for a in arrays]
        gal_p16 = [percentile(a * scale, 16) for a in arrays]
        gal_p84 = [percentile(a * scale, 84) for a in arrays]
        
        central.append(aggregate(gal_central, method))
        p16.append(aggregate(gal_p16, method))
        p84.append(aggregate(gal_p84, method))
    
    return np.array(central), np.array(p16), np.array(p84)

def galaxy_dm_profile(dms_per_lat, latitudes, sin_correct=False, method=None):
    """
    Compute DM profile for a single galaxy.
    
    Parameters
    ----------
    dms_per_lat : dict
        DM values per latitude
    latitudes : array_like
        Latitude values
    sin_correct : bool
        Whether to apply sin(b) correction
    method : str, optional
        Aggregation method
        
    Returns
    -------
    profile : ndarray
        DM profile
    """
    if method is None:
        method = config.get_aggregation_method()
    
    sin_b = np.array([abs(math.sin(math.radians(lat))) for lat in latitudes])
    out = []
    
    for i, lat in enumerate(latitudes):
        if lat in dms_per_lat and len(dms_per_lat[lat]) > 0:
            scale = sin_b[i] if sin_correct else 1.0
            out.append(aggregate(dms_per_lat[lat] * scale, method))
        else:
            out.append(np.nan)
    
    return np.array(out)

def galaxy_dm_profile_lon(dms_per_lon, longitudes, method=None):
    """
    Compute DM profile vs longitude for a single galaxy.
    
    Parameters
    ----------
    dms_per_lon : dict
        DM values per longitude
    longitudes : array_like
        Longitude values
    method : str, optional
        Aggregation method
        
    Returns
    -------
    profile : ndarray
        DM profile
    """
    if method is None:
        method = config.get_aggregation_method()
    
    out = []
    for lon in longitudes:
        if lon in dms_per_lon and len(dms_per_lon[lon]) > 0:
            out.append(aggregate(dms_per_lon[lon], method))
        else:
            out.append(np.nan)
    
    return np.array(out)

def classify_sfr(m_star_raw, sfr_raw, h):
    """
    Classify galaxies by star formation activity.
    
    Parameters
    ----------
    m_star_raw : array_like
        Stellar mass in code units
    sfr_raw : array_like
        Star formation rate
    h : float
        Hubble parameter
        
    Returns
    -------
    classification : dict
        Dictionary with is_sf, is_gv, is_q masks and statistics
    """
    from .constants import LOG_SFR_FLOOR
    
    m_star_msun = m_star_raw * 1e10 / h
    log_mstar = np.log10(np.clip(m_star_msun, 1e-6, None))
    log_sfr = np.where(sfr_raw > 0.0, np.log10(sfr_raw), LOG_SFR_FLOOR)
    
    # Donnari et al. 2019 boundaries
    y1 = 0.8 * log_mstar - 8.7
    y2 = 0.8 * log_mstar - 9.2
    
    is_zero = sfr_raw == 0.0
    is_sf = (~is_zero) & (log_sfr > y1)
    is_gv = (~is_zero) & (log_sfr <= y1) & (log_sfr >= y2)
    is_q = is_zero | (log_sfr < y2)
    
    return {
        'is_sf': is_sf,
        'is_gv': is_gv,
        'is_q': is_q,
        'is_zero': is_zero,
        'log_mstar': log_mstar,
        'log_sfr': log_sfr,
        'n_sf': int(np.sum(is_sf)),
        'n_gv': int(np.sum(is_gv)),
        'n_q': int(np.sum(is_q)),
        'n_zero': int(np.sum(is_zero)),
    }