"""
Statistical analysis utilities.
"""

import math
import numpy as np
from . import config

def aggregate(data, method=None, axis=None):
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
        return np.median(data, axis=axis)
    elif method == 'mean':
        return np.mean(data, axis=axis)
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


# ── New additions ────────────────────────────────────────────────────────────

def aggregate_with_axis(data, method=None, axis=None):
    """
    Aggregate an array using median or mean, with optional axis argument.

    Parameters
    ----------
    data : array-like
    method : {'median', 'mean'} or None
        If None, uses config.get_aggregation_method().
    axis : int or None

    Returns
    -------
    numpy.ndarray
    """
    from . import config  # local import to avoid circular deps
    import numpy as np

    if method is None:
        method = config.get_aggregation_method()

    if method == 'median':
        return np.median(data, axis=axis)
    elif method == 'mean':
        return np.mean(data, axis=axis)
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}. "
                         f"Choose 'median' or 'mean'.")


def compute_group_stats(group_data, method, latitudes, longitudes,
                        x_common, models):
    """
    Compute DM, temperature, and electron-density statistics for each
    SFR classification group.

    Parameters
    ----------
    group_data : dict
        Output of :func:`build_sfr_group_data`.
    method : {'median', 'mean'}
    latitudes : array-like
    longitudes : array-like
    x_common : array-like
        Radial grid for n_e profiles (r / R_vir).
    models : list
        Analytical CGM models from :func:`tng.models.get_analytical_models`.

    Returns
    -------
    dict
        Keys are group names; values are dicts of statistics.
    """
    import numpy as np
    from . import models as tng_models

    stats = {}
    for gname, gd in group_data.items():
        med_raw, p16_raw, p84_raw = compute_dm_stats(
            gd['dms_lat'], latitudes, sin_correct=False, method=method)
        med_sinb, p16_sinb, p84_sinb = compute_dm_stats(
            gd['dms_lat'], latitudes, sin_correct=True, method=method)
        med_lon, p16_lon, p84_lon = compute_dm_stats(
            gd['dms_lon'], longitudes, sin_correct=False, method=method)

        cfrac  = aggregate_with_axis(gd['frac'],     axis=0, method=method)
        ctotal = aggregate_with_axis(gd['total_DM'], method=method)
        p16_frac = np.percentile(gd['frac'], 16, axis=0)
        p84_frac = np.percentile(gd['frac'], 84, axis=0)

        ne_c   = aggregate_with_axis(gd['ne_interp'], axis=0, method=method)
        ne_p16 = np.nanpercentile(gd['ne_interp'], 16, axis=0)
        ne_p84 = np.nanpercentile(gd['ne_interp'], 84, axis=0)

        model_res = tng_models.rank_models(models, x_common, ne_c)

        stats[gname] = dict(
            n=gd['n'],
            med_raw=med_raw,   p16_raw=p16_raw,   p84_raw=p84_raw,
            med_sinb=med_sinb, p16_sinb=p16_sinb, p84_sinb=p84_sinb,
            med_lon=med_lon,   p16_lon=p16_lon,   p84_lon=p84_lon,
            cfrac=cfrac, p16_frac=p16_frac, p84_frac=p84_frac,
            ctotal=ctotal,
            ne_c=ne_c, ne_p16=ne_p16, ne_p84=ne_p84,
            model_res=model_res,
        )
    return stats


def build_sfr_group_data(sfr_class, all_galaxy_dms_arr, all_galaxy_dms_lon_arr,
                          lat_keys, lon_keys, all_temp_binned, all_total_DM,
                          all_fractions, ne_med_interp_all):
    """
    Slice the full results arrays into per-SFR-class sub-dicts.

    Parameters
    ----------
    sfr_class : dict
        Output of :func:`classify_sfr`.
    all_galaxy_dms_arr : ndarray, shape (N_gal, N_lat, N_lon)
    all_galaxy_dms_lon_arr : ndarray, shape (N_gal, N_lon_keys, ...)
    lat_keys, lon_keys : list of sorted keys
    all_temp_binned, all_total_DM, all_fractions, ne_med_interp_all : ndarray

    Returns
    -------
    dict
        Keys are group names ('Star-forming', 'Green valley', 'Quiescent').
    """
    import numpy as np

    group_masks = {
        'Star-forming': sfr_class['is_sf'],
        'Green valley': sfr_class['is_gv'],
        'Quiescent':    sfr_class['is_q'],
    }

    group_data = {}
    for gname, mask in group_masks.items():
        n = mask.sum()
        if n == 0:
            print(f"  ⚠  No {gname} galaxies — skipping")
            continue
        if n < 5:
            print(f"  ⚠  Only {n} {gname} galaxies — percentiles may be unreliable")

        gal_indices = np.where(mask)[0]

        dms_lat_dict = {
            lat: [all_galaxy_dms_arr[j, i, :] for j in gal_indices]
            for i, lat in enumerate(lat_keys)
        }
        dms_lon_dict = {
            lon: [all_galaxy_dms_lon_arr[j, i, :] for j in gal_indices]
            for i, lon in enumerate(lon_keys)
        }

        group_data[gname] = dict(
            n=n,
            dms_lat=dms_lat_dict,
            dms_lon=dms_lon_dict,
            temp=all_temp_binned[mask],
            total_DM=all_total_DM[mask],
            frac=all_fractions[mask],
            ne_interp=ne_med_interp_all[mask],
        )
    return group_data


def fit_lognormal(dm_samples_by_lat):
    """
    Fit a log-normal distribution to the pooled DM distribution at each
    latitude in *dm_samples_by_lat*.

    Parameters
    ----------
    dm_samples_by_lat : dict
        {latitude (int/float): list of 1-D arrays}  (same format as
        results['all_galaxy_dms'])

    Returns
    -------
    dict
        {latitude: dict with keys mu_ln, sigma_ln, median_dm,
         mean_dm_lognorm, mean_dm_empirical, std_dm_empirical,
         ks_stat, ks_p, sample, n}
    """
    import numpy as np
    from scipy import stats

    fit_results = {}
    for lat, arrays in dm_samples_by_lat.items():
        sample = np.concatenate(arrays)
        sample = sample[np.isfinite(sample) & (sample > 0)]
        if len(sample) < 10:
            continue

        shape, loc, scale = stats.lognorm.fit(sample, floc=0)
        sigma_ln = shape
        mu_ln    = np.log(scale)
        ks_stat, ks_p = stats.kstest(
            sample, 'lognorm', args=(shape, loc, scale)
        )

        fit_results[lat] = dict(
            sample=sample, n=len(sample),
            mu_ln=mu_ln, sigma_ln=sigma_ln,
            median_dm=np.exp(mu_ln),
            mean_dm_lognorm=np.exp(mu_ln + 0.5 * sigma_ln**2),
            mean_dm_empirical=float(np.mean(sample)),
            std_dm_empirical=float(np.std(sample)),
            ks_stat=ks_stat, ks_p=ks_p,
        )
    return fit_results