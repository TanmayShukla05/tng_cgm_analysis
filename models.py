"""
Analytical model comparison utilities.
"""

import numpy as np
from frb.halos.models import ModifiedNFW, MB04, YF17

def get_analytical_models(log_mhalo=12.0):
    """
    Get dictionary of analytical CGM models.
    
    Parameters
    ----------
    log_mhalo : float
        Log10 of halo mass
        
    Returns
    -------
    models : dict
        Dictionary of model instances
    """
    return {
        'Modified NFW default': ModifiedNFW(log_Mhalo=log_mhalo),
        'Modified NFW y2 a2': ModifiedNFW(alpha=2, y0=2, log_Mhalo=log_mhalo),
        'Modified NFW y4 a2': ModifiedNFW(alpha=2, y0=4, log_Mhalo=log_mhalo),
        'MB04': MB04(log_Mhalo=log_mhalo),
        'YF17': YF17(log_Mhalo=log_mhalo),
    }

def evaluate_model(model, x_array):
    """
    Evaluate electron density for a model.
    
    Parameters
    ----------
    model : FRB halo model
        Model instance
    x_array : array_like
        Normalized radii (r/R_vir)
        
    Returns
    -------
    ne_prof : ndarray
        Electron density profile
    """
    r200_kpc = model.r200.value if hasattr(model, 'r200') else 200.0
    r_kpc = x_array * r200_kpc
    xyz = np.zeros((3, len(r_kpc)))
    xyz[0, :] = r_kpc
    ne_prof = model.ne(xyz)
    
    if hasattr(ne_prof, 'value'):
        ne_prof = ne_prof.value
    
    return ne_prof

def rank_models(models, x_common, master_ne_median):
    """
    Rank models by goodness of fit to TNG data.
    
    Parameters
    ----------
    models : dict
        Dictionary of model instances
    x_common : array_like
        Normalized radii
    master_ne_median : array_like
        TNG median electron density
        
    Returns
    -------
    model_results : list
        List of model results sorted by RMSE
    """
    valid_tng = np.isfinite(master_ne_median) & (master_ne_median > 0)
    model_results = []
    
    for name, model in models.items():
        ne_prof = evaluate_model(model, x_common)
        ne_prof = np.where(ne_prof <= 0, 1e-10, ne_prof)
        valid_m = np.isfinite(ne_prof) & (ne_prof > 0)
        overlap = valid_tng & valid_m & (ne_prof > 1e-9)
        
        if np.sum(overlap) > 0:
            rmse = np.sqrt(np.mean(
                (np.log10(master_ne_median[overlap]) - 
                 np.log10(ne_prof[overlap]))**2
            ))
        else:
            rmse = np.inf
        
        model_results.append({
            'name': name,
            'ne_prof': ne_prof,
            'rmse': rmse,
            'valid': valid_m,
        })
    
    model_results.sort(key=lambda x: x['rmse'])
    return model_results