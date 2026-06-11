"""
Thermodynamic calculations for gas properties.
"""

import numpy as np
from .constants import MP, KB, GAMMA, XH, T_HOT_ISM, T_COLD_ISM, MSUN_KG, KPC_CM

def mean_molecular_weight(xe):
    """
    Calculate mean molecular weight.
    
    Parameters
    ----------
    xe : array_like
        Electron abundance
        
    Returns
    -------
    mu : ndarray
        Mean molecular weight in kg
    """
    return 4.0 / (1.0 + 3.0*XH + 4.0*XH*xe) * MP

def gas_temperature(xe, u):
    """
    Calculate gas temperature from internal energy.
    
    Parameters
    ----------
    xe : array_like
        Electron abundance
    u : array_like
        Internal energy
        
    Returns
    -------
    T : ndarray
        Temperature in K
    """
    mu = mean_molecular_weight(xe)
    return (GAMMA - 1.0) * u / (KB * 1e-6) * mu

def internal_energy_from_T(T, xe):
    """
    Calculate internal energy from temperature.
    
    Parameters
    ----------
    T : array_like
        Temperature in K
    xe : array_like
        Electron abundance
        
    Returns
    -------
    u : ndarray
        Internal energy
    """
    mu = mean_molecular_weight(xe)
    return T * KB * 1e-6 / ((GAMMA - 1.0) * mu)

def electron_number_density(density_code, xe, h):
    """
    Calculate electron number density.
    
    Parameters
    ----------
    density_code : array_like
        Density in code units
    xe : array_like
        Electron abundance
    h : float
        Hubble parameter
        
    Returns
    -------
    ne : ndarray
        Electron number density in cm^-3
    """
    density_conversion = MSUN_KG / KPC_CM**3 * 1e10 * h**2
    rho_kg_cm3 = density_code * density_conversion
    return xe * XH * rho_kg_cm3 / MP

def two_phase_ne_correction(ne, u, xe, sfr):
    """
    Apply two-phase ISM correction to electron density.
    
    Parameters
    ----------
    ne : array_like
        Electron number density
    u : array_like
        Internal energy
    xe : array_like
        Electron abundance
    sfr : array_like
        Star formation rate
        
    Returns
    -------
    ne_corr : ndarray
        Corrected electron number density
    """
    ne_corr = ne.copy()
    sf_mask = sfr > 0.0
    
    if not np.any(sf_mask):
        return ne_corr
    
    xe_sf = xe[sf_mask]
    u_sf = u[sf_mask]
    
    u_h = internal_energy_from_T(T_HOT_ISM, xe_sf)
    u_c = internal_energy_from_T(T_COLD_ISM, xe_sf)
    
    x = np.clip((u_h - u_sf) / (u_h - u_c), 0.0, 1.0)
    ne_corr[sf_mask] = ne[sf_mask] * (1.0 - x)
    
    return ne_corr