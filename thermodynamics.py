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

def compute_entropy_pressure(il, basepath, snap, selected_ids, selected_orig, rvir_vals,
                              fitted_geom, h):
    """
    Per-galaxy CGM entropy and pressure (median over gas cells outside
    the fitted disk boundary, within R_vir, non-star-forming). Returns
    (entropy_list, pressure_list, all_T_K_pooled): entropy_list and
    pressure_list are one value per galaxy (np.nan if no valid gas),
    all_T_K_pooled is the pooled per-cell CGM gas temperature array
    (all galaxies together).
    """
    from .geometry import outside_disk_mask_fit

    XH_local = 0.76
    gamma = 5 / 3
    k_B_erg = 1.380649e-16
    k_B_keV = 8.617333e-8
    m_p = 1.67262192e-24
    UnitVelocity_cms = 1e5
    UnitMass_g = 1.989e43 / h
    UnitLength_cm = 3.0857e21 / h

    entropy_list = []
    pressure_list = []
    all_T_values_list = []

    for i, sid in enumerate(selected_ids):
        r200_kpc = rvir_vals[i]
        if r200_kpc <= 0:
            entropy_list.append(np.nan)
            pressure_list.append(np.nan)
            continue

        gas = il.snapshot.loadSubhalo(basepath, snap, int(sid), 'gas',
                                       fields=['InternalEnergy', 'ElectronAbundance', 'Density',
                                               'Coordinates', 'StarFormationRate'])
        if gas['count'] == 0:
            entropy_list.append(np.nan)
            pressure_list.append(np.nan)
            continue

        sub_pos = il.groupcat.loadSingle(basepath, snap, subhaloID=int(sid))['SubhaloPos']
        dx_kpc = (gas['Coordinates'] - sub_pos) / h
        r_kpc = np.linalg.norm(dx_kpc, axis=1)

        geom = fitted_geom[int(selected_orig[i])]
        outside_disk = outside_disk_mask_fit(dx_kpc, geom['normal'], geom['radius_kpc'], geom['height_kpc'])

        in_cgm = (r_kpc < r200_kpc) & (gas['StarFormationRate'] <= 0) & outside_disk
        if in_cgm.sum() == 0:
            entropy_list.append(np.nan)
            pressure_list.append(np.nan)
            continue

        ne_frac = gas['ElectronAbundance'][in_cgm]
        mu = 4.0 / (1 + 3 * XH_local + 4 * XH_local * ne_frac)
        T_K = (gamma - 1) * gas['InternalEnergy'][in_cgm] * UnitVelocity_cms**2 * mu * m_p / k_B_erg

        n_e_cm3 = ne_frac * XH_local * (gas['Density'][in_cgm] * UnitMass_g / UnitLength_cm**3) / m_p

        # total number density - includes H, He and their ions, not just electrons
        n_tot_cm3 = n_e_cm3 / (ne_frac * XH_local + 1e-12) * (XH_local + (1 - XH_local) / 4 + XH_local * ne_frac)

        entropy_list.append(np.median(k_B_keV * T_K / np.maximum(n_e_cm3, 1e-12) ** (2 / 3)))
        pressure_list.append(np.median(n_tot_cm3 * T_K))
        all_T_values_list.append(T_K)

    all_T_K_pooled = np.concatenate(all_T_values_list) if all_T_values_list else np.array([])

    return entropy_list, pressure_list, all_T_K_pooled
