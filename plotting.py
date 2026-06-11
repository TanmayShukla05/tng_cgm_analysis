"""
Plotting utilities for DM analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_dm_vs_latitude(latitudes, med, p16, p84, 
                       ylabel=r'DM [pc cm$^{-3}$]',
                       title='DM vs Galactic Latitude',
                       label='16th–84th percentile',
                       ax=None, color='steelblue',
                       special_profiles=None):
    """
    Plot DM vs galactic latitude.
    
    Parameters
    ----------
    latitudes : array_like
        Latitude values
    med : array_like
        Central values
    p16, p84 : array_like
        16th and 84th percentiles
    ylabel : str
        Y-axis label
    title : str
        Plot title
    label : str
        Legend label
    ax : matplotlib axis, optional
        Axis to plot on
    color : str
        Plot color
    special_profiles : dict, optional
        Dictionary of special galaxy profiles to overlay
        
    Returns
    -------
    ax : matplotlib axis
        The plot axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.errorbar(latitudes, med,
                yerr=[med - p16, p84 - med],
                fmt='o', color=color, markersize=5,
                capsize=4, elinewidth=1.2,
                label=label, zorder=5)
    
    if special_profiles:
        colors_sp = {'high': 'red', 'low': 'green'}
        markers_sp = {'high': 's', 'low': '^'}
        
        for key, profile_data in special_profiles.items():
            c = colors_sp.get(key.split('_')[0], 'gray')
            m = markers_sp.get(key.split('_')[0], 'o')
            ax.plot(latitudes, profile_data['profile'], 
                   f'{m}-', color=c, markersize=4, lw=1.5,
                   label=profile_data['label'], zorder=6)
    
    ax.set_xlabel('Galactic Latitude $b$ (°)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(-90, 91, 15))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_dm_vs_longitude(longitudes, med, p16, p84,
                        ylabel=r'DM [pc cm$^{-3}$]',
                        title='DM vs Galactic Longitude',
                        label='16th–84th percentile',
                        ax=None, color='steelblue',
                        special_profiles=None):
    """
    Plot DM vs galactic longitude.
    
    Parameters
    ----------
    longitudes : array_like
        Longitude values
    med : array_like
        Central values
    p16, p84 : array_like
        16th and 84th percentiles
    ylabel : str
        Y-axis label
    title : str
        Plot title
    label : str
        Legend label
    ax : matplotlib axis, optional
        Axis to plot on
    color : str
        Plot color
    special_profiles : dict, optional
        Dictionary of special galaxy profiles to overlay
        
    Returns
    -------
    ax : matplotlib axis
        The plot axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.errorbar(longitudes, med,
                yerr=[med - p16, p84 - med],
                fmt='o', color=color, markersize=3,
                capsize=3, elinewidth=1.0,
                label=label, zorder=5)
    
    if special_profiles:
        colors_sp = {'high': 'red', 'low': 'green'}
        markers_sp = {'high': 's', 'low': '^'}
        
        for key, profile_data in special_profiles.items():
            c = colors_sp.get(key.split('_')[0], 'gray')
            m = markers_sp.get(key.split('_')[0], 'o')
            ax.plot(longitudes, profile_data['profile'],
                   f'{m}-', color=c, markersize=3, lw=1.5,
                   label=profile_data['label'], zorder=6)
    
    ax.set_xlabel('Galactic Longitude $l$ (°)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_temperature_histogram(median_fractions, p16_fractions, p84_fractions,
                               T_BINS, n_galaxies, median_total_DM,
                               special_data=None, title_suffix=''):
    """
    Plot DM contribution vs temperature.
    
    Parameters
    ----------
    median_fractions : array_like
        Median DM fractions
    p16_fractions, p84_fractions : array_like
        Percentile fractions
    T_BINS : array_like
        Temperature bin edges
    n_galaxies : int
        Number of galaxies
    median_total_DM : float
        Median total DM
    special_data : dict, optional
        Special galaxy data to overlay
    title_suffix : str
        Additional title text
        
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bin_w = np.diff(T_BINS)
    bin_c = (T_BINS[:-1] + T_BINS[1:]) / 2
    
    ax.bar(T_BINS[:-1], median_fractions, width=bin_w, align='edge',
           edgecolor='steelblue', linewidth=0.6, alpha=0.55,
           color='steelblue', label=f'All galaxies (N={n_galaxies})', 
           zorder=3)
    
    ax.errorbar(bin_c, median_fractions,
                yerr=[median_fractions - p16_fractions, 
                      p84_fractions - median_fractions],
                fmt='none', ecolor='steelblue', capsize=2, alpha=0.6, 
                zorder=4)
    
    if special_data:
        colors = {'high': 'red', 'low': 'green'}
        for key, data in special_data.items():
            if data['temp_fraction'] is not None:
                color_key = key.split('_')[0]
                ax.step(np.concatenate([T_BINS[:-1], [T_BINS[-1]]]),
                       np.concatenate([data['temp_fraction'], 
                                      [data['temp_fraction'][-1]]]),
                       where='post', color=colors.get(color_key, 'gray'),
                       lw=2.0, label=data['label'], zorder=5)
    
    ax.set_xscale('log')
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$',
                       r'$10^6$', r'$10^7$', r'$10^8$'], fontsize=10)
    ax.set_xlim(1e3, 1e8)
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('DM contribution fraction', fontsize=12)
    ax.set_title(f'DM vs Temperature{title_suffix}\n'
                f'N = {n_galaxies} galaxies  |  '
                f'Median total DM = {median_total_DM:.1f} pc cm⁻³',
                fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_ne_profile(x_common, master_ne_median, master_ne_p16, master_ne_p84,
                   n_galaxies, model_results=None, special_data=None,
                   title_suffix=''):
    """
    Plot electron density profile.
    
    Parameters
    ----------
    x_common : array_like
        Normalized radius
    master_ne_median : array_like
        Median electron density
    master_ne_p16, master_ne_p84 : array_like
        Percentile densities
    n_galaxies : int
        Number of galaxies
    model_results : list, optional
        Analytical model results
    special_data : dict, optional
        Special galaxy data
    title_suffix : str
        Additional title text
        
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    valid_tng = np.isfinite(master_ne_median) & (master_ne_median > 0)
    
    ax.plot(x_common[valid_tng], master_ne_median[valid_tng],
            color='black', lw=4.0, zorder=10,
            label=f'TNG Median (N={n_galaxies})')
    
    valid_pct = (np.isfinite(master_ne_p16) & np.isfinite(master_ne_p84) &
                (master_ne_p16 > 0) & (master_ne_p84 > 0))
    ax.fill_between(x_common[valid_pct], master_ne_p16[valid_pct], 
                   master_ne_p84[valid_pct],
                   color='gray', alpha=0.3, zorder=9,
                   label='TNG 16th–84th Percentile')
    
    if special_data:
        colors = {'high': 'red', 'low': 'green'}
        markers = {'high': 's', 'low': '^'}
        for key, data in special_data.items():
            if data['ne_profile'] is not None:
                ne_sp = data['ne_profile']
                vm = np.isfinite(ne_sp) & (ne_sp > 0)
                if vm.any():
                    color_key = key.split('_')[0]
                    ax.plot(x_common[vm], ne_sp[vm],
                           f"{markers.get(color_key, 'o')}-",
                           color=colors.get(color_key, 'gray'),
                           markersize=3, lw=1.5, label=data['label'], 
                           zorder=8)
    
    if model_results:
        ax.plot([], [], ' ', label='─── Models (best → worst) ───')
        colors_model = plt.cm.tab10.colors
        for i, res in enumerate(model_results):
            ls = '-' if 'Fixed' not in res['name'] else '--'
            lw = 2.5 if i == 0 else 1.5
            ax.plot(x_common[res['valid']], res['ne_prof'][res['valid']],
                   label=f"{res['name']} (RMSE {res['rmse']:.2f})",
                   lw=lw, ls=ls, color=colors_model[i % len(colors_model)],
                   alpha=0.9)
    
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r\ /\ R_{\rm vir}$', fontsize=13)
    ax.set_ylabel(r'$\langle n_e \rangle\ \rm [cm^{-3}]$', fontsize=13)
    ax.set_title(f'Electron Density Profile{title_suffix}\n'
                r'TNG vs Analytical Models (Ranked by Log-RMSE)',
                fontsize=13)
    ax.set_xlim(0.0, 1.05)
    ax.axvline(1.0, color='gray', lw=1.2, ls='--', alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.text(1.02, ymin * 5, r'$R_{\rm vir}$', color='gray', fontsize=11)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    return fig

def plot_sfr_vs_mstar(log_mstar, log_sfr, is_sf, is_gv, is_q, is_zero,
                     n_sf, n_gv, n_q, n_zero, LOG_SFR_FLOOR):
    """
    Plot SFR vs stellar mass classification.
    
    Parameters
    ----------
    log_mstar : array_like
        Log stellar mass
    log_sfr : array_like
        Log SFR
    is_sf, is_gv, is_q, is_zero : array_like
        Classification masks
    n_sf, n_gv, n_q, n_zero : int
        Counts in each category
    LOG_SFR_FLOOR : float
        SFR floor value
        
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    C_SF = '#2166AC'
    C_GV = '#4DAC26'
    C_Q = '#D6604D'
    C_Q0 = '#8B0000'
    
    ALPHA_SCATTER = 0.80
    SCATTER_SIZE = 55
    
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    
    # Plot boundaries
    m_plot = np.linspace(log_mstar.min() - 0.15, log_mstar.max() + 0.15, 300)
    sfms_ridge = 0.8 * m_plot - 8.2
    sfms_y1 = 0.8 * m_plot - 8.7
    sfms_y2 = 0.8 * m_plot - 9.2
    
    ax.fill_between(m_plot, sfms_y2, sfms_y1, color=C_GV, alpha=0.10, zorder=0)
    
    # Plot points
    ax.scatter(log_mstar[is_q & ~is_zero], log_sfr[is_q & ~is_zero],
              c=C_Q, s=SCATTER_SIZE, alpha=ALPHA_SCATTER, 
              edgecolors='none', zorder=3)
    ax.scatter(log_mstar[is_zero], np.full(n_zero, LOG_SFR_FLOOR),
              c=C_Q0, s=SCATTER_SIZE, alpha=ALPHA_SCATTER,
              marker='v', edgecolors='none', zorder=3)
    ax.scatter(log_mstar[is_gv], log_sfr[is_gv],
              c=C_GV, s=SCATTER_SIZE, alpha=ALPHA_SCATTER,
              edgecolors='none', zorder=3)
    ax.scatter(log_mstar[is_sf], log_sfr[is_sf],
              c=C_SF, s=SCATTER_SIZE, alpha=ALPHA_SCATTER,
              edgecolors='none', zorder=4)
    
    # Plot lines
    ax.plot(m_plot, sfms_y1, color='k', lw=1.6, ls='--', zorder=5)
    ax.plot(m_plot, sfms_y2, color='k', lw=1.6, ls=':', zorder=5)
    ax.plot(m_plot, sfms_ridge, color='steelblue', lw=1.0, ls='-',
           alpha=0.45, zorder=2)
    
    # Legend
    patch_sf = mpatches.Patch(color=C_SF, label=f'Star-Forming (N={n_sf})')
    patch_gv = mpatches.Patch(color=C_GV, label=f'Green Valley (N={n_gv})')
    patch_q = mpatches.Patch(color=C_Q, label=f'Quiescent (N={n_q})')
    
    all_handles = [patch_sf, patch_gv, patch_q]
    all_labels = [h.get_label() for h in all_handles]
    
    ax.set_xlabel(r'$\log_{10}(M_\ast\,/\,M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR}\,/\,[M_\odot\,\mathrm{yr}^{-1}])$',
                 fontsize=14)
    ax.set_title(r'TNG50-1 MW Analogue Sample — SFR vs Stellar Mass' '\n'
                r'(Donnari et al. 2019 SFMS boundaries, $z=0$)',
                fontsize=13)
    ax.set_xlim(m_plot[0], m_plot[-1])
    ax.set_ylim(LOG_SFR_FLOOR - 0.3, 1.8)
    ax.axhline(LOG_SFR_FLOOR, color='gray', lw=0.8, ls='-', alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.45, zorder=0)
    ax.legend(handles=all_handles, labels=all_labels,
             loc='upper left', fontsize=9.5, framealpha=0.88)
    
    plt.tight_layout()
    return fig