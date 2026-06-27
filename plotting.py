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

def plot_sfr_class_comparison(stats, method_label, latitudes, longitudes,
                               T_BINS, x_common, models,
                               colors=None, markers=None,
                               master_ne=None, master_ne_p16=None,
                               master_ne_p84=None, n_master=None,
                               save_prefix='sfr_class'):
    """
    Generate the five standard SFR-class comparison plots for a given
    aggregation method.

    Plots produced
    --------------
    1. DM vs latitude (raw) — both groups on same axes
    2. DM × |sin b| vs latitude — both groups on same axes
    3. DM vs longitude
    4. Temperature histogram
    5. Electron-density profile

    Parameters
    ----------
    stats : dict
        Output of :func:`tng.statistics.compute_group_stats`.
    method_label : str
        Label used in titles and filenames, e.g. 'median' or 'mean'.
    latitudes, longitudes : array-like
    T_BINS : array-like
    x_common : array-like  (r / R_vir grid)
    models : list  (analytical models from tng.models)
    colors, markers : dict  {group_name: colour/marker}  — optional
    master_ne, master_ne_p16, master_ne_p84 : array-like or None
        Full-sample n_e statistics to overlay on panel 5.
    n_master : int or None
    save_prefix : str

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt

    _default_colors  = {'Star-forming': '#1976D2',
                        'Green valley': '#388E3C',
                        'Quiescent':    '#D32F2F'}
    _default_markers = {'Star-forming': 'o',
                        'Green valley': 's',
                        'Quiescent':    '^'}
    colors  = colors  or _default_colors
    markers = markers or _default_markers
    gnames  = list(stats.keys())
    figs    = []

    # ── Plots 1 & 2: DM and DM × |sin b| vs latitude ────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    figs.append(fig)

    for ax, key, ylabel, title in [
        (ax1, 'raw',  r'DM [pc cm$^{-3}$]',
               f'DM vs Latitude ({method_label})'),
        (ax2, 'sinb', r'DM $\times$ |sin $b$| [pc cm$^{-3}$]',
               f'DM × |sin b| vs Latitude ({method_label})'),
    ]:
        for gn in gnames:
            s  = stats[gn]
            c  = colors[gn]
            mk = markers[gn]
            med  = s[f'med_{key}']
            p16  = s[f'p16_{key}']
            p84  = s[f'p84_{key}']
            ax.errorbar(latitudes, med,
                        yerr=[med - p16, p84 - med],
                        fmt=f'{mk}-', color=c, markersize=4,
                        capsize=3, elinewidth=1.0,
                        label=f'{gn} (N={s["n"]})', zorder=5)
        ax.set_xlabel('Galactic Latitude $b$ (°)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(-90, 91, 15))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'SFR-classified CGM DM ({method_label.capitalize()})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_DM_vs_latitude_{method_label}.pdf',
                dpi=150, bbox_inches='tight')

    # ── Plot 3: DM vs longitude ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    figs.append(fig)
    for gn in gnames:
        s  = stats[gn]
        c  = colors[gn]
        mk = markers[gn]
        ax.errorbar(longitudes, s['med_lon'],
                    yerr=[s['med_lon'] - s['p16_lon'],
                          s['p84_lon'] - s['med_lon']],
                    fmt=f'{mk}-', color=c, markersize=3,
                    capsize=3, elinewidth=1.0,
                    label=f'{gn} (N={s["n"]})', zorder=5)
    ax.set_xlabel('Galactic Longitude $l$ (°)', fontsize=12)
    ax.set_ylabel(r'DM [pc cm$^{-3}$]', fontsize=12)
    ax.set_title(f'DM vs Longitude ({method_label})', fontsize=12)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_DM_vs_longitude_{method_label}.pdf',
                dpi=150, bbox_inches='tight')

    # ── Plot 4: Temperature histogram ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    figs.append(fig)
    bin_c = (T_BINS[:-1] + T_BINS[1:]) / 2

    for gn in gnames:
        s  = stats[gn]
        c  = colors[gn]
        edges = np.concatenate([T_BINS[:-1], [T_BINS[-1]]])
        vals  = np.concatenate([s['cfrac'], [s['cfrac'][-1]]])
        ax.step(edges, vals, where='post', color=c, lw=2.2,
                label=f'{gn} (N={s["n"]})', zorder=5)
        ax.errorbar(bin_c, s['cfrac'],
                    yerr=[s['cfrac'] - s['p16_frac'],
                          s['p84_frac'] - s['cfrac']],
                    fmt='none', ecolor=c, capsize=2, alpha=0.6, zorder=4)

    ax.set_xscale('log')
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$',
                         r'$10^6$', r'$10^7$', r'$10^8$'])
    ax.set_xlim(1e3, 1e8)
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('DM contribution fraction', fontsize=12)
    ax.set_title(f'DM vs Temperature by SFR Class ({method_label})',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_DM_temperature_{method_label}.pdf',
                dpi=150, bbox_inches='tight')

    # ── Plot 5: Electron density profile ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    figs.append(fig)

    if master_ne is not None:
        valid = np.isfinite(master_ne) & (master_ne > 0)
        n_lbl = f' (N={n_master})' if n_master else ''
        ax.plot(x_common[valid], master_ne[valid],
                color='black', lw=3.5, ls=':', zorder=11,
                label=f'TNG All {method_label.capitalize()}{n_lbl}')
        if master_ne_p16 is not None and master_ne_p84 is not None:
            vp = (np.isfinite(master_ne_p16) & np.isfinite(master_ne_p84)
                  & (master_ne_p16 > 0) & (master_ne_p84 > 0))
            ax.fill_between(x_common[vp],
                            master_ne_p16[vp], master_ne_p84[vp],
                            color='gray', alpha=0.15, zorder=10)

    for gn in gnames:
        s = stats[gn]
        c = colors[gn]
        valid = np.isfinite(s['ne_c']) & (s['ne_c'] > 0)
        ax.plot(x_common[valid], s['ne_c'][valid],
                color=c, lw=2.5, zorder=9,
                label=f'{gn} TNG (N={s["n"]})')
        vp = (np.isfinite(s['ne_p16']) & np.isfinite(s['ne_p84'])
              & (s['ne_p16'] > 0) & (s['ne_p84'] > 0))
        ax.fill_between(x_common[vp], s['ne_p16'][vp], s['ne_p84'][vp],
                        color=c, alpha=0.15, zorder=8)
        if s['model_res']:
            best = s['model_res'][0]
            bv = best['valid']
            ax.plot(x_common[bv], best['ne_prof'][bv],
                    color=c, lw=1.5, ls='--', alpha=0.85, zorder=7,
                    label=(f'{gn} best: {best["name"]} '
                           f'(RMSE {best["rmse"]:.2f})'))

    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r\ /\ R_{\rm vir}$', fontsize=13)
    ax.set_ylabel(r'$\langle n_e \rangle\ \rm [cm^{-3}]$', fontsize=13)
    ax.set_title(f'Electron Density by SFR Class '
                 f'({method_label.capitalize()})', fontsize=13)
    ax.set_xlim(0.0, 1.05)
    ax.axvline(1.0, color='gray', lw=1.2, ls='--', alpha=0.7)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_ne_profile_{method_label}.pdf',
                dpi=150, bbox_inches='tight')

    return figs


def plot_extreme_galaxy_comparison(latitudes, longitudes,
                                    med_raw, p16_raw, p84_raw,
                                    med_sinb, p16_sinb, p84_sinb,
                                    med_lon, p16_lon, p84_lon,
                                    sp_dm_raw, sp_dm_sinb, sp_dm_lon,
                                    special_data,
                                    high_key, low_key,
                                    property_name, method_label,
                                    T_BINS, x_common,
                                    central_fractions, p16_fractions, p84_fractions,
                                    master_ne_central, master_ne_p16, master_ne_p84,
                                    model_results, ne_med_interp_all,
                                    agg_color='steelblue',
                                    save_prefix='MWM31'):
    """
    Generate the five standard extreme-galaxy comparison plots for one
    galaxy property (SFR / M200 / Rvir) and one aggregation method.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt

    hk, lk = high_key, low_key
    figs = []
    method_cap = method_label.capitalize()

    # ── Plots 1 & 2: DM vs latitude (raw & sin b) ───────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    figs.append(fig)

    for ax, med, p16, p84, sp_dict, ylabel, title in [
        (ax1, med_raw,  p16_raw,  p84_raw,  sp_dm_raw,
         r'DM [pc cm$^{-3}$]',
         f'DM vs Latitude ({method_label}, {property_name} extremes)'),
        (ax2, med_sinb, p16_sinb, p84_sinb, sp_dm_sinb,
         r'DM $\times$ |sin $b$| [pc cm$^{-3}$]',
         f'DM × |sin b| vs Latitude ({method_label}, {property_name} extremes)'),
    ]:
        ax.errorbar(latitudes, med, yerr=[med - p16, p84 - med],
                    fmt='o', color=agg_color, markersize=5,
                    capsize=4, elinewidth=1.2,
                    label=f'{method_cap} (132 galaxies)', zorder=5)
        ax.plot(latitudes, sp_dict[hk], 's-', color='red',
                markersize=4, lw=1.5,
                label=special_data[hk]['label'], zorder=6)
        ax.plot(latitudes, sp_dict[lk], '^-', color='green',
                markersize=4, lw=1.5,
                label=special_data[lk]['label'], zorder=6)
        ax.set_xlabel('Galactic Latitude $b$ (°)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(np.arange(-90, 91, 15))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'MW Analogs CGM DM '
                 f'({method_label}, {property_name} comparison)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(
        f'{save_prefix}_DM_vs_latitude_{property_name}_comparison'
        f'_{method_label}.pdf',
        bbox_inches='tight', dpi=150)

    # ── Plot 3: DM vs longitude ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    figs.append(fig)
    ax.errorbar(longitudes, med_lon,
                yerr=[med_lon - p16_lon, p84_lon - med_lon],
                fmt='o', color=agg_color, markersize=3,
                capsize=3, elinewidth=1.0,
                label=f'{method_cap} (132 galaxies)', zorder=5)
    ax.plot(longitudes, sp_dm_lon[hk], 's-', color='red',
            markersize=3, lw=1.5, label=special_data[hk]['label'], zorder=6)
    ax.plot(longitudes, sp_dm_lon[lk], '^-', color='green',
            markersize=3, lw=1.5, label=special_data[lk]['label'], zorder=6)
    ax.set_xlabel('Galactic Longitude $l$ (°)', fontsize=12)
    ax.set_ylabel(r'DM [pc cm$^{-3}$]', fontsize=12)
    ax.set_title(f'DM vs Longitude ({method_label}, {property_name} extremes)',
                 fontsize=11)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f'{save_prefix}_DM_vs_longitude_{property_name}_comparison'
        f'_{method_label}.pdf',
        bbox_inches='tight', dpi=150)

    # ── Plot 4: Temperature histogram ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    figs.append(fig)
    bin_w = np.diff(T_BINS)
    bin_c = (T_BINS[:-1] + T_BINS[1:]) / 2

    ax.bar(T_BINS[:-1], central_fractions, width=bin_w, align='edge',
           edgecolor=agg_color, linewidth=0.6, alpha=0.55, color=agg_color,
           label=f'{method_cap} (132 galaxies)', zorder=3)
    ax.errorbar(bin_c, central_fractions,
                yerr=[central_fractions - p16_fractions,
                      p84_fractions - central_fractions],
                fmt='none', ecolor=agg_color, capsize=2, alpha=0.6, zorder=4)

    for key_, clr in [(hk, 'red'), (lk, 'green')]:
        tf = special_data[key_]['temp_fraction']
        if tf is not None:
            ax.step(np.concatenate([T_BINS[:-1], [T_BINS[-1]]]),
                    np.concatenate([tf, [tf[-1]]]),
                    where='post', color=clr, lw=2.0,
                    label=special_data[key_]['label'], zorder=5)

    ax.set_xscale('log')
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels([r'$10^3$', r'$10^4$', r'$10^5$',
                         r'$10^6$', r'$10^7$', r'$10^8$'])
    ax.set_xlim(1e3, 1e8)
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('DM contribution fraction', fontsize=12)
    ax.set_title(f'DM vs Temperature ({method_label}, {property_name} extremes)',
                 fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f'{save_prefix}_DM_temperature_{property_name}_comparison'
        f'_{method_label}.pdf',
        bbox_inches='tight', dpi=150)

    # ── Plot 5: Electron density profile ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    figs.append(fig)

    valid_tng = np.isfinite(master_ne_central) & (master_ne_central > 0)
    ax.plot(x_common[valid_tng], master_ne_central[valid_tng],
            color='black', lw=4.0, zorder=10,
            label=f'TNG {method_cap} (N={len(ne_med_interp_all)})')
    valid_pct = (np.isfinite(master_ne_p16) & np.isfinite(master_ne_p84)
                 & (master_ne_p16 > 0) & (master_ne_p84 > 0))
    ax.fill_between(x_common[valid_pct],
                    master_ne_p16[valid_pct], master_ne_p84[valid_pct],
                    color='gray', alpha=0.3, zorder=9,
                    label='TNG 16th–84th percentile')

    for key_, clr, mkr in [(hk, 'red', 's'), (lk, 'green', '^')]:
        ne_sp = special_data[key_]['ne_profile']
        if ne_sp is not None:
            vm = np.isfinite(ne_sp) & (ne_sp > 0)
            if vm.any():
                ax.plot(x_common[vm], ne_sp[vm],
                        f'{mkr}-', color=clr, markersize=3, lw=1.5,
                        label=special_data[key_]['label'], zorder=8)

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
    ax.set_title(
        f'Electron Density: TNG vs Models ({method_label}, {property_name} extremes)',
        fontsize=13)
    ax.set_xlim(0.0, 1.05)
    ax.axvline(1.0, color='gray', lw=1.2, ls='--', alpha=0.7)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(
        f'{save_prefix}_ne_profile_{property_name}_comparison'
        f'_{method_label}.pdf',
        dpi=150, bbox_inches='tight')

    return figs


def plot_healpix_suite(ell, valid_cl, valid_cl_norm, aniso_vals, aniso_ells,
                        aniso_labels_text, pctl_cl, method_label,
                        n_galaxies, save_prefix='healpix'):
    """
    Generate the three standard HEALPix summary plots:
      1. Angular power spectrum
      2. Normalised angular power spectrum
      3. Anisotropy ratio bar chart

    Parameters
    ----------
    ell : array-like
    valid_cl, valid_cl_norm, aniso_vals : ndarray, shape (N_gal, N_ell)
    aniso_ells : list of int  (multipole indices)
    aniso_labels_text : list of str
    pctl_cl : dict  {label: cl_array}  (individual galaxies to overlay)
    method_label : str
    n_galaxies : int
    save_prefix : str

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from . import statistics as stats_mod  # for aggregate_with_axis

    method = method_label.lower()
    figs = []

    agg  = lambda arr: stats_mod.aggregate_with_axis(arr, method=method, axis=0)
    p16  = lambda arr: np.percentile(arr, 16, axis=0)
    p84  = lambda arr: np.percentile(arr, 84, axis=0)

    cl_central      = agg(valid_cl)
    cl_p16, cl_p84  = p16(valid_cl), p84(valid_cl)

    cl_norm_central      = agg(valid_cl_norm)
    cl_norm_p16          = p16(valid_cl_norm)
    cl_norm_p84          = p84(valid_cl_norm)

    aniso_central     = agg(aniso_vals)
    aniso_p16_arr     = p16(aniso_vals)
    aniso_p84_arr     = p84(aniso_vals)

    colors_hp = plt.cm.viridis(np.linspace(0, 1, max(len(pctl_cl), 1)))

    # ── Plot 1: Angular power spectrum ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    figs.append(fig)
    ax.semilogy(ell[1:], cl_central[1:], color='black', lw=2.5,
                label=f'{method_label.capitalize()} (N={n_galaxies})')
    ax.fill_between(ell[1:], cl_p16[1:], cl_p84[1:],
                    color='steelblue', alpha=0.3, label='16th–84th percentile')
    for i, (label, cl) in enumerate(pctl_cl.items()):
        ax.semilogy(ell[1:], cl[1:],
                    color=colors_hp[i], lw=1.0, alpha=0.7, ls='--', label=label)
    ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
    ax.set_ylabel(r'$C_\ell$ [pc$^2$ cm$^{-6}$]', fontsize=13)
    ax.set_title(f'Angular Power Spectrum ({method_label})', fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_power_spectra_{method_label}.pdf',
                bbox_inches='tight', dpi=150)

    # ── Plot 2: Normalised power spectrum ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    figs.append(fig)
    ax.semilogy(ell[1:], cl_norm_central[1:], color='black', lw=2.5,
                label=f'{method_label.capitalize()} (N={n_galaxies})')
    ax.fill_between(ell[1:], cl_norm_p16[1:], cl_norm_p84[1:],
                    color='steelblue', alpha=0.3, label='16th–84th percentile')
    for i, (label, cl) in enumerate(pctl_cl.items()):
        if cl[0] > 0:
            ax.semilogy(ell[1:], cl[1:] / cl[0],
                        color=colors_hp[i], lw=1.0, alpha=0.7, ls='--',
                        label=label)
    ax.set_xlabel(r'Multipole $\ell$', fontsize=13)
    ax.set_ylabel(r'$C_\ell\ /\ C_0$', fontsize=13)
    ax.set_title(f'Normalized Angular Power Spectrum ({method_label})',
                 fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_power_spectra_normalised_{method_label}.pdf',
                bbox_inches='tight', dpi=150)

    # ── Plot 3: Anisotropy ratios ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    figs.append(fig)
    x_pos = np.arange(len(aniso_ells))
    bar_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(aniso_ells)))

    ax.bar(x_pos, aniso_central, color=bar_colors, edgecolor='k',
           linewidth=0.6, alpha=0.85, zorder=3,
           yerr=[aniso_central - aniso_p16_arr, aniso_p84_arr - aniso_central],
           capsize=5, error_kw={'lw': 1.5, 'ecolor': 'dimgray'})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(aniso_labels_text, fontsize=11)
    ax.set_ylabel(r'$\sqrt{C_\ell\,/\,C_0}$', fontsize=13)
    ax.set_title(f'DM Anisotropy Ratios — All Galaxies (N={n_galaxies})\n'
                 f'{method_label.capitalize()} ± 16th–84th percentile',
                 fontsize=13)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_ylim(0, None)
    for xi, med, p84v in zip(x_pos, aniso_central, aniso_p84_arr):
        ax.text(xi, p84v + 0.005, f'{med:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_anisotropy_ratios_{method_label}.pdf',
                bbox_inches='tight', dpi=150)

    return figs


def plot_lognormal_fits(fit_results, fiducial_lats, ncols=4):
    """
    Plot histogram + log-normal overlay for each fiducial latitude.

    Parameters
    ----------
    fit_results : dict  {lat: {...}}  (output of statistics.fit_lognormal)
    fiducial_lats : list of int/float
    ncols : int

    Returns
    -------
    list of matplotlib.figure.Figure  (one grid figure + one parameter figure)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    n_fid = len(fiducial_lats)
    nrows = int(np.ceil(n_fid / ncols))
    figs = []

    # ── Grid of histograms ───────────────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows))
    figs.append(fig)
    axes = np.atleast_1d(axes).ravel()

    for ax, lat in zip(axes, fiducial_lats):
        fr = fit_results.get(lat)
        if fr is None:
            ax.axis('off')
            continue
        sample = fr['sample']
        counts, bins, _ = ax.hist(
            sample, bins=60, density=True, color='steelblue',
            alpha=0.55, edgecolor='none', label='TNG50-1 sample')
        x = np.linspace(bins[0], bins[-1], 400)
        pdf = stats.lognorm.pdf(
            x, fr['sigma_ln'], loc=0, scale=np.exp(fr['mu_ln']))
        ax.plot(x, pdf, color='crimson', lw=2,
                label=(f"Log-normal fit\n"
                       f"$\\mu$={fr['mu_ln']:.2f}, "
                       f"$\\sigma$={fr['sigma_ln']:.2f}\n"
                       f"KS $p$={fr['ks_p']:.2f}"))
        ax.set_title(f"$b$ = {lat}°  (N={fr['n']})", fontsize=10)
        ax.set_xlabel(r'DM [pc cm$^{-3}$]', fontsize=9)
        ax.set_ylabel('Probability density', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)

    for ax in axes[n_fid:]:
        ax.axis('off')

    plt.suptitle(
        'DM Distribution at Fiducial Latitudes: Log-Normal Fits (132 galaxies)',
        fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('dm_lognormal_fits_grid.pdf', dpi=150, bbox_inches='tight')

    # ── Parameter trend plots ────────────────────────────────────────────
    lats_arr  = np.array(fiducial_lats, dtype=float)
    mu_arr    = np.array([fit_results[l]['mu_ln']    for l in fiducial_lats])
    sigma_arr = np.array([fit_results[l]['sigma_ln'] for l in fiducial_lats])

    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    figs.append(fig2)

    axA.plot(lats_arr, mu_arr, 'o-', color='darkorange', markersize=7, lw=1.5)
    axA.set_xlabel('Galactic Latitude $b$ (°)', fontsize=12)
    axA.set_ylabel(r'$\mu$ (mean of $\ln$ DM)', fontsize=12)
    axA.set_title(r'Log-normal $\mu$ vs Latitude', fontsize=12)
    axA.set_xticks(np.arange(-90, 91, 30))
    axA.grid(alpha=0.3)

    axB.plot(lats_arr, sigma_arr, 's-', color='teal', markersize=7, lw=1.5)
    axB.set_xlabel('Galactic Latitude $b$ (°)', fontsize=12)
    axB.set_ylabel(r'$\sigma$ (std of $\ln$ DM)', fontsize=12)
    axB.set_title(r'Log-normal $\sigma$ vs Latitude', fontsize=12)
    axB.set_xticks(np.arange(-90, 91, 30))
    axB.grid(alpha=0.3)

    plt.suptitle('Log-Normal Parameters vs Galactic Latitude',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('dm_lognormal_params_vs_latitude.pdf', dpi=150,
                bbox_inches='tight')

    return figs