"""
Density projection and visualization utilities.
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def plot_density_projection(coords_rotated, masses, h,
                            view='faceon', bins=500, lim_kpc=150,
                            title=None, ax=None, cmap='magma',
                            vmin=4, vmax=9):
    """
    Plot gas density projection.
    
    Parameters
    ----------
    coords_rotated : ndarray
        Rotated coordinates
    masses : ndarray
        Gas masses
    h : float
        Hubble parameter
    view : str
        'faceon' or 'edgeon'
    bins : int
        Number of bins
    lim_kpc : float
        Plot limit in kpc
    title : str, optional
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on
    cmap : str
        Colormap
    vmin, vmax : float
        Color scale limits
        
    Returns
    -------
    ax : matplotlib axis
        The plot axis
    """
    lim = lim_kpc * h
    
    if view == 'faceon':
        xi, yi = coords_rotated[:, 0], coords_rotated[:, 1]
        xlabel, ylabel = 'x [kpc]', 'y [kpc]'
    else:
        xi, yi = coords_rotated[:, 0], coords_rotated[:, 2]
        xlabel, ylabel = 'x [kpc]', 'z [kpc]'
    
    stat, xe_, ye_, _ = scipy.stats.binned_statistic_2d(
        xi, yi, masses, statistic='sum', bins=bins,
        range=[[-lim, lim], [-lim, lim]])
    
    dx_kpc = (2 * lim / bins) / h
    sigma = stat * 1e10 / h / (dx_kpc**2)
    sigma = np.where(sigma > 0, sigma, np.nan)
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(np.log10(sigma).T, origin='lower', cmap=cmap, aspect='equal',
                   extent=[-lim_kpc, lim_kpc, -lim_kpc, lim_kpc],
                   vmin=vmin, vmax=vmax)
    
    ax.figure.colorbar(im, ax=ax,
                      label=r'$\log_{10}(\Sigma_{\rm gas})$ [$M_\odot$ kpc$^{-2}$]')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Gas density ({view})')
    
    return ax

def plot_galaxy_orientations(coords_faceon, coords_edgeon, masses, h,
                             lim_kpc=150, bins=400, suptitle=''):
    """
    Plot face-on and edge-on views side by side.
    
    Parameters
    ----------
    coords_faceon : ndarray
        Face-on rotated coordinates
    coords_edgeon : ndarray
        Edge-on rotated coordinates
    masses : ndarray
        Gas masses
    h : float
        Hubble parameter
    lim_kpc : float
        Plot limit in kpc
    bins : int
        Number of bins
    suptitle : str
        Overall title
        
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    plot_density_projection(coords_faceon, masses, h, view='faceon',
                           bins=bins, lim_kpc=lim_kpc, title='Face-on', 
                           ax=axes[0])
    plot_density_projection(coords_edgeon, masses, h, view='edgeon',
                           bins=bins, lim_kpc=lim_kpc, title='Edge-on', 
                           ax=axes[1])
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)
    
    plt.tight_layout()
    return fig