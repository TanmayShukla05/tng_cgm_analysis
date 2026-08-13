"""
Line-of-sight construction and analysis.
"""

import math
import numpy as np
from .coordinates import spherical_to_cartesian

def _tuna_can_minr(theta, phi, disk_radius, disk_height):
    """
    Calculate minimum radius for diskless line of sight.
    
    Parameters
    ----------
    theta : float
        Polar angle
    phi : float
        Azimuthal angle
    disk_radius : float
        Disk radius
    disk_height : float
        Disk height
        
    Returns
    -------
    r_min : float
        Minimum radius
    """
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    
    if cos_t < 1e-9:
        return disk_radius / sin_t
    if sin_t < 1e-9:
        return disk_height / cos_t
    
    return min(disk_height / cos_t, disk_radius / sin_t)

def construct_diskless_LOS(theta, phi, disk_radius, disk_height,
                           r_max, n_pts=500, spacing='log'):
    """
    Construct a diskless line of sight.
    
    Parameters
    ----------
    theta : float
        Polar angle
    phi : float
        Azimuthal angle
    disk_radius : float
        Disk radius to avoid
    disk_height : float
        Disk height to avoid
    r_max : float
        Maximum radius
    n_pts : int
        Number of points
    spacing : str
        'log' or 'linear'
        
    Returns
    -------
    LOS_pts : ndarray
        Line of sight points
    r : ndarray
        Radii
    """
    r_min = max(_tuna_can_minr(theta, phi, disk_radius, disk_height), 1e-3)
    
    if spacing == 'log':
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_pts)
    else:
        r = np.linspace(r_min, r_max, n_pts)
    
    return spherical_to_cartesian(r, theta, phi), r

def find_nearest_cells(LOS_pts, cell_coords, max_dist=None, chunk_size=2000):
    """
    Find nearest cells to line-of-sight points.
    
    Parameters
    ----------
    LOS_pts : ndarray
        Line of sight points
    cell_coords : ndarray
        Cell coordinates
    max_dist : float, optional
        Maximum distance threshold
    chunk_size : int
        Chunk size for processing
        
    Returns
    -------
    idx : ndarray
        Indices of nearest cells
    valid : ndarray
        Boolean array of valid points
    """
    M = len(LOS_pts)
    idx = np.empty(M, dtype=np.intp)
    dist2_best = np.empty(M)
    
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        diff = cell_coords[np.newaxis, :, :] - LOS_pts[start:end, np.newaxis, :]
        d2 = np.sum(diff**2, axis=2)
        best = np.argmin(d2, axis=1)
        idx[start:end] = best
        dist2_best[start:end] = d2[np.arange(end - start), best]
    
    valid = (dist2_best <= max_dist**2 
             if max_dist is not None 
             else np.ones(M, dtype=bool))
    
    return idx, valid