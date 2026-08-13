"""
Coordinate transformation utilities.
"""

import numpy as np

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian to spherical coordinates.
    
    Parameters
    ----------
    x, y, z : array_like
        Cartesian coordinates
        
    Returns
    -------
    r, theta, phi : ndarray
        Spherical coordinates (radius, polar angle, azimuthal angle)
    """
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical to Cartesian coordinates.
    
    Parameters
    ----------
    r : array_like
        Radial distance
    theta : array_like
        Polar angle
    phi : array_like
        Azimuthal angle
        
    Returns
    -------
    coords : ndarray
        Cartesian coordinates (x, y, z)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)