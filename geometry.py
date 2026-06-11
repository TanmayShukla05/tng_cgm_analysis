"""
Galaxy geometry and rotation utilities.
"""

import numpy as np
from scipy.spatial.transform import Rotation

def half_mass_radius(coords, masses):
    """
    Calculate half-mass radius.
    
    Parameters
    ----------
    coords : ndarray
        Particle coordinates
    masses : ndarray
        Particle masses
        
    Returns
    -------
    r_half : float
        Half-mass radius
    """
    r = np.linalg.norm(coords, axis=1)
    idx = np.argsort(r)
    cum = np.cumsum(masses[idx])
    return r[idx][np.searchsorted(cum, cum[-1] / 2.0)]

def inertia_tensor(coords, masses):
    """
    Calculate inertia tensor.
    
    Parameters
    ----------
    coords : ndarray
        Particle coordinates
    masses : ndarray
        Particle masses
        
    Returns
    -------
    I : ndarray
        Inertia tensor (3x3)
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r2 = x**2 + y**2 + z**2
    m = masses
    
    I = np.zeros((3, 3))
    I[0, 0] = np.sum(m * (r2 - x*x))
    I[1, 1] = np.sum(m * (r2 - y*y))
    I[2, 2] = np.sum(m * (r2 - z*z))
    I[0, 1] = I[1, 0] = -np.sum(m * x * y)
    I[0, 2] = I[2, 0] = -np.sum(m * x * z)
    I[1, 2] = I[2, 1] = -np.sum(m * y * z)
    
    return I

def compute_galaxy_rotation_matrices(star_coords, star_masses,
                                     sfgas_coords, sfgas_masses,
                                     subhalo_pos, hmr_factor=2.0):
    """
    Compute rotation matrices for face-on and edge-on views.
    
    Parameters
    ----------
    star_coords : ndarray
        Star particle coordinates
    star_masses : ndarray
        Star particle masses
    sfgas_coords : ndarray
        Star-forming gas coordinates
    sfgas_masses : ndarray
        Star-forming gas masses
    subhalo_pos : ndarray
        Subhalo position
    hmr_factor : float
        Factor for half-mass radius cutoff
        
    Returns
    -------
    R_fo : ndarray
        Face-on rotation matrix
    R_eo : ndarray
        Edge-on rotation matrix
    eigvecs : ndarray
        Eigenvectors of inertia tensor
    """
    sc = star_coords - subhalo_pos
    sgc = sfgas_coords - subhalo_pos
    
    r_half = half_mass_radius(sc, star_masses)
    r_cut = hmr_factor * r_half
    
    mask_s = np.linalg.norm(sc, axis=1) < r_cut
    mask_sg = np.linalg.norm(sgc, axis=1) < r_cut
    
    combined_coords = np.vstack([sc[mask_s], sgc[mask_sg]])
    combined_masses = np.concatenate([star_masses[mask_s], sfgas_masses[mask_sg]])
    
    I = inertia_tensor(combined_coords, combined_masses)
    eigvals, eigvecs = np.linalg.eigh(I)
    
    normal = eigvecs[:, 0]
    z_hat = np.array([0., 0., 1.])
    axis = np.cross(normal, z_hat)
    sinA = np.linalg.norm(axis)
    cosA = np.dot(normal, z_hat)
    
    if sinA < 1e-10:
        R_fo = np.eye(3)
    else:
        axis /= sinA
        angle = np.arctan2(sinA, cosA)
        R_fo = Rotation.from_rotvec(angle * axis).as_matrix()
    
    R_eo = Rotation.from_euler('x', 90, degrees=True).as_matrix() @ R_fo
    
    return R_fo, R_eo, eigvecs

def apply_rotation(coords, R):
    """
    Apply rotation matrix to coordinates.
    
    Parameters
    ----------
    coords : ndarray
        Coordinates to rotate
    R : ndarray
        Rotation matrix
        
    Returns
    -------
    coords_rotated : ndarray
        Rotated coordinates
    """
    return (R @ coords.T).T