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

# ============================================================================
# HEALPix Geometry Analysis Functions
# ============================================================================

def compute_disk_to_sphere_ratio(healpix_map, nside=64):
    """
    Compute the ratio of DM in disk regions vs polar regions.
    
    Parameters
    ----------
    healpix_map : ndarray
        HEALPix map of DM values
    nside : int, optional
        HEALPix resolution parameter (default: 64)
        
    Returns
    -------
    ratio : float
        Disk/Polar DM ratio
        - ratio > 1.2: disk-like (enhanced DM in galactic plane)
        - ratio ~ 1.0: spherically symmetric
        - ratio < 0.8: polar enhancement (rare)
    disk_dm : float
        Median DM in disk region (|b| < 30°)
    polar_dm : float
        Median DM in polar region (|b| > 60°)
        
    Notes
    -----
    Disk region: |latitude| < 30° → colatitude θ ∈ [π/3, 2π/3]
    Polar region: |latitude| > 60° → colatitude θ < π/6 or θ > 5π/6
    """
    import healpy as hp
    
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    
    # Define regions in colatitude (θ)
    # Disk: galactic plane ± 30° → θ ∈ [60°, 120°] = [π/3, 2π/3]
    disk_mask = (theta > np.pi/3) & (theta < 2*np.pi/3)
    
    # Poles: |b| > 60° → θ < 30° or θ > 150° = θ < π/6 or θ > 5π/6
    polar_mask = (theta < np.pi/6) | (theta > 5*np.pi/6)
    
    # Get valid (non-NaN) pixels
    valid = ~np.isnan(healpix_map)
    
    disk_pixels = healpix_map[disk_mask & valid]
    polar_pixels = healpix_map[polar_mask & valid]
    
    if len(disk_pixels) == 0 or len(polar_pixels) == 0:
        return np.nan, np.nan, np.nan
    
    disk_dm = np.median(disk_pixels)
    polar_dm = np.median(polar_pixels)
    ratio = disk_dm / polar_dm if polar_dm > 0 else np.nan
    
    return ratio, disk_dm, polar_dm


def correlate_with_galaxy_orientation(healpix_map, disk_vector, nside=64):
    """
    Compute correlation between DM pattern and galaxy disk orientation.
    
    Parameters
    ----------
    healpix_map : ndarray
        DM HEALPix map
    disk_vector : ndarray, shape (3,)
        Unit vector of disk angular momentum [x, y, z]
        Typically from SubhaloSpin in TNG catalogs
    nside : int, optional
        HEALPix resolution (default: 64)
        
    Returns
    -------
    correlation : float
        Pearson correlation coefficient between DM and disk-plane indicator
        - Positive correlation: DM enhanced when looking through disk
        - Negative correlation: DM enhanced at poles
        - Near zero: no alignment with disk geometry
        
    Notes
    -----
    The disk-plane indicator is (1 - |cos(θ)|), where θ is the angle
    between the line of sight and the disk normal. This is:
    - High (→1) when looking through the galactic plane
    - Low (→0) when looking perpendicular to the plane (at poles)
    """
    import healpy as hp
    
    npix = hp.nside2npix(nside)
    
    # Get unit vectors for each HEALPix pixel (line-of-sight directions)
    vectors = np.array(hp.pix2vec(nside, np.arange(npix)))  # shape: (3, npix)
    
    # Compute angle from disk normal
    cos_theta = np.abs(np.dot(disk_vector, vectors))  # |cos(θ)|
    
    # Disk-plane indicator: high when looking through disk
    disk_indicator = 1.0 - cos_theta
    
    # Compute correlation with DM
    valid = ~np.isnan(healpix_map)
    if np.sum(valid) < 10:
        return np.nan
    
    corr = np.corrcoef(healpix_map[valid], disk_indicator[valid])[0, 1]
    
    return corr


def extract_quadrupole_pattern(healpix_map, nside=64):
    """
    Extract the quadrupole (ℓ=2) component from a HEALPix map.
    
    Parameters
    ----------
    healpix_map : ndarray
        Input HEALPix map
    nside : int, optional
        HEALPix resolution (default: 64)
        
    Returns
    -------
    quad_map : ndarray
        HEALPix map containing only the quadrupole component
        
    Notes
    -----
    Decomposes the map into spherical harmonics up to ℓ=5, then
    reconstructs using only the ℓ=2 (quadrupole) terms.
    
    The quadrupole has 5 m-modes: m ∈ {-2, -1, 0, 1, 2}
    This creates a four-lobed pattern that can reveal:
    - Disk geometry effects
    - Bipolar outflows
    - Satellite accretion streams
    """
    import healpy as hp
    
    # Compute spherical harmonic coefficients
    alm = hp.map2alm(healpix_map, lmax=5)
    
    # Create array for quadrupole-only alm
    alm_quad = np.zeros_like(alm, dtype=complex)
    
    # Keep only ℓ=2 terms (m = 0, 1, 2; negative m stored implicitly)
    for m in range(3):  # m = 0, 1, 2
        idx = hp.Alm.getidx(5, 2, m)
        alm_quad[idx] = alm[idx]
    
    # Reconstruct map from quadrupole only
    quad_map = hp.alm2map(alm_quad, nside)
    
    return quad_map


def compute_multipole_anisotropy(cl):
    """
    Compute anisotropy metrics from angular power spectrum.
    
    Parameters
    ----------
    cl : ndarray
        Angular power spectrum C_ℓ
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'dipole': √(C_1/C_0) - dipole amplitude
        - 'quadrupole': √(C_2/C_0) - quadrupole amplitude
        - 'octupole': √(C_3/C_0) - octupole amplitude
        - 'quad_to_dipole': C_2/C_1 - disk geometry indicator
        - 'even_to_odd': (C_2+C_4)/(C_1+C_3) - parity asymmetry
        
    Notes
    -----
    For disk-like distributions:
    - quad_to_dipole > 1: quadrupole dominates (typical for disks)
    - even_to_odd > 1: even multipoles enhanced
    
    For spherical distributions:
    - All ratios ~ 1
    """
    if cl is None or len(cl) < 5 or cl[0] <= 0:
        return {}
    
    metrics = {
        'dipole': np.sqrt(cl[1] / cl[0]),
        'quadrupole': np.sqrt(cl[2] / cl[0]),
        'octupole': np.sqrt(cl[3] / cl[0]),
        'quad_to_dipole': cl[2] / cl[1] if cl[1] > 0 else np.nan,
        'even_to_odd': (cl[2] + cl[4]) / (cl[1] + cl[3]) if (cl[1] + cl[3]) > 0 else np.nan,
    }
    
    return metrics
