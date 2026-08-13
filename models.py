import numpy as np
from scipy.interpolate import interp1d
from frb.halos.models import ModifiedNFW, MB04, YF17

# ==========================================
# Custom Model Classes
# ==========================================

class Stern18Model:
    """
    Custom model for Stern+18 featuring the density cliff at 145 kpc.
    """
    def __init__(self, r_vir=236.0):
        # We wrap r_vir in an object with a .value attribute to match FRB repo style
        self.r200 = type('R200', (object,), {'value': r_vir})()
        
        # Data points
        r_inner = np.array([10, 20, 30, 40, 60, 80, 100, 120, 135, 145])
        ne_inner = np.array([1.0e-2, 2.7e-3, 1.1e-3, 6.0e-4, 2.6e-4, 1.4e-4, 9.0e-5, 6.5e-5, 5.8e-5, 5.3e-5])
        r_outer = np.array([145.01, 160, 200, 250, 300])
        ne_outer = np.array([1.15e-5, 1.0e-5, 6.5e-6, 4.3e-6, 3.1e-6])

        # Interpolators
        self.f_in = interp1d(r_inner, np.log10(ne_inner), kind='cubic', bounds_error=False, fill_value=-10)
        self.f_out = interp1d(r_outer, np.log10(ne_outer), kind='linear', bounds_error=False, fill_value=-10)

    def ne(self, xyz):
        # xyz is (3, N), extract radial distance
        r = np.sqrt(np.sum(xyz**2, axis=0)) if xyz.ndim > 1 else xyz[0]
        ne_prof = np.zeros_like(r)
        
        mask_in = (r <= 145.0)
        mask_out = (r > 145.0)
        
        ne_prof[mask_in] = 10**self.f_in(r[mask_in])
        ne_prof[mask_out] = 10**self.f_out(r[mask_out])
        
        # Ensure values below 10kpc or above 300kpc are handled
        ne_prof[r < 10] = 1.0e-2 
        return ne_prof

class Fielding17Model:
    """
    Custom model for Fielding+17 starting at 15.1 kpc.
    """
    def __init__(self, r_vir=236.0):
        self.r200 = type('R200', (object,), {'value': r_vir})()
        
        # Data points
        r_curve = np.array([15.1, 20, 30, 45, 60, 80, 100, 150, 200, 250, 300])
        ne_curve = np.array([3.5e-3, 1.5e-3, 4.5e-4, 2.7e-4, 2.3e-4, 1.8e-4, 1.4e-4, 7.5e-5, 4.2e-5, 2.8e-5, 2.1e-5])

        self.f_curve = interp1d(r_curve, np.log10(ne_curve), kind='cubic', bounds_error=False, fill_value=-10)

    def ne(self, xyz):
        r = np.sqrt(np.sum(xyz**2, axis=0)) if xyz.ndim > 1 else xyz[0]
        ne_prof = np.zeros_like(r)
        
        mask = (r >= 15.1)
        ne_prof[mask] = 10**self.f_curve(r[mask])
        ne_prof[r < 15.1] = 1e-10 # Mimic the vertical drop/empty inner cavity
        return ne_prof

# ==========================================
# Updated Analytical Functions
# ==========================================

def get_analytical_models(log_mhalo=12.0):
    """
    Get dictionary of analytical CGM models, including Fielding and Stern.
    """
    # Note: R_vir = 236.0 corresponds to logM ~ 12.1-12.2 depending on cosmology.
    # We pass it here to match your specific interpolation data.
    r_vir_fixed = 236.0 

    return {
        'Modified NFW default': ModifiedNFW(log_Mhalo=log_mhalo),
        'Modified NFW y2 a2': ModifiedNFW(alpha=2, y0=2, log_Mhalo=log_mhalo),
        'Modified NFW y4 a2': ModifiedNFW(alpha=2, y0=4, log_Mhalo=log_mhalo),
        'MB04': MB04(log_Mhalo=log_mhalo),
        'YF17': YF17(log_Mhalo=log_mhalo),
        'Stern+18': Stern18Model(r_vir=r_vir_fixed),
        'Fielding+17': Fielding17Model(r_vir=r_vir_fixed)
    }

def evaluate_model(model, x_array):
    """
    Evaluate electron density for a model.
    """
    # Use the model's intrinsic r200 to convert normalized x back to physical r
    r200_kpc = model.r200.value if hasattr(model.r200, 'value') else model.r200
    r_kpc = x_array * r200_kpc
    
    # Prepare xyz array (3, N) for the .ne() method
    xyz = np.zeros((3, len(r_kpc)))
    xyz[0, :] = r_kpc
    
    ne_prof = model.ne(xyz)
    
    if hasattr(ne_prof, 'value'):
        ne_prof = ne_prof.value
    
    return ne_prof

def rank_models(models, x_common, master_ne_median):
    """
    Rank models by goodness of fit to TNG data.
    """
    valid_tng = np.isfinite(master_ne_median) & (master_ne_median > 0)
    model_results = []
    
    for name, model in models.items():
        ne_prof = evaluate_model(model, x_common)
        
        # Safety for log calculations
        ne_prof_clipped = np.where(ne_prof <= 1e-10, 1e-10, ne_prof)
        
        valid_m = np.isfinite(ne_prof) & (ne_prof > 0)
        overlap = valid_tng & valid_m & (ne_prof > 1e-9)
        
        if np.sum(overlap) > 0:
            rmse = np.sqrt(np.mean(
                (np.log10(master_ne_median[overlap]) - 
                 np.log10(ne_prof_clipped[overlap]))**2
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


# ==========================================
# Milky Way foreground DM models (Yamasaki & Totani)
# ==========================================

def dm_halo_yt19(ra, dec):
    """
    Yamasaki & Totani (2019) polynomial DM(l,b) halo model, evaluated at
    equatorial coordinates (ra, dec) in degrees. Returns DM in pc cm^-3.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    x = [[250.12, -871.06, 1877.5, -2553.0, 2181.3, -1127.5, 321.72, -38.905],
         [-154.82, 783.43, -1593.9, 1727.6, -1046.5, 332.09, -42.815, 0],
         [-116.72, -76.815, 428.49, -419.00, 174.60, -27.610, 0, 0],
         [216.67, -193.30, 12.234, 32.145, -8.3602, 0, 0, 0],
         [-129.95, 103.80, -22.800, 0.44171, 0, 0, 0, 0],
         [39.652, -21.398, 2.7694, 0, 0, 0, 0, 0],
         [-6.1926, 1.6162, 0, 0, 0, 0, 0, 0],
         [0.39346, 0, 0, 0, 0, 0, 0, 0]]
    x = np.array(x)
    dm = 0
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    l, b = c.galactic.l.degree, c.galactic.b.degree
    if l > 180:
        l = l - 360
    l, b = (l / 180.0) * np.pi, (b / 180.0) * np.pi
    for i in range(8):
        for j in range(8):
            dm += x[i, j] * np.abs(l) ** i * np.abs(b) ** j
    return dm


# Yamasaki & Totani (2020) spherical-halo model defaults
YT20_UPSILON_FIXED = 2.6
YT20_R_VIR_KPC = 260.0
YT20_D_SUN_KPC = 8.5     # Galactocentric distance of the Sun (as in the paper)
YT20_N0_SPH_PAPER = 3.7e-4    # cm^-3, original paper fiducial value
YT20_R_S_PAPER = YT20_R_VIR_KPC / 12.0   # kpc, r_vir / c_vir


def n_sphere_yt20(r_kpc, n0_sph, r_s, upsilon=YT20_UPSILON_FIXED):
    """Yamasaki & Totani (2020) spherical-halo electron density profile."""
    x = r_kpc / r_s
    return n0_sph * np.exp(-upsilon * (1.0 - np.log(1.0 + x) / x))


def dm_halo_sphere_yt20(l_deg, b_deg, n0_sph, r_s, r_vir_kpc=YT20_R_VIR_KPC, d_sun_kpc=YT20_D_SUN_KPC):
    """
    Spherical-halo DM toward Galactic (l, b), in pc cm^-3, by direct
    line-of-sight integration of the n_sphere_yt20 model with the given
    (n0_sph, r_s).
    """
    from scipy.integrate import quad

    l, b = np.radians(l_deg), np.radians(b_deg)
    cosb, cosl = np.cos(b), np.cos(l)
    # path length to the virial sphere (law of cosines)
    bcoef = -2.0 * d_sun_kpc * cosb * cosl
    ccoef = d_sun_kpc**2 - r_vir_kpc**2
    smax = (-bcoef + np.sqrt(bcoef**2 - 4 * ccoef)) / 2.0

    def integrand(s):
        r = np.sqrt(s**2 + d_sun_kpc**2 - 2 * s * d_sun_kpc * cosb * cosl)
        return n_sphere_yt20(r, n0_sph, r_s)

    dm_kpc_cm3, _ = quad(integrand, 0, smax, limit=200)
    return dm_kpc_cm3 * 1e3   # kpc -> pc


# ==========================================
# Power-law fitting and model ranking
# ==========================================

def power_law(x, A, alpha):
    return A * x ** (-alpha)


def fit_power_law_log(x, y):
    """
    Log-based power-law fit: n_e(x) = A * x^-alpha.

    A log-log linear regression (np.polyfit on log10(x), log10(y))
    supplies the initial guess, then scipy.optimize.curve_fit refines it
    with nonlinear least squares. Returns (A_fit, alpha_fit, perr, x_fit)
    where perr are the 1-sigma parameter errors and x_fit is the
    (masked, finite, positive) x array the fit was evaluated on.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_fit, y_fit = x[mask], y[mask]

    log_x, log_y = np.log10(x_fit), np.log10(y_fit)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    A_guess, alpha_guess = 10 ** intercept, -slope

    popt, pcov = curve_fit(power_law, x_fit, y_fit, p0=[A_guess, alpha_guess], maxfev=10000)
    A_fit, alpha_fit = popt
    perr = np.sqrt(np.diag(pcov))
    return A_fit, alpha_fit, perr, x_fit


def chi2_models(models, x, data, sigma):
    """
    Rank analytical models by reduced chi^2 against data, using sigma
    derived from the 16th-84th percentile spread: sigma ~= (P84 - P16) / 2.
    """
    results = []
    for name, model in models.items():
        pred = evaluate_model(model, x)
        mask = np.isfinite(data) & np.isfinite(pred) & np.isfinite(sigma) & (sigma > 0)
        chi2 = np.sum(((data[mask] - pred[mask]) / sigma[mask])**2)
        dof = mask.sum()
        results.append({'name': name, 'chi2': chi2, 'reduced_chi2': chi2 / dof})
    return sorted(results, key=lambda z: z['reduced_chi2'])
