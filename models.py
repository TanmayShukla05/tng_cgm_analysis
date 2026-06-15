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
