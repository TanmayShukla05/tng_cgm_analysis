"""
Configuration parameters for TNG CGM analysis.
"""

import os

# Paths
MWPATH = '../sims.TNG/TNG50-1/postprocessing/MWM31s'
BASEPATH = '../sims.TNG/TNG50-1/output'
IDS_HDF5_PATH = 'ids_MW.hdf5'

# Simulation parameters
SNAP = 99

DISK_RADIUS_MULT = 5.0   # disk_radius = DISK_RADIUS_MULT * DiskScaleLength
DISK_HEIGHT_MULT = 3.0   # disk_height = DISK_HEIGHT_MULT * DiskScaleHeightThin

# Analysis parameters
N_PTS_LOS = 500
APPLY_TWO_PHASE = True
MAX_DIST = None
N_BINS_NE = 60
SPACING = 'log'

# Temperature bins (logspace)
import numpy as np
T_BINS = np.logspace(3, 8, 5)

# Latitude and longitude grids
LATITUDES = np.arange(-89, 89, 1)
LONGITUDES = np.arange(0, 360, 1)

# HEALPix parameters
NSIDE = 64

# Cache files
CACHE_FILE = "galaxy_processing_cache.pkl"
HEALPIX_CACHE_FILE = "healpix_dm_maps_cache.pkl"

# Plotting parameters
LIM_KPC = 150
BINS = 400
VMIN = 4
VMAX = 9

# Statistical aggregation method: 'median' or 'mean'
AGGREGATION_METHOD = 'median'

def set_aggregation_method(method):
    """Set the aggregation method globally."""
    global AGGREGATION_METHOD
    if method not in ['median', 'mean']:
        raise ValueError("method must be 'median' or 'mean'")
    AGGREGATION_METHOD = method

def get_aggregation_method():
    """Get the current aggregation method."""
    return AGGREGATION_METHOD
