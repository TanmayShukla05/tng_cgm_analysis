"""
TNG CGM Analysis Package
A package for analyzing circumgalactic medium (CGM) dispersion measures
from TNG simulations.
"""

__version__ = "1.0.0"

# Import all modules for easy access
from . import config
from . import constants
from . import coordinates
from . import thermodynamics
from . import geometry
from . import los
from . import density
from . import data_loader
from . import processing
from . import statistics
from . import plotting
from . import healpix_analysis
from . import models
from . import cache_manager

__all__ = [
    'config',
    'constants',
    'coordinates',
    'thermodynamics',
    'geometry',
    'los',
    'density',
    'data_loader',
    'processing',
    'statistics',
    'plotting',
    'healpix_analysis',
    'models',
]