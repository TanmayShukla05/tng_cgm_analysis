"""
cache_manager.py
Handles saving/loading of processed result caches with hash-based invalidation.
"""

import os
import pickle
import hashlib
import numpy as np

__all__ = [
    'compute_results_hash',
    'save_processed_results',
    'load_processed_results',
]

REQUIRED_KEYS = {
    'results', 'special_data', 'galaxy_orientations',
    'med_raw', 'p16_raw', 'p84_raw',
    'med_sinb', 'p16_sinb', 'p84_sinb',
    'med_lon', 'p16_lon', 'p84_lon',
    'sp_dm_raw', 'sp_dm_sinb', 'sp_dm_lon',
}


def compute_results_hash(selected_ids, selected_orig, config, special_ids):
    """
    Compute a SHA-256 hash of all inputs that determine the processed outputs.
    If any input changes, the processed cache should be regenerated.

    Parameters
    ----------
    selected_ids : array-like
    selected_orig : array-like
    config : module or object with attributes SNAP, N_PTS_LOS, APPLY_TWO_PHASE,
             MAX_DIST, N_BINS_NE, SPACING, LATITUDES, LONGITUDES
    special_ids : dict

    Returns
    -------
    str : hex digest
    """
    h = hashlib.sha256()
    h.update(np.asarray(selected_ids).tobytes())
    h.update(np.asarray(selected_orig).tobytes())
    h.update(str(config.SNAP).encode())
    h.update(str(config.N_PTS_LOS).encode())
    h.update(str(config.APPLY_TWO_PHASE).encode())
    h.update(str(config.MAX_DIST).encode())
    h.update(str(config.N_BINS_NE).encode())
    h.update(str(config.SPACING).encode())
    h.update(np.asarray(config.LATITUDES).tobytes())
    h.update(np.asarray(config.LONGITUDES).tobytes())
    h.update(str(special_ids).encode())
    return h.hexdigest()


def save_processed_results(filepath, **kwargs):
    """
    Save processed results dict to a pickle file.

    Parameters
    ----------
    filepath : str
    **kwargs : arbitrary keyword arguments to store
    """
    with open(filepath, 'wb') as f:
        pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓ Saved processed results → {filepath}")


def load_processed_results(filepath, required_keys=None):
    """
    Load processed results from a pickle file.

    Parameters
    ----------
    filepath : str
    required_keys : set or None
        If provided, validate that all keys are present.

    Returns
    -------
    dict or None
        Returns None if file missing, corrupt, or keys absent.
    """
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  ⚠ Error loading cache {filepath}: {e}")
        return None

    keys_to_check = required_keys if required_keys is not None else REQUIRED_KEYS
    missing = keys_to_check - data.keys()
    if missing:
        print(f"  ⚠ Cache missing keys: {missing}")
        return None

    print(f"  ✓ Loaded processed results ← {filepath}")
    return data