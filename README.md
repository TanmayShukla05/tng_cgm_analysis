# TNG CGM Analysis

A Python package for analyzing circumgalactic medium (CGM) dispersion measures (DM) around Milky Way / M31-analog galaxies in the **TNG50-1** IllustrisTNG cosmological simulation.

The pipeline traces synthetic lines of sight (LOS) through a galaxy's gas halo, computes the resulting dispersion measure as a function of sky position (galactic latitude/longitude or full HEALPix sky maps), and compares the simulated electron density profiles against analytical CGM models from the literature (NFW-type halo models, Stern+18, Fielding+17, etc.).

## Features

- **Diskless line-of-sight construction** — a "tuna-can" exclusion geometry removes the galactic disk from each sight line so that only halo gas contributes to the DM.
- **Two-phase ISM correction** — corrects electron density for the hot/cold ISM phase mixing in star-forming gas cells.
- **Galaxy orientation** — computes face-on / edge-on rotation matrices from the stellar + star-forming gas inertia tensor.
- **DM vs. latitude / longitude profiles** — aggregated across a galaxy sample with configurable median/mean statistics and 16th–84th percentile bands.
- **HEALPix sky maps** — full-sky DM maps per galaxy, angular power spectra, quadrupole extraction, and disk-vs-polar anisotropy diagnostics.
- **Electron density radial profiles** — binned, interpolated n_e(r/R200c) profiles compared against analytical halo models.
- **SFR/stellar-mass classification** — star-forming / green-valley / quiescent classification using Donnari et al. (2019) main-sequence boundaries.
- **Caching** — pickle-based caching of per-galaxy and HEALPix results to avoid reprocessing.

## Repository Structure

| Module | Description |
|---|---|
| `config.py` | Paths, simulation snapshot, grid resolution, and other run-time parameters. |
| `constants.py` | Physical constants (proton mass, Boltzmann constant, hydrogen fraction, etc.) and ISM temperature thresholds. |
| `coordinates.py` | Cartesian ↔ spherical coordinate transforms. |
| `thermodynamics.py` | Gas temperature, internal energy, electron number density, and two-phase ISM correction. |
| `geometry.py` | Half-mass radius, inertia tensor, face-on/edge-on rotation matrices, and HEALPix-map geometry diagnostics (disk/polar ratio, orientation correlation, quadrupole extraction, multipole anisotropy). |
| `los.py` | Diskless line-of-sight construction and nearest-gas-cell lookup. |
| `data_loader.py` | Loads subhalo catalogs, ID mappings, simulation headers, and per-galaxy gas cutouts from HDF5. |
| `processing.py` | Main per-galaxy and sample-wide DM pipeline (LOS tracing → DM, temperature binning, electron density profile). |
| `healpix_analysis.py` | Full-sky HEALPix DM map generation, angular power spectra, and related plots. |
| `statistics.py` | Aggregation (median/mean), percentiles, DM profile statistics, and SFR/M* galaxy classification. |
| `models.py` | Analytical CGM electron density models (`ModifiedNFW`, `MB04`, `YF17` from the `frb` package, plus custom `Stern18Model` and `Fielding17Model`) and model-ranking by RMSE against TNG data. |
| `plotting.py` | Plotting utilities: DM vs. latitude/longitude, DM vs. temperature, electron density profile, SFR–M* diagram. |
| `density.py` | Gas surface density projections (face-on/edge-on). |

## Installation

```bash
git clone https://github.com/TanmayShukla05/tng_cgm_analysis.git
cd tng_cgm_analysis
pip install -e .
```

### Dependencies

- numpy, scipy, matplotlib, pandas, astropy
- h5py
- healpy
- [`illustris_python`](https://github.com/illustristng/illustris_python) (for reading TNG group catalogs/snapshots)
- [`frb`](https://github.com/FRBs/FRB) (for `ModifiedNFW`, `MB04`, `YF17` analytical halo models)

> Note: `illustris_python` and `frb` are not on PyPI and aren't listed in `setup.py`'s `install_requires` — install them separately from their GitHub repos.

## Data Requirements

This package expects access to:
- TNG50-1 simulation output (snapshot + group catalogs), e.g. via `BASEPATH` in `config.py`.
- MW/M31-analog host catalog (`mwm31s_hostcatalog.hdf5`) and HVC/cloud catalog (`cloud_catalog.hdf5`) under `MWPATH`.
- A precomputed list of selected subhalo IDs (`ids_MW.hdf5`).
- Per-galaxy gas cutouts at `{MWPATH}/cutouts/snap_0{SNAP}/{halo_id}.hdf5`, containing `PartType0` fields: `RotatedCoordinates`, `Density`, `ElectronAbundance`, `InternalEnergy`, `StarFormationRate`.

These paths are configured in `config.py`.

## Basic Usage

```python
from tng_cgm_analysis import config, data_loader, processing, statistics, plotting

# Load catalogs
selected_ids = data_loader.load_subhalo_ids(config.IDS_HDF5_PATH)
mw_catalog, hvc_catalog = data_loader.open_catalogs(config.MWPATH)
header = data_loader.load_header(config.BASEPATH, config.SNAP)
h = header['HubbleParam']

selected_orig = data_loader.get_id_mapping(mw_catalog, selected_ids)

# Run the DM pipeline across the sample
results = processing.process_all_galaxies(
    selected_ids, selected_orig, mw_catalog, hvc_catalog, h,
    config.LATITUDES, config.LONGITUDES,
    cache_file=config.CACHE_FILE,
    snap=config.SNAP,
    n_pts=config.N_PTS_LOS,
    apply_two_phase=config.APPLY_TWO_PHASE,
    max_dist=config.MAX_DIST,
    spacing=config.SPACING,
)

# Aggregate and plot DM vs. latitude
central, p16, p84 = statistics.compute_dm_stats(
    results['all_galaxy_dms'], config.LATITUDES
)
plotting.plot_dm_vs_latitude(config.LATITUDES, central, p16, p84)
```

## License

MIT License (see `LICENSE`).

## Author

Tanmay Shukla ([@TanmayShukla05](https://github.com/TanmayShukla05))
