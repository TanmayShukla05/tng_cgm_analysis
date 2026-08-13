"""
Setup script for TNG CGM Analysis package.
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tng_cgm_analysis",
    version="1.0.0",
    author="Tanmay Shukla",
    author_email="your.email@example.com",
    description="Analysis tools for TNG CGM dispersion measures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TanmayShukla05/tng_cgm_analysis",
    py_modules=[
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
    ],
    package_dir={'': '.'},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "h5py>=3.0.0",
        "pandas>=1.3.0",
        "astropy>=5.0.0",
        "healpy>=1.15.0",
    ],
)