# Lesion Visualization Tools

A comprehensive Python toolkit for analyzing and visualizing brain lesion data in multiple sclerosis (MS) research. This repository contains tools for processing diffusion tensor imaging (DTI), magnetization transfer (MT), and other advanced MRI metrics to study lesion characteristics and their evolution over time.

## Overview

This toolkit is designed for neuroimaging researchers working with MS patient data. It provides automated workflows for:

- **Lesion-specific Analysis**: Processing and analyzing individual lesions and their surrounding tissue
- **Bundle-specific Analysis**: Examining white matter tracts and their intersection with lesions
- **Bundle-section-specific Analysis**: Examining subsections of white matter tracts and their intersection with lesions
- **Longitudinal Studies**: Tracking lesion evolution across multiple imaging sessions
- **Multi-metric support**: Several metrics can be added simultaneously

## Scripts

### Data Processing
- `save-per-lesion-patient-data.py`: Extracts per-lesion metrics and saves to CSV
- `save-per-bundle-patient-data.py`: Processes bundle-specific lesion data
- `save-per-bundle-hc-data.py`: Extracts healthy control data for comparison

### Visualization
- `plot-new-lesion-boxplots.py`: Creates longitudinal boxplots comparing lesion, penumbra, and NAWM metrics
- `plot-subject-data.py`: Generates individual subject longitudinal grid of plots
- `plot-cohort-data.py`: Creates cohort-level longitudinal grid of plots

### Utilities
- `utils.py`: Common functions for metric loading, masking, and data processing

## Supported Metrics

### Diffusion Metrics
- Fractional Anisotropy (FA)
- Mean Diffusivity (MD)
- Axial Diffusivity (AD)
- Radial Diffusivity (RD)

### Magnetization Transfer Metrics
- MT Saturation (MTsat)
- Magnetization Transfer Ratio (MTR)
- Inhomogeneous MT Saturation (ihMTsat)
- Inhomogeneous MTR (ihMTR)

### Multi-tensor Metrics
- Fixel-based AD (fixel-ad)
- Fixel-based RD (fixel-rd)
- Fixel-based MD (fixel-md)
- Fixel-based FA (fixel-fa)
- MRDS Isotropic Volume Fraction (mrds-isovf)

### Other Metrics
- TODI Number of Components (todi-nufo)
- Free Water (FW)

## Requirements

- Python 3.7+
- NumPy
- NiBabel
- Matplotlib
- Pandas
- SciPy
- Scikit-learn

## Usage


python .\pipeline\plot-new-lesion-boxplots.py 
    sub-004-ms # patient tag
    ./lesion_masks # directory with the lesion_masks
    ./bundle_masks # directory with the bundle_masks
    --metrics fixel-fa fixel-rd mrds-isovf MTsat # target metrics
    --results_tractometry_dir ./results_tractometry_imk # directory with tractometry_results
    --MRDS_dir ./MRDS  # directory with MRDS results
    --MTsat_dir ./MTsat # directory with MT metric maps (only for MTsat and MTR)
    --DTI_dir ./DTI # directory with DTI metric maps (only required for FA, RD, AD and MD)
    --FW_dir ./FW # directory with FW maps (only for FW)
    --target_session 2 # target session of the new lesions

The scripts are designed to work with organized neuroimaging data directories containing:
- Lesion masks and labels
- White matter masks
- Bundle-specific masks
- Metric maps (DTI, MT, etc.)

Each script includes command-line arguments for specifying data directories and analysis parameters.

