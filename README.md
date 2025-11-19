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
- `plot-tract-profiles.py`: Plot tract profiles comparing a patient with a cohort as in https://doi.org/10.3389/fnins.2024.1467786.
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

- Python 3.10.x

## Usage

### Tract Profiles

The script that plot tract profiles require `tractometry_results` directory to plot. For example, to plot a patient tract profile for a specific bundle and metric:
```bash
python plot-tract-profiles.py 
    --patient sub-004-ms_ses-1 
    --patient_tractometry_results_dir ms_6months/results_tractometry_imk
    --metrics fixel-RD
    --bundles IFOF_L 
    -add_legend
```

To include a cohort for comparison:
```bash
python plot-tract-profiles.py 
    --patient sub-004-ms_ses-1 
    --patient_tractometry_results_dir ms_6months/results_tractometry_imk
    --cohort_tractometry_results_dir myelo_inferno_imk/results_tractometry
    --metrics fixel-RD
    --bundles IFOF_L 
    -add_legend
```

To plot tract profile grid for several metrics and bundles:
```bash
python plot-tract-profiles.py 
    --patient sub-004-ms_ses-1 
    --patient_tractometry_results_dir ms_6months/results_tractometry_imk
    --cohort_tractometry_results_dir myelo_inferno_imk/results_tractometry
    --metrics fixel-RD fixel-FA MTsat
    --bundles IFOF_L AF_R
    -add_legend
```

### CSV File generation

The scripts that plot longitudinal grids require a .CSV file with the data to plot. For example, the .CSV to plot per-bundle longitudinal grid for subject should have the following columns
```bash
subject,bundle,metric,session,lesion_mean,penumbra_mean,nawm_mean,lesion_std,penumbra_std,nawm_std,lesion_median,penumbra_median,nawm_median
```
See the `data/sub-004-ms_per_bundle_data.csv` file to see an example.

To generate the a .CSV with the per-bundle and per-bundle-section data for a single patient, use `save-per-bundle-patient-data.py`. To generate the .CSV of the whole HC cohort to be added to as reference in the plots use the `save-per-bundle-hc-data.py`. To generate the .CSV with the per-lesion data for a single patient, use the ``save-per-lesion-patient-data.py``

Example of `save-per-bundle-patient-data.py` to generate the `sub-004-ms_per_bundle_data.csv` and `sub-004-ms_per_bundle_section_data.csv` files
```bash
python .\pipeline\save-per-bundle-patient-data.py 
    sub-004-ms # patient tag 
    .lesion_masks # directory with lesion masks
    .bundle_masks # directory with bundle masks
    --bundles AF_L AF_R CC_1 CC_2a CC_2b CC_3 CC_4 CC_5 CC_6 CC_7 CG_L CG_R CR_L CR_R CST_L CST_R ICP_L ICP_R IFOF_L IFOF_R ILF_L ILF_R MCP OR_L OR_R SLF_1_L SLF_1_R SLF_2_L SLF_2_R SLF_3_L SLF_3_R UF_L UF_R # target bundles
    --metrics fixel-ad fixel-rd fixel-fa fixel-md ad rd fa md MTsat MTR ihMTsat ihMTR fw mrds-isovf todi-nufo # target metrics
    --results_tractometry_dir ./results_tractometry_imk # directory with tractometry results (only for fixel-based MRDS metrics)
    --DTI_dir ./DTI # directory with DTI maps (only for DTI metrics)
    --MRDS_dir ./MRDS # directory with MRDS maps (only for mrds-isovf)
    --MTsat_dir ./MTsat # directory with MT maps (only for MTsat and MTR)
    --FW_dir ./FW # directory with FW maps (only for FW)
```

For the HC cohort the use is similar but passing a list of subject tags instead of a single patient tag, and passing a wm_masks directory instead of a lesion_masks directory
```bash
python save-per-bundle-hc-data.py 
    sub-003-hc sub-004-hc sub-015-hc
    ./wm_masks
    ./bundle_masks
    --bundles IFOF_L 
    --metrics ad rd md fa fixel-ad fixel-rd fixel-md fixel-fa mrds-isovf fw MTsat 
    --results_tractometry_dir ./results_tractometry
    --DTI_dir ./DTI
    --MRDS_dir ./MRDS
    --MTsat_dir ./MTsat
    --FW_dir ./FW
```

### Longitudinal Visualization

To plot the per-bundle longitudinal grid for a single patient use the `plot-subject-data.py`

Example of `plot-subject-data.py` to plot per-bundle longitudinal grid
```bash
python plot-subject-data.py 
    --patient sub-004-ms # patient tag
    --patient_csv_dir ms_6months/results_csv # directory with patients CSV files
    --metrics fixel-fa fixel-rd mrds-isovf MTsat # target metrics
    --bundles IFOF_L AF_L CC_7 CST_L # target bundles
    --hc_subject sub-003-hc # HC subject tags (optional)
    --hc_csv_dir myelo_inferno/results_csv # directory with HC subjects CSV files
```

To plot the per-bundle longitudinal grid for a MS cohort, use the `plot-cohort-data.py`
```bash
python plot-cohort-data.py 
    --patients sub-006-ms sub-008-ms sub-010-ms sub-018-ms # patient tags list
    --patient_csv_dir ms_6months/results_csv # directory with patients CSV files
    --bundle IFOF_L # target bundle
    --metrics fixel-fa fixel-rd mrds-isovf MTsat # target metrics
    --hc_subjects sub-003-hc # HC subjects tags (optional)
    --hc_csv_dir myelo_inferno/results_csv # directory with HC subjects CSV files
```

To plot the per-bundle-section longitudinal grid for a single patient use the `plot-subject-data.py`

Example of plot-subject-data.py to plot per-bundle-section longitudinal grid
```bash
python plot-subject-data.py 
    --patient sub-004-ms # patient tag
    --patient_csv_dir ms_6months/results_csv # directory with patients CSV files
    --metrics fixel-fa fixel-rd mrds-isovf MTsat # target metrics
    --bundle IFOF_L # target bundle
    --bundle_sections 4 5 6 7 # bundle section labels
    --hc_subject sub-003-hc # HC subject tags (optional)
    --hc_csv_dir myelo_inferno/results_csv # directory with HC subjects CSV files
```

To plot the per-lesion longitudinal grid for a single patient use the `plot-subject-data.py`

Example of plot-subject-data.py to plot per-lesion longitudinal grid
```bash
python plot-subject-data.py 
    --patient sub-004-ms # patient tag
    --patient_csv_dir ms_6months/results_csv # directory with patients CSV files
    --metrics fixel-fa fixel-rd mrds-isovf MTsat # target metrics
    --lesions 197 36 34 218 # lesion labels
    --hc_subject sub-003-hc # HC subject tags (optional)
    --hc_csv_dir myelo_inferno/results_csv # directory with HC subjects CSV files
```

To plot the new-lesion longitudinal grid for a single patient use the `plot-subject-data.py`

Example of plot-new-lesion-boxplots.py to plot new-lesion longitudinal grid
```bash
python plot-new-lesion-boxplots.py
    sub-004-ms # patient tag
    ./lesion_masks # directory with the lesion_masks
    ./bundle_masks # directory with the bundle_masks
    --metrics fixel-fa fixel-rd mrds-isovf MTsat # target metrics
    --results_tractometry_dir ./results_tractometry_imk # directory with tractometry_results
    --MRDS_dir ./MRDS # directory with MRDS results
    --MTsat_dir ./MTsat # directory with MT metric maps (only for MTsat and MTR)
    --DTI_dir ./DTI # directory with DTI metric maps (only required for FA, RD, AD and MD)
    --FW_dir ./FW # directory with FW maps (only for FW)
    --target_session 2 # target session of the new lesions
```

The scripts are designed to work with organized neuroimaging data directories (results_tractometry, MRDS, MTsat, DTI, and FW). These folder have to organize the data in the following example format

```bash
[DTI]
├── sub-001-ms_ses-1
│   ├── fa.nii.gz
│   ├── md.nii.gz
│   ├── rd.nii.gz
│   └── ad.nii.gz
├── sub-001-ms_ses-2
│   ├── fa.nii.gz
│   ├── md.nii.gz
│   ├── rd.nii.gz
│   └── ad.nii.gz
├── sub-002-ms_ses-1
│   ├── fa.nii.gz
│   ├── md.nii.gz
│   ├── rd.nii.gz
│   └── ad.nii.gz
...
```
