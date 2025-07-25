#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
import pandas as pd
import argparse
from utils import load_metric

"""
Save bundle, metric, label, session data into a CSV file

Two CSV files are generated:
1. {subject}_per_bundle_section_data.csv - Contains per-bundle-section metrics with columns:
   - subject: Subject ID
   - bundle: Name of the fiber bundle
   - metric: Metric name
   - session: Session number
   - label: Section label number (1-20 by default)
   - lesion_mean: Mean metric value across all lesions in the bundle
   - penumbra_mean: Mean metric value across all penumbra areas in the bundle
   - nawm_mean: Mean metric value across all normal-appearing white matter in the bundle
   - lesion_std: Standard deviation of metric value across all lesions in the bundle
   - penumbra_std: Standard deviation of metric value across all penumbra areas in the bundle
   - nawm_std: Standard deviation of metric value across all normal-appearing white matter in the bundle
   - lesion_median: Median metric value across all lesions in the bundle
   - penumbra_median: Median metric value across all penumbra areas in the bundle
   - nawm_median: Median metric value across all normal-appearing white matter in the bundle

2. {subject}_per_bundle_data.csv - Contains per-bundle metrics with columns:
   - subject: Subject ID
   - bundle: Name of the fiber bundle
   - metric: Metric name
   - session: Session number
   - lesion_mean: Mean metric value across all lesions in the bundle
   - penumbra_mean: Mean metric value across all penumbra areas in the bundle
   - nawm_mean: Mean metric value across all normal-appearing white matter in the bundle
   - lesion_std: Standard deviation of metric value across all lesions in the bundle
   - penumbra_std: Standard deviation of metric value across all penumbra areas in the bundle
   - nawm_std: Standard deviation of metric value across all normal-appearing white matter in the bundle
   - lesion_median: Median metric value across all lesions in the bundle
   - penumbra_median: Median metric value across all penumbra areas in the bundle
   - nawm_median: Median metric value across all normal-appearing white matter in the bundle
"""

def load_data(metrics_dirs, lesion_mask, penumbra_mask, nawm_mask, bundle_mask, section_labels, subject, scan, bundle, metric, n_labels=20):
    """
    Load and prepare all necessary data for the analysis.
    """
    
    metric_map = load_metric(subject, scan, bundle, metric, metrics_dirs)
    
    lesion_by_label = {label: [] for label in range(n_labels)}
    penumbra_by_label = {label: [] for label in range(n_labels)}
    nawm_by_label = {label: [] for label in range(n_labels)}

    for label in range(1, n_labels+1):
        lesion_label_mask = (section_labels == label) & (lesion_mask > 0)
        penumbra_label_mask = (section_labels == label) & (penumbra_mask > 0)
        nawm_label_mask = (section_labels == label) & (nawm_mask > 0)

        metric_in_lesion = metric_map[lesion_label_mask].flatten()
        metric_in_penumbra = metric_map[penumbra_label_mask].flatten()
        metric_in_nawm = metric_map[nawm_label_mask].flatten()

        metric_in_lesion = metric_in_lesion[metric_in_lesion > 0]
        metric_in_penumbra = metric_in_penumbra[metric_in_penumbra > 0]
        metric_in_nawm = metric_in_nawm[metric_in_nawm > 0]

        lesion_by_label[label-1] = metric_in_lesion
        penumbra_by_label[label-1] = metric_in_penumbra
        nawm_by_label[label-1] = metric_in_nawm

    return {
        'lesion': lesion_by_label,
        'penumbra': penumbra_by_label,
        'nawm': nawm_by_label,
    }

def save_data(metrics_dirs, lesion_mask_dir, bundle_masks_dir, subject, bundles, metrics, n_sessions=5, n_labels=20):
    # Initialize data list
    data_rows = []
    all_data_rows = []

    # Iterate over bundles and metrics
    for session in range(1, n_sessions+1):
        print(f'Processing session: {session}')

        # Load masks
        lesion_mask = nib.load(os.path.join(
            lesion_mask_dir, 
            f'{subject}_ses-{session}',
            f'{subject}_ses-{session}__labeled_lesion_mask_in_original_space.nii.gz')).get_fdata().astype(np.uint16)
        penumbra_mask = nib.load(os.path.join(
            lesion_mask_dir, 
            f'{subject}_ses-{session}',
            f'{subject}_ses-{session}__labeled_penumbra_mask_in_original_space.nii.gz')).get_fdata().astype(np.uint16)
        nawm_mask = nib.load(os.path.join(
            lesion_mask_dir, 
            f'{subject}_ses-{session}',
            f'{subject}_ses-{session}__labeled_nawm_mask_in_original_space.nii.gz')).get_fdata().astype(np.uint16)

        for bundle in bundles:
            print(f'  * Processing bundle: {bundle}')

            # Load bundle mask
            bundle_mask_filename = os.path.join(bundle_masks_dir, f'{subject}_ses-{session}__{bundle}.nii.gz')
            if os.path.isfile(bundle_mask_filename):
                bundle_mask = nib.load(bundle_mask_filename).get_fdata().astype(np.uint8)
            else:
                bundle_mask = np.zeros(lesion_mask.shape, dtype=np.uint8)

            # Load bundle section labels
            section_labels = nib.load(os.path.join(
                metrics_dirs['results_tractometry'],
                f'{subject}_ses-{session}', 
                f'Bundle_Label_And_Distance_Maps', 
                f'{subject}_ses-{session}__{bundle}_labels.nii.gz')).get_fdata()

            for metric in metrics:
                print(f'    * Processing metric: {metric}')
                session_data = load_data(
                    metrics_dirs,
                    lesion_mask,
                    penumbra_mask,
                    nawm_mask,
                    bundle_mask,
                    section_labels,
                    subject, 
                    session, 
                    bundle, 
                    metric
                )

                all_lesion_vals = []
                all_penumbra_vals = []
                all_nawm_vals = []

                for label in range(n_labels):
                    lesion_vals = session_data['lesion'][label]
                    penumbra_vals = session_data['penumbra'][label]
                    nawm_vals = session_data['nawm'][label]

                    if lesion_vals.shape[0] > 0:
                        all_lesion_vals.extend(lesion_vals)
                    if penumbra_vals.shape[0] > 0:
                        all_penumbra_vals.extend(penumbra_vals)
                    if nawm_vals.shape[0] > 0:
                        all_nawm_vals.extend(nawm_vals)

                    lesion_mean = np.mean(lesion_vals) if lesion_vals.shape[0] > 0 else 0
                    penumbra_mean = np.mean(penumbra_vals) if penumbra_vals.shape[0] > 0 else 0
                    nawm_mean = np.mean(nawm_vals) if nawm_vals.shape[0] > 0 else 0

                    lesion_std = np.std(lesion_vals) if lesion_vals.shape[0] > 0 else 0
                    penumbra_std = np.std(penumbra_vals) if penumbra_vals.shape[0] > 0 else 0
                    nawm_std = np.std(nawm_vals) if nawm_vals.shape[0] > 0 else 0

                    lesion_median = np.median(lesion_vals) if lesion_vals.shape[0] > 0 else 0
                    penumbra_median = np.median(penumbra_vals) if penumbra_vals.shape[0] > 0 else 0
                    nawm_median = np.median(nawm_vals) if nawm_vals.shape[0] > 0 else 0

                    # Add to bundle-section data list
                    data_rows.append({
                        'subject': subject,
                        'bundle': bundle,
                        'metric': metric,
                        'session': session,
                        'label': label+1,
                        'lesion_mean': lesion_mean,
                        'penumbra_mean': penumbra_mean,
                        'nawm_mean': nawm_mean,
                        'lesion_std': lesion_std,
                        'penumbra_std': penumbra_std,
                        'nawm_std': nawm_std,
                        'lesion_median': lesion_median,
                        'penumbra_median': penumbra_median,
                        'nawm_median': nawm_median
                    })

                all_lesion_mean = np.mean(all_lesion_vals) if len(all_lesion_vals) > 0 else 0
                all_penumbra_mean = np.mean(all_penumbra_vals) if len(all_penumbra_vals) > 0 else 0
                all_nawm_mean = np.mean(all_nawm_vals) if len(all_nawm_vals) > 0 else 0

                all_lesion_std = np.std(all_lesion_vals) if len(all_lesion_vals) > 0 else 0
                all_penumbra_std = np.std(all_penumbra_vals) if len(all_penumbra_vals) > 0 else 0
                all_nawm_std = np.std(all_nawm_vals) if len(all_nawm_vals) > 0 else 0

                all_lesion_median = np.median(all_lesion_vals) if len(all_lesion_vals) > 0 else 0
                all_penumbra_median = np.median(all_penumbra_vals) if len(all_penumbra_vals) > 0 else 0
                all_nawm_median = np.median(all_nawm_vals) if len(all_nawm_vals) > 0 else 0

                all_data_rows.append({
                    'subject': subject,
                    'bundle': bundle,
                    'metric': metric,
                    'session': session,
                    'lesion_mean': all_lesion_mean,
                    'penumbra_mean': all_penumbra_mean,
                    'nawm_mean': all_nawm_mean,
                    'lesion_std': all_lesion_std,
                    'penumbra_std': all_penumbra_std,
                    'nawm_std': all_nawm_std,
                    'lesion_median': all_lesion_median,
                    'penumbra_median': all_penumbra_median,
                    'nawm_median': all_nawm_median
                })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_rows)
    df.to_csv(f'{subject}_per_bundle_section_data.csv', index=False)
    print(f"Data saved to {subject}_per_bundle_section_data.csv")

    df_all = pd.DataFrame(all_data_rows)
    df_all.to_csv(f'{subject}_per_bundle_data.csv', index=False)
    print(f"Data saved to {subject}_per_bundle_data.csv")

def main():
    parser = argparse.ArgumentParser(description='Save patient per bundle and per bundle-section data into a CSV file')
    parser.add_argument('subject', help='Subject ID')
    parser.add_argument('lesion_masks_dir', help='Path to lesion masks directory')
    parser.add_argument('bundle_masks_dir', help='Path to bundle masks directory')
    parser.add_argument('--bundles', nargs='+', required=True, help='List of bundle names')
    parser.add_argument('--metrics', nargs='+', required=True, help='List of metric names')
    parser.add_argument('--results_tractometry_dir', help='Path to directory with the results_tractometry directory')
    parser.add_argument('--DTI_dir', help='Path to directory with the DTI metric files')
    parser.add_argument('--MRDS_dir', help='Path to directory with the MRDS metric files')
    parser.add_argument('--MTsat_dir', help='Path to directory with the MTsat metric files')
    parser.add_argument('--FW_dir', help='Path to directory with the FW metric files')
    parser.add_argument('--n_sessions', type=int, default=5, help='Number of sessions')
    parser.add_argument('--n_labels', type=int, default=20, help='Number of section labels per bundle. Default is 20.')
    args = parser.parse_args()
    
    metrics_dirs = {
        'results_tractometry': args.results_tractometry_dir,
        'DTI': args.DTI_dir,
        'MRDS': args.MRDS_dir,
        'MTsat': args.MTsat_dir,
        'FW': args.FW_dir
    }

    # Call save_data with the provided arguments
    save_data(
        metrics_dirs,
        args.lesion_masks_dir,
        args.bundle_masks_dir,
        args.subject,
        args.bundles,
        args.metrics,
        args.n_sessions,
        args.n_labels
    )

if __name__ == '__main__':
    main()