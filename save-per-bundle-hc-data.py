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
   - mean: Mean metric value across all white matter in the bundle
   - std: Std metric value across all white matter in the bundle
   - median: Median metric value across all white matter in the bundle

2. {subject}_per_bundle_data.csv - Contains per-bundle metrics with columns:
   - subject: Subject ID
   - bundle: Name of the fiber bundle
   - metric: Metric name
   - session: Session number
   - mean: Mean metric value across all white matter in the bundle
   - std: Std metric value across all white matter in the bundle
   - median: Median metric value across all white matter in the bundle

3. {subject}_per_lesion_data.csv - Contains all safe white matter metrics with columns:
   - subject: Subject ID
   - metric: Metric name
   - session: Session number
   - mean: Mean metric value across all white matter in the bundle
   - std: Std metric value across all white matter in the bundle
   - median: Median metric value across all white matter in the bundle
"""

def load_data(metrics_dirs, wm_mask, bundle_mask, section_labels, subject, scan, bundle, metric, n_labels=20):
    """
    Load and prepare all necessary data for the analysis.
    """
    
    metric_map = load_metric(subject, scan, bundle, metric, metrics_dirs)
    
    data_by_label = {label: [] for label in range(n_labels)}

    for label in range(1, n_labels+1):
        wm_label_mask = (section_labels == label) & (wm_mask > 0)

        metric_in_wm = metric_map[wm_label_mask].flatten()

        metric_in_wm = metric_in_wm[metric_in_wm > 0]

        data_by_label[label-1] = metric_in_wm

    return data_by_label

def save_data_for_safe_white_matter(metrics_dirs, wm_masks_dir, subject, metrics, n_sessions):
    data_rows = []

    for session in range(1, n_sessions+1):
        # Load masks
        wm_mask = nib.load(os.path.join(
            wm_masks_dir, 
            f'{subject}_ses-{session}',
            'safe_wm_mask.nii.gz'
        )).get_fdata().astype(np.uint8)

        for metric in metrics:
            metric_map = load_metric(subject, session, 'IFOF_L', metric, metrics_dirs)
            metric_in_wm = metric_map[wm_mask > 0].flatten()
            metric_in_wm = metric_in_wm[metric_in_wm > 0]

            data_rows.append({
                'subject': subject,
                'metric': metric,
                'session': session,
                'mean': np.mean(metric_in_wm),
                'std': np.std(metric_in_wm),
                'median': np.median(metric_in_wm)
            })

    df = pd.DataFrame(data_rows)
    df.to_csv(f'{subject}_per_lesion_data.csv', index=False)
    print(f"Data saved to {subject}_per_lesion_data.csv")

def save_data(metrics_dirs, wm_masks_dir, bundle_masks_dir, subject, bundles, metrics, n_sessions=5, n_labels=20):
    # Initialize data list
    data_rows = []
    all_data_rows = []

    # Iterate over bundles and metrics
    for session in range(1, n_sessions+1):
        print(f'Processing session: {session}')

        # Load masks
        wm_mask = nib.load(os.path.join(
            wm_masks_dir, 
            f'{subject}_ses-{session}',
            'safe_wm_mask.nii.gz'
        )).get_fdata().astype(np.uint8)

        for bundle in bundles:
            print(f'  * Processing bundle: {bundle}')

            # Load bundle mask
            bundle_mask_filename = os.path.join(
                bundle_masks_dir, 
                f'{subject}_ses-{session}',
                f'{bundle}.nii.gz'
            )
            if os.path.isfile(bundle_mask_filename):
                bundle_mask = nib.load(bundle_mask_filename).get_fdata().astype(np.uint8)
            else:
                bundle_mask = np.zeros(wm_mask.shape, dtype=np.uint8)

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
                    wm_mask,
                    bundle_mask,
                    section_labels,
                    subject, 
                    session, 
                    bundle, 
                    metric,
                    n_labels
                )

                all_vals = []

                for label in range(n_labels):
                    vals = session_data[label]

                    if vals.shape[0] > 0:
                        all_vals.extend(vals)

                    mean = np.mean(vals) if vals.shape[0] > 0 else 0
                    std = np.std(vals) if vals.shape[0] > 0 else 0
                    median = np.median(vals) if vals.shape[0] > 0 else 0

                    # Add to bundle-section data list
                    data_rows.append({
                        'subject': subject,
                        'bundle': bundle,
                        'metric': metric,
                        'session': session,
                        'label': label+1,
                        'mean': mean,
                        'std': std,
                        'median': median
                    })

                all_mean = np.mean(all_vals) if len(all_vals) > 0 else 0
                all_std = np.std(all_vals) if len(all_vals) > 0 else 0
                all_median = np.median(all_vals) if len(all_vals) > 0 else 0

                all_data_rows.append({
                    'subject': subject,
                    'bundle': bundle,
                    'metric': metric,
                    'session': session,
                    'mean': all_mean,
                    'std': all_std,
                    'median': all_median
                })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_rows)
    df.to_csv(f'{subject}_per_bundle_section_data.csv', index=False)
    print(f"Data saved to {subject}_per_bundle_section_data.csv")

    df_all = pd.DataFrame(all_data_rows)
    df_all.to_csv(f'{subject}_per_bundle_data.csv', index=False)
    print(f"Data saved to {subject}_per_bundle_data.csv")

def main():
    parser = argparse.ArgumentParser(description='Save patient per bundle and per bundle-section data into a CSV file.')
    parser.add_argument('subject', help='Subject ID.')
    parser.add_argument('wm_masks_dir', help='Path to white matter masks directory. Safe white matter mask is recommended.')
    parser.add_argument('bundle_masks_dir', help='Path to bundle masks directory.')
    parser.add_argument('--bundles', nargs='+', required=True, help='List of bundle names.')
    parser.add_argument('--metrics', nargs='+', required=True, help='List of metric names.')
    parser.add_argument('--results_tractometry_dir', help='Path to directory with the results_tractometry directory.')
    parser.add_argument('--DTI_dir', help='Path to directory with the DTI metric files.')
    parser.add_argument('--MRDS_dir', help='Path to directory with the MRDS metric files.')
    parser.add_argument('--MTsat_dir', help='Path to directory with the MTsat metric files.')
    parser.add_argument('--FW_dir', help='Path to directory with the FW metric files.')
    parser.add_argument('--n_sessions', type=int, default=5, help='Number of sessions. Default is 5.')
    parser.add_argument('--n_labels', type=int, default=20, help='Number of section labels per bundle. Default is 20.')
    args = parser.parse_args()
    
    metrics_dirs = {
        'results_tractometry': args.results_tractometry_dir,
        'DTI': args.DTI_dir,
        'MRDS': args.MRDS_dir,
        'MTsat': args.MTsat_dir,
        'FW': args.FW_dir
    }

    save_data_for_safe_white_matter(
        metrics_dirs,
        args.wm_masks_dir,
        args.subject,
        args.metrics,
        args.n_sessions
    )

    # Call save_data with the provided arguments
    """
    save_data(
        metrics_dirs,
        args.wm_masks_dir,
        args.bundle_masks_dir,
        args.subject,
        args.bundles,
        args.metrics,
        args.n_sessions,
        args.n_labels
    )
    """

if __name__ == '__main__':
    main()