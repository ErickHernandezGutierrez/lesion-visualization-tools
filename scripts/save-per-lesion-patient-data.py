#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from utils import find_lesion_bundles_intersections

"""
Save bundle, metric, label, session data into a CSV file

Two CSV files are generated:
1. {subject}_per_lesion_data.csv - Contains per-label metrics with columns:
   - subject: Subject ID
   - metric: Metric name
   - session: Session number
   - label: Lesion label number
   - lesion_mean: Mean metric value in lesion areas
   - penumbra_mean: Mean metric value in penumbra areas
   - nawm_mean: Mean metric value in normal-appearing white matter
   - lesion_std: Standard deviation of metric value in lesion areas
   - penumbra_std: Standard deviation of metric value in penumbra areas
   - nawm_std: Standard deviation of metric value in normal-appearing white matter
"""

def load_metric(subject, session, bundle, metric, metrics_dirs):
    """
    Load the metric map for a given bundle and metric.
    """

    if metric in ['fixel-ad', 'fixel-rd', 'fixel-fa', 'fixel-md']:
        # Rename metric to match the format of the metric files
        renamed_metric = metric.replace('fixel-', 'ic_')
        
        # Load metric
        metric_map = nib.load(os.path.join(
            metrics_dirs['results_tractometry'],
            f'{subject}_ses-{session}',
            f'Fixel_MRDS',
            f'{bundle}_{renamed_metric}_metric.nii.gz')).get_fdata()
    elif metric in ['ad', 'rd', 'fa', 'md']:
        metric_map = nib.load(os.path.join(
            metrics_dirs['DTI'],
            f'{subject}_ses-{session}',
            f'{metric}.nii.gz')).get_fdata()
    elif metric in ['MTsat', 'MTR', 'ihMTsat', 'ihMTR']:
        metric_map = nib.load(os.path.join(
            metrics_dirs['MTsat'],
            f'{subject}_ses-{session}',
            f'{metric}.nii.gz')).get_fdata()
    elif metric == 'fw':
        metric_map = nib.load(os.path.join(
            metrics_dirs['FW'],
            f'{subject}_ses-{session}',
            f'{metric}.nii.gz')).get_fdata()
    elif metric == 'mrds-isovf':
        metric_map = nib.load(os.path.join(
            metrics_dirs['MRDS'],
            f'{subject}_ses-{session}',
            f'results_MRDS_Diff_TODI_ISOTROPIC.nii.gz')).get_fdata()[:,:,:,0]
    elif metric == 'todi-nufo':
        metric_map = nib.load(os.path.join(
            metrics_dirs['MRDS'],
            f'{subject}_ses-{session}',
            f'results_MRDS_Diff_TODI_NUM_COMP.nii.gz')).get_fdata()
    else:
        raise ValueError(f'Metric {metric} not supported')

    return metric_map

def load_data(metrics_dirs, lesion_mask, penumbra_mask, nawm_mask, lesion_labels, intersections, subject, session, metric):
    """
    Load and prepare all necessary data for the analysis.
    """
    
    lesion_values_by_label = {label: [] for label in lesion_labels}
    penumbra_values_by_label = {label: [] for label in lesion_labels}
    nawm_values_by_label = {label: [] for label in lesion_labels}

    for label in lesion_labels:
        lesion_label_mask = (lesion_mask == label)
        penumbra_label_mask = (penumbra_mask == label)
        nawm_label_mask = (nawm_mask == label)

        bundles = intersections[label] if metric in ['fixel-ad', 'fixel-rd', 'fixel-fa', 'fixel-md'] else ['_']

        for bundle in bundles:
            metric_map = load_metric(subject, session, bundle, metric, metrics_dirs)

            # Mask the metric map for each tissue type
            metric_in_lesion = metric_map[lesion_label_mask].flatten()
            metric_in_penumbra = metric_map[penumbra_label_mask].flatten()
            metric_in_nawm = metric_map[nawm_label_mask].flatten()

            # Remove 0 values
            metric_in_lesion = metric_in_lesion[metric_in_lesion > 0]
            metric_in_penumbra = metric_in_penumbra[metric_in_penumbra > 0]
            metric_in_nawm = metric_in_nawm[metric_in_nawm > 0]

            # Add to data list
            lesion_values_by_label[label].extend(metric_in_lesion)
            penumbra_values_by_label[label].extend(metric_in_penumbra)
            nawm_values_by_label[label].extend(metric_in_nawm)

        lesion_values_by_label[label] = np.array(lesion_values_by_label[label])
        penumbra_values_by_label[label] = np.array(penumbra_values_by_label[label])
        nawm_values_by_label[label] = np.array(nawm_values_by_label[label])

    return {
        'lesion': lesion_values_by_label, # metric values for each lesion label
        'penumbra': penumbra_values_by_label, # metric values for each penumbra of lesion label
        'nawm': nawm_values_by_label, # metric values for each nawm of lesion label
    }

def save_data(subject, metrics, metrics_dirs, lesion_masks_dir, bundle_masks_dir, n_sessions=5):
    # Initialize data list
    data_rows = []

    # Iterate over sessions
    for session in range(1, n_sessions+1):
        print(f'Processing session: {session}')

        # Load masks
        lesion_mask = nib.load(os.path.join(lesion_masks_dir, f'{subject}_ses-{session}__labeled_lesion_mask_in_original_space.nii.gz')).get_fdata().astype(np.uint16)
        penumbra_mask = nib.load(os.path.join(lesion_masks_dir, f'{subject}_ses-{session}__labeled_penumbra_mask_in_original_space.nii.gz')).get_fdata().astype(np.uint16)
        nawm_mask = nib.load(os.path.join(lesion_masks_dir, f'{subject}_ses-{session}__labeled_nawm_mask_in_original_space.nii.gz')).get_fdata().astype(np.uint16)

        # Find lesion labels
        lesion_labels = np.unique(lesion_mask[lesion_mask > 0])

        # Find bundles intersected by each lesion label
        print(f'  * Finding bundles intersected by each lesion label')
        intersections = find_lesion_bundles_intersections(subject+f'_ses-{session}', bundle_masks_dir, lesion_mask)

        for metric in metrics:
            print(f'  * Processing metric: {metric}')

            session_data = load_data(
                metrics_dirs,
                lesion_mask,
                penumbra_mask,
                nawm_mask,
                lesion_labels,
                intersections,
                subject,
                session,
                metric
            )

            # Add each lesion label data
            for label in lesion_labels:
                lesion_vals = session_data['lesion'][label]
                penumbra_vals = session_data['penumbra'][label]
                nawm_vals = session_data['nawm'][label]

                lesion_mean = np.mean(lesion_vals) if lesion_vals.shape[0] > 0 else 0
                penumbra_mean = np.mean(penumbra_vals) if penumbra_vals.shape[0] > 0 else 0
                nawm_mean = np.mean(nawm_vals) if nawm_vals.shape[0] > 0 else 0

                lesion_std = np.std(lesion_vals) if lesion_vals.shape[0] > 0 else 0
                penumbra_std = np.std(penumbra_vals) if penumbra_vals.shape[0] > 0 else 0
                nawm_std = np.std(nawm_vals) if nawm_vals.shape[0] > 0 else 0

                lesion_median = np.median(lesion_vals) if lesion_vals.shape[0] > 0 else 0
                penumbra_median = np.median(penumbra_vals) if penumbra_vals.shape[0] > 0 else 0
                nawm_median = np.median(nawm_vals) if nawm_vals.shape[0] > 0 else 0

                # Add to data list
                data_rows.append({
                    'subject': subject,
                    'metric': metric,
                    'session': session,
                    'label': label,
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

    # Create DataFrame and save to CSV
    out_filename = f'{subject}_per_lesion_data.csv'
    df = pd.DataFrame(data_rows)
    df.to_csv(out_filename, index=False)
    print(f"Data saved to {out_filename}")

def main():
    parser = argparse.ArgumentParser(description='Save patient per lesion data across sessions into a CSV file')
    parser.add_argument('subject', help='Subject ID. e.g. "sub-004-ms"')
    parser.add_argument('lesion_masks_dir', help='Path to lesion masks directory')
    parser.add_argument('bundle_masks_dir', help='Path to bundle masks directory')
    parser.add_argument('--metrics', nargs='+', required=True, help='List of metric names')
    parser.add_argument('--results_tractometry_dir', help='Path to directory with the results_tractometry directory')
    parser.add_argument('--DTI_dir', help='Path to directory with the DTI metric files')
    parser.add_argument('--MRDS_dir', help='Path to directory with the MRDS metric files')
    parser.add_argument('--MTsat_dir', help='Path to directory with the MTsat metric files')
    parser.add_argument('--FW_dir', help='Path to directory with the FW metric files')
    parser.add_argument('--n_sessions', type=int, default=5, help='Number of sessions. Default is 5.')
    args = parser.parse_args()

    metrics_dirs = {
        'results_tractometry': args.results_tractometry_dir,
        'DTI': args.DTI_dir,
        'MRDS': args.MRDS_dir,
        'MTsat': args.MTsat_dir,
        'FW': args.FW_dir
    }
    
    save_data(
        args.subject,
        args.metrics,
        metrics_dirs,
        args.lesion_masks_dir,
        args.bundle_masks_dir,
        args.n_sessions
    )

if __name__ == '__main__':
    main()
