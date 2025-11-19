import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from utils import find_lesion_bundles_intersections
from utils import load_metric
import pickle
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

patient_tags = {
    'sub-006-ms': 'Patient-A',
    'sub-008-ms': 'Patient-B',
    'sub-010-ms': 'Patient-C',
    'sub-018-ms': 'Patient-D',
    'sub-004-ms': 'Patient-E',
    'sub-002-ms': 'Patient-F',
    'sub-020-ms': 'Patient-G',
    'sub-005-ms': 'Patient-H',
}

lesion_tags = {
    169: 'F1',
    175: 'F2',
    182: 'F3',
    50: 'G1',
    47: 'G2',
    48: 'G3',
    9: 'H3',
    10: 'H1',
    78: 'H2',
    197: 'E1',
    36: 'E2',
    218: 'E3',
    34: 'E4',
    372: 'E5',
    389: 'E6',
    426: 'E7',
}

def load_hc_data(metrics):
    hc_subjects = [f'sub-003-hc_ses-{i}' for i in range(2, 6)]
    wm_masks_dir = 'D:\\SCIL\\myelo_inferno_imk\\wm_masks'
    hc_metrics_dir = {
        'results_tractometry': 'D:\\SCIL\\myelo_inferno_imk\\results_tractometry',
        'DTI': 'D:\\SCIL\\myelo_inferno_imk\\DTI',
        'MTsat': 'D:\\SCIL\\myelo_inferno_imk\\MTsat',
        'FW': 'D:\\SCIL\\myelo_inferno_imk\\FW',
        'MRDS': 'D:\\SCIL\\myelo_inferno_imk\\MRDS',
    }
    
    hc_data = {
        metric: [] for metric in metrics
    }

    for subject in hc_subjects:
        wm_mask = nib.load(os.path.join(
            wm_masks_dir, 
            subject,
            'safe_wm_mask.nii.gz'
        )).get_fdata().astype(np.uint8)

        for metric in metrics:
            if metric in ['fixel-ad', 'fixel-rd', 'fixel-fa', 'fixel-md']:
                renamed_metric = metric.replace('fixel-', '').upper()
                metric_map = nib.load(os.path.join(
                    hc_metrics_dir['MRDS'],
                    subject,
                    f'results_MRDS_Diff_TODI_{renamed_metric}.nii.gz'
                )).get_fdata()
            elif metric in ['ad', 'rd', 'fa', 'md']:
                metric_map = nib.load(os.path.join(
                    hc_metrics_dir['DTI'],
                    subject,
                    f'{metric}.nii.gz'
                )).get_fdata()
            elif metric in ['MTsat', 'MTR', 'ihMTsat', 'ihMTR']:
                metric_map = nib.load(os.path.join(
                    hc_metrics_dir['MTsat'],
                    subject,
                    f'{metric}.nii.gz'
                )).get_fdata()
            elif metric == 'fw':
                metric_map = nib.load(os.path.join(
                    hc_metrics_dir['FW'],
                    subject,
                    f'{metric}.nii.gz'
                )).get_fdata()
            elif metric == 'mrds-isovf':
                metric_map = nib.load(os.path.join(
                    hc_metrics_dir['MRDS'],
                    subject,
                    f'results_MRDS_Diff_TODI_ISOTROPIC.nii.gz'
                )).get_fdata()[:,:,:,0]
            else:
                raise ValueError(f'Metric {metric} not supported')

            # Mask metric map with WM mask
            if metric in ['fixel-ad', 'fixel-rd', 'fixel-fa', 'fixel-md']:
                metric_map = metric_map * wm_mask[..., np.newaxis]
            else:
                metric_map = metric_map * wm_mask

            # Flat and remove 0 values
            metric_map = metric_map.flatten()
            metric_map = metric_map[metric_map > 0]
            hc_data[metric].extend(metric_map)

    return hc_data
    
def load_session_data(metrics_dirs, lesion_mask, penumbra_mask, nawm_mask, lesion_labels, intersections, subject, session, metric):
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

        bundles = intersections[label] if metric in ['fixel-ad', 'fixel-rd', 'fixel-fa', 'fixel-md'] else ['-']
        #print(f'  * Bundles for label {label}, metric {metric}, session {session}: {bundles}')

        for bundle in bundles:
            metric_map = load_metric(subject, session, bundle, metric, metrics_dirs)

            if metric_map is None:
                print(f'  * Metric map for {subject}_ses-{session}, {bundle}, {metric} not found')
                continue

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
        #print(f'  * Lesion values for label {label}, metric {metric}, session {session} shape: {lesion_values_by_label[label].shape}')

    return {
        'lesion': lesion_values_by_label, # metric values for each lesion label
        'penumbra': penumbra_values_by_label, # metric values for each penumbra of lesion label
        'nawm': nawm_values_by_label, # metric values for each nawm of lesion label
    }

def remove_outliers_iqr(data, iqr_multiplier=1.5):
    """
    Remove outliers from data using the IQR method.
    
    Parameters:
    -----------
    data : array-like
        Input data
    iqr_multiplier : float, optional
        Multiplier for IQR to determine outlier boundaries (default: 1.5)
        
    Returns:
    --------
    array
        Data with outliers removed
    """
    if len(data) == 0:
        return data
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def subsample_hc_data(hc_data, sample_size=1000):
    """
    Subsample HC data for each metric.
    
    Parameters:
    -----------
    hc_data : dict
        Dictionary containing HC values for each metric
        Structure: hc_data[metric] = values
    sample_size : int, optional
        Number of samples to subsample (default: 1000)
        
    Returns:
    --------
    dict
        Subsampled HC data for each metric
    """
    subsampled_data = {}

    for metric in hc_data:
        data_size = min(sample_size, len(hc_data[metric]))
        subsampled_data[metric] = np.random.choice(hc_data[metric], data_size, replace=False)

    return subsampled_data

def subsample_nawm_data(nawm_data, sample_size=1000):
    """
    Subsample NAWM data for each metric and lesion label.
    
    Parameters:
    -----------
    nawm_data : dict
        Dictionary containing NAWM values for each metric and lesion label
        Structure: nawm_data[metric][lesion_label] = values
    sample_size : int, optional
        Number of samples to subsample (default: 1000)
        
    Returns:
    --------
    dict
        Subsampled NAWM data
    """
    subsampled_data = {}
    
    for metric in nawm_data:
        subsampled_data[metric] = {}
        for label in nawm_data[metric]:
            data_size = min(sample_size, len(nawm_data[metric][label]))
            subsampled_data[metric][label] = np.random.choice(nawm_data[metric][label], data_size, replace=False)
            
    return subsampled_data

def remove_hc_outliers(hc_data):
    """
    Remove outliers from HC data for each metric.
    
    Parameters:
    -----------
    hc_data : dict
        Dictionary containing HC values for each metric
        Structure: hc_data[metric] = values
        
    Returns:
    --------
    dict
        HC data with outliers removed
    """
    cleaned_data = {}
    
    for metric in hc_data:
        cleaned_data[metric] = remove_outliers_iqr(hc_data[metric])
        
    return cleaned_data

def remove_nawm_outliers(nawm_data):
    """
    Remove outliers from NAWM data for each metric and lesion label.
    
    Parameters:
    -----------
    nawm_data : dict
        Dictionary containing NAWM values for each metric and lesion label
        Structure: nawm_data[metric][lesion_label] = values
        
    Returns:
    --------
    dict
        NAWM data with outliers removed
    """
    cleaned_data = {}
    
    for metric in nawm_data:
        cleaned_data[metric] = {}
        for label in nawm_data[metric]:
            cleaned_data[metric][label] = remove_outliers_iqr(nawm_data[metric][label])
            
    return cleaned_data

def perform_anova_tests(boxplot_data, nawm_data, metrics, new_lesions, n_sessions):
    """
    Perform one-way ANOVA tests between NAWM and each session's lesion values separately.
    
    Parameters:
    -----------
    boxplot_data : dict
        Dictionary containing lesion values for each metric, lesion label, and session
    nawm_data : dict
        Dictionary containing NAWM values for each metric and lesion label
    metrics : list
        List of metrics to analyze
    new_lesions : list
        List of lesion labels to analyze
    n_sessions : int
        Number of sessions
        
    Returns:
    --------
    dict
        Dictionary containing ANOVA results for each metric, lesion label, and session comparison
        Structure: results[metric][lesion_label][session] = {
            'f_statistic': float,
            'p_value': float,
            'groups': list of group names that were compared
        }
    """
    results = {}
    
    for metric in metrics:
        results[metric] = {}
        
        for lesion_label in new_lesions:
            results[metric][lesion_label] = {}
            
            # Get NAWM data if available
            if len(nawm_data[metric][lesion_label]) > 0:
                nawm_values = nawm_data[metric][lesion_label]
                if metric in ['rd', 'fixel-rd']:
                    nawm_values = nawm_values * 1e3
                
                # Compare NAWM with each session separately
                for session in range(1, n_sessions + 1):
                    if len(boxplot_data[metric][lesion_label][session]) > 0:
                        lesion_values = boxplot_data[metric][lesion_label][session]
                        if metric in ['rd', 'fixel-rd']:
                            lesion_values = lesion_values * 1e3
                        
                        # Perform ANOVA between NAWM and this session
                        f_statistic, p_value = stats.f_oneway(nawm_values, lesion_values)
                        
                        results[metric][lesion_label][session] = {
                            'f_statistic': f_statistic,
                            'p_value': p_value,
                            'groups': ['NAWM', f'ses-{session}']
                        }
                    else:
                        results[metric][lesion_label][session] = None
            else:
                # No NAWM data available for comparison
                for session in range(1, n_sessions + 1):
                    results[metric][lesion_label][session] = None
    
    return results

def save_data(subject, metrics, metrics_dirs, lesion_masks_dir, bundle_masks_dir, n_sessions=5, target_session=2):
    new_lesions_per_patient = {
        'sub-002-ms': [169, 175, 182],
        'sub-004-ms': [372, 389, 426],
        'sub-005-ms': [9, 10, 78],
        'sub-017-ms': [55, 62, 61],
        'sub-020-ms': [50, 47, 48],
        'sub-022-ms': [507, 514, 517]
    }
    
    new_lesions = new_lesions_per_patient[subject]
    
    # Check if boxplot data already exists
    boxplot_data_filename = f'{subject}_boxplot_data.pkl'
    nawm_data_filename = f'{subject}_nawm_data.pkl'
    if os.path.exists(boxplot_data_filename) and os.path.exists(nawm_data_filename):
        print(f'Loading boxplot data from: {boxplot_data_filename}')
        with open(boxplot_data_filename, 'rb') as f:
            boxplot_data = pickle.load(f)
        
        print(f'Loading nawm data from: {nawm_data_filename}')
        with open(nawm_data_filename, 'rb') as f:
            nawm_data = pickle.load(f)
    else:
        # Initialize data list
        #data_rows = []
        
        # Structure: boxplot_data[metric][lesion_label][session] = values
        boxplot_data = {
            metric: {
                label: {session: [] for session in range(1, n_sessions+1)} 
                for label in new_lesions
            } for metric in metrics
        }
        
        # Structure: nawm_data[metric][lesion_label] = values
        nawm_data = {
            metric: {label: [] for label in new_lesions} 
            for metric in metrics
        }

        # Iterate over sessions
        for session in range(1, n_sessions+1):
            print(f'Processing session: {session}')

            # Load masks
            if session != target_session:
                lesion_mask_filename = f'{subject}_ses-{target_session}__labeled_lesion_mask_in_ses-{session}_space.nii.gz'
                penumbra_mask_filename = f'{subject}_ses-{target_session}__labeled_penumbra_mask_in_ses-{session}_space.nii.gz'
                nawm_mask_filename = f'{subject}_ses-{target_session}__labeled_nawm_mask_in_ses-{session}_space.nii.gz'
            else:
                lesion_mask_filename = f'{subject}_ses-{session}__labeled_lesion_mask_in_original_space.nii.gz'
                penumbra_mask_filename = f'{subject}_ses-{session}__labeled_penumbra_mask_in_original_space.nii.gz'
                nawm_mask_filename = f'{subject}_ses-{session}__labeled_nawm_mask_in_original_space.nii.gz'
            lesion_mask = nib.load(os.path.join(
                lesion_masks_dir, 
                f'{subject}_ses-{target_session}',
                lesion_mask_filename)).get_fdata().astype(np.uint16)
            penumbra_mask = nib.load(os.path.join(
                lesion_masks_dir, 
                f'{subject}_ses-{target_session}',
                penumbra_mask_filename)).get_fdata().astype(np.uint16)
            nawm_mask = nib.load(os.path.join(
                lesion_masks_dir, 
                f'{subject}_ses-{target_session}',
                nawm_mask_filename)).get_fdata().astype(np.uint16)

            # Find lesion labels
            lesion_labels = np.unique(lesion_mask[lesion_mask > 0])

            # Find bundles intersected by each lesion label
            print(f'  * Finding bundles intersected by each lesion label')
            intersections = find_lesion_bundles_intersections(subject+f'_ses-{session}', bundle_masks_dir+f'/{subject}_ses-{session}', lesion_mask)

            for metric in metrics:
                print(f'  * Processing metric: {metric}')

                session_data = load_session_data(
                    metrics_dirs,
                    lesion_mask,
                    penumbra_mask,
                    nawm_mask,
                    new_lesions,  # Only process our target lesions
                    intersections,
                    subject,
                    session,
                    metric
                )

                # Add each lesion label data
                for label in new_lesions:
                    lesion_vals = session_data['lesion'][label]
                    penumbra_vals = session_data['penumbra'][label]
                    nawm_vals = session_data['nawm'][label]

                    #print(f'  * Lesion values for label {label}, metric {metric}, session {session} shape: {lesion_vals.shape}')
                    
                    # Store lesion values for boxplot
                    if lesion_vals.shape[0] > 0:
                        boxplot_data[metric][label][session] = lesion_vals
                    
                    # Store NAWM values for all sessions
                    if nawm_vals.shape[0] > 0:
                        nawm_data[metric][label].extend(nawm_vals)

        # Save boxplot data to file
        with open(boxplot_data_filename, 'wb') as f:
            pickle.dump(boxplot_data, f)
        print(f'Boxplot data saved to: {boxplot_data_filename}')

        # Save NAWM data to file
        with open(nawm_data_filename, 'wb') as f:
            pickle.dump(nawm_data, f)
        print(f'NAWM data saved to: {nawm_data_filename}')

    hc_data = load_hc_data(metrics)

    # Clean HC data
    hc_data = subsample_hc_data(hc_data)
    hc_data = remove_hc_outliers(hc_data)

    # Clean NAWM data
    nawm_data = subsample_nawm_data(nawm_data)
    nawm_data = remove_nawm_outliers(nawm_data)
    
    # Create boxplots
    plot_boxplots(subject, boxplot_data, nawm_data, hc_data, metrics, new_lesions, n_sessions, target_session)
    
    # Perform ANOVA tests
    anova_results = perform_anova_tests(boxplot_data, nawm_data, metrics, new_lesions, n_sessions)
    
    # Print ANOVA results
    print("\nANOVA Test Results:")
    print("===================")
    for metric in metrics:
        print(f"\nMetric: {metric}")
        for lesion_label in new_lesions:
            print(f"\nLesion-{lesion_tags[lesion_label]}:")
            for session in range(1, n_sessions + 1):
                result = anova_results[metric][lesion_label][session]
                if result is not None:
                    print(f"\n  Session {session} vs NAWM:")
                    print(f"    F-statistic: {result['f_statistic']:.4f}")
                    print(f"    p-value: {result['p_value']:.4e}")
                    if result['p_value'] < 0.05:
                        print("    * Significant difference found (p < 0.05)")
                else:
                    print(f"\n  Session {session} vs NAWM: Insufficient data for comparison")

def plot_boxplots(subject, boxplot_data, nawm_data, hc_data, metrics, new_lesions, n_sessions, target_session):
    """
    Create boxplots for lesion values across sessions and metrics.
    
    Parameters:
    -----------
    subject : str
        Subject ID
    boxplot_data : dict
        Dictionary containing lesion values for each metric, lesion label, and session
    nawm_data : dict
        Dictionary containing NAWM values for each metric and lesion label
    hc_data : dict
        Dictionary containing HC values for each metric and bundle
    metrics : list
        List of metrics to plot
    new_lesions : list
        List of lesion labels to plot
    n_sessions : int
        Number of sessions
    """
    # Set up the figure
    n_metrics = len(metrics)
    n_lesions = len(new_lesions)
    print(n_lesions, n_metrics)
    fig, axes = plt.subplots(n_lesions, n_metrics, figsize=(5*n_metrics, 6*n_lesions), squeeze=False, sharex=True)
    
    # Define colors for sessions
    colors = ['darkred' if session < target_session else '#d62728' for session in range(1, n_sessions+1)]

    # Set font size
    import matplotlib
    font_size = 24
    plt.rcParams["font.family"] = "Times New Roman"

    # Create boxplots for each metric and lesion
    for i, lesion_label in enumerate(new_lesions):
        for j, metric in enumerate(metrics):
            if n_lesions == 1 and n_metrics == 1:
                ax = axes
            elif n_lesions > 1 and n_metrics > 1:
                ax = axes[i][j]
            elif n_metrics > 1:
                ax = axes[0][j]
            else:
                ax = axes[i][0]
            
            # Prepare data for boxplot
            data_to_plot = []
            labels = []
            box_colors = []
            volumes = []  # Store volumes for annotations

            # Add HC data if available
            if len(hc_data[metric]) > 0:
                print(f'HC {metric}: {np.mean(hc_data[metric])} ± {np.std(hc_data[metric])}')
                data_to_plot.append(hc_data[metric]*1e3 if metric in ['rd', 'fixel-rd'] else hc_data[metric])
                labels.append('HC')
                box_colors.append('#2ca02c')
                volumes.append(f'{len(hc_data[metric])}')
        
            # Add NAWM data if available
            if len(nawm_data[metric][lesion_label]) > 0:
                print(f'NAWM {metric}, lesion {lesion_label}: {np.mean(nawm_data[metric][lesion_label])} ± {np.std(nawm_data[metric][lesion_label])}')
                data_to_plot.append(nawm_data[metric][lesion_label]*1e3 if metric in ['rd', 'fixel-rd'] else nawm_data[metric][lesion_label])
                labels.append('NAWM')
                box_colors.append('#1f77b4')
                volumes.append(f'{len(nawm_data[metric][lesion_label])}')
            
            # Add new lesion data for each session
            for session in range(1, n_sessions+1):
                if len(boxplot_data[metric][lesion_label][session]) > 0:
                    print(f'Lesion {metric}, lesion {lesion_label}, session {session}: {np.mean(boxplot_data[metric][lesion_label][session])} ± {np.std(boxplot_data[metric][lesion_label][session])}')
                    data_to_plot.append(boxplot_data[metric][lesion_label][session]*1e3 if metric in ['rd', 'fixel-rd'] else boxplot_data[metric][lesion_label][session])
                    labels.append(f'ses-{session}')
                    box_colors.append(colors[session-1])
                    volumes.append(f'{len(boxplot_data[metric][lesion_label][session])}')
        
            # Create boxplot if data exists
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, patch_artist=True)

                # Set x-tick positions to match the number of boxes
                ax.set_xticks(range(1, len(data_to_plot) + 1))

                # Format y-axis ticks to show 2 decimal places
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                
                # Color the boxes based on type (HC, NAWM, or session)
                for k, box in enumerate(bp['boxes']):
                    box.set(facecolor=box_colors[k])
                    
                    # Add volume annotation above each box only for the first column
                    if j == 0:  # Only for first column
                        ymax = np.max(data_to_plot[k])
                        y_pos = ymax + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02  # Position slightly above the box
                        ax.text(k + 1, y_pos, volumes[k], 
                               horizontalalignment='center',
                               verticalalignment='bottom',
                               fontsize=font_size-10)
                
                # Set labels and title
                if i == n_lesions-1:
                    ax.set_xticklabels(labels, rotation=45)
                if i == 0:
                    ax.set_title(f'{metric}', fontsize=font_size)
                if j == 0:
                    ax.set_ylabel(f'Lesion-{lesion_tags[lesion_label]}', fontsize=font_size)
                
                # Set y-axis tick label size
                ax.tick_params(axis='both', which='major', labelsize=font_size-8)

                if metric == 'fixel-fa':
                    ax.set_ylim(0.0, 1.1)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No data available', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)
        print('\n\n')
    
    # Add overall title
    fig.suptitle(f'New-Lesion-Specific Longitudinal Grid (Tool 5) | {patient_tags[subject]}', fontsize=font_size+4)
    
    # Add legend
    handles = [
        plt.Rectangle((0,0),1,1, color='#2ca02c', label='HC'),
        plt.Rectangle((0,0),1,1, color='#1f77b4', label='NAWM'),
        plt.Rectangle((0,0),1,1, color='darkred', label=f'Pre-lesion'),
        plt.Rectangle((0,0),1,1, color='#d62728', label=f'Lesion')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.022), fontsize=font_size-8)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.show()

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
    parser.add_argument('--target_session', type=int, default=2, help='Target session to start tracking new lesions. Default is 2.')
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
        args.n_sessions,
        args.target_session
    )

if __name__ == '__main__':
    main()
