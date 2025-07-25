import numpy as np
import nibabel as nib
import json
import itertools
import os

def rename_metric(bundle_name, metric):
    if 'ad' in metric:
        if bundle_name in metric:
            return 'fixel_ad'
        else:
            return 'ad'
    elif 'rd' in metric:
        if bundle_name in metric:
            return 'fixel_rd'
        else:
            return 'rd'
    elif 'md' in metric:
        if bundle_name in metric:
            return 'fixel_md'
        else:
            return 'md'
    elif 'fa' in metric:
        if bundle_name in metric:
            return 'fixel_fa'
        else:
            return 'fa'
    elif 'ihMTR' in metric:
        return 'ihMTR'
    elif 'ihMTsat' in metric:
        return 'ihMTsat'
    elif 'MTR' in metric:
        return 'MTR'
    elif 'MTsat' in metric:
        return 'MTsat'
    elif 'afd_total' in metric:
        return 'afd_total'
    elif 'afd' in metric:
        return 'afd'
    elif 'nufo' in metric:
        return 'nufo'
    elif 'isovf' in metric:
        return 'isovf'
    elif 'fw' in metric:
        return 'fw'
    
    return metric

def load_metric(subject, session, bundle, metric, metrics_dirs):
    """
    Load the metric map for a given bundle and metric.
    Returns None if the metric map file does not exist.
    """

    if metric in ['fixel-ad', 'fixel-rd', 'fixel-fa', 'fixel-md']:
        # Rename metric to match the format of the metric files
        renamed_metric = metric.replace('fixel-', 'ic_')
        metric_map_filename = os.path.join(
            metrics_dirs['results_tractometry'],
            f'{subject}_ses-{session}',
            f'Fixel_MRDS',
            f'{bundle}_{renamed_metric}_metric.nii.gz')
    elif metric in ['ad', 'rd', 'fa', 'md']:
        metric_map_filename = os.path.join(
            metrics_dirs['DTI'],
            f'{subject}_ses-{session}',
            f'{metric}.nii.gz')
    elif metric in ['MTsat', 'MTR', 'ihMTsat', 'ihMTR']:
        metric_map_filename = os.path.join(
            metrics_dirs['MTsat'],
            f'{subject}_ses-{session}',
            f'{metric}.nii.gz')
    elif metric == 'fw':
        metric_map_filename = os.path.join(
            metrics_dirs['FW'],
            f'{subject}_ses-{session}',
            f'{metric}.nii.gz')
    elif metric == 'mrds-isovf':
        metric_map_filename = os.path.join(
            metrics_dirs['MRDS'],
            f'{subject}_ses-{session}',
            f'results_MRDS_Diff_TODI_ISOTROPIC.nii.gz')
    elif metric == 'todi-nufo':
        metric_map_filename = os.path.join(
            metrics_dirs['MRDS'],
            f'{subject}_ses-{session}',
            f'results_MRDS_Diff_TODI_NUM_COMP.nii.gz')
    else:
        raise ValueError(f'Metric {metric} not supported')

    # Check if file exists
    if not os.path.exists(metric_map_filename):
        return None

    metric_map = nib.load(metric_map_filename).get_fdata()

    # Special case for MRDS-isovf, isovf is the first volume of the 4D file
    if metric == 'mrds-isovf':
        return metric_map[:,:,:,0]

    return metric_map

def load_group_stats(json_filename):
    with open(json_filename, 'r+') as f:
        mean_std_per_point = json.load(f)

    stats_means = {}
    stats_stds = {}

    for bundle_name, bundle_stats in mean_std_per_point.items():
        for metric, metric_stats in bundle_stats.items():
            if (bundle_name in metric) or (not 'IFOF' in metric): #bundle
                nb_points = len(metric_stats)
                num_digits_labels = len(list(metric_stats.keys())[0])
                means = []
                stds = []
                for label_int in range(1, nb_points+1):
                    label = str(label_int).zfill(num_digits_labels)
                    mean = metric_stats.get(label, {'mean': 0})['mean']
                    std = metric_stats.get(label, {'std': 0})['std']
                    if not isinstance(mean, list):
                        mean = [mean]
                        std = [std]

                    means += [mean]
                    stds += [std]

                color = '0x727272'

                metric = rename_metric(bundle_name, metric)

                # Robustify for missing data
                means = np.array(list(itertools.zip_longest(*means,
                                                            fillvalue=np.nan))).T
                stds = np.array(list(itertools.zip_longest(*stds,
                                                            fillvalue=np.nan))).T
                for i in range(len(means)):
                    _nan = np.isnan(means[i, :])
                    if np.count_nonzero(_nan) > 0:
                        if np.count_nonzero(_nan) < len(means[i, :]):
                            means[i, _nan] = np.average(means[i, ~_nan])
                            stds[i, _nan] = np.average(stds[i, ~_nan])
                        else:
                            means[i, _nan] = -1
                            stds[i, _nan] = -1

                stats_means[(bundle_name, metric)] = means
                stats_stds[(bundle_name, metric)] = stds

    return stats_means, stats_stds

def subtract_binary_masks(mask1, mask2):
    """
    Subtracts mask1 from mask2 and returns a binary result.
    
    Parameters:
    mask1, mask2: numpy arrays or lists containing binary values (0 and 1)
    
    Returns:
    Binary mask where 1 represents areas that are in mask2 but not in mask1
    """
    
    # Perform the subtraction and keep it binary
    # This will give 1 only where mask2 is 1 AND mask1 is 0
    result = np.logical_and(mask2.astype(bool), np.logical_not(mask1.astype(bool)))
    
    # Convert back to int dtype (0 and 1)
    return result.astype(bool)

def mask_metrics_by_lesions(prev_metric_map, curr_metric_map, prev_bundle_mask, curr_bundle_mask, prev_labeled_mask, curr_labeled_mask):
    """
    Masks two 3D metric maps using a bundle and labeled masks and returns flattened results.
    
    Parameters:
    prev_metric_map: Previous metric map as 3D numpy array
    curr_metric_map: Current metric map as 3D numpy array
    prev_bundle_mask: Previous bundle mask as 3D binary numpy array
    curr_bundle_mask: Current bundle mask as 3D binary numpy array
    prev_labeled_mask: Previous labeled mask as 3D numpy array with integer labels
    curr_labeled_mask: Current labeled mask as 3D numpy array with integer labels
    
    Returns:
    tuple: (flattened_masked_prev_metric_map, flattened_masked_curr_metric_map)
    """
    
    # Convert inputs to numpy arrays if they aren't already
    prev_metric_map = np.array(prev_metric_map)
    curr_metric_map = np.array(curr_metric_map)
    prev_bundle_mask = np.array(prev_bundle_mask, dtype=np.uint8)
    curr_bundle_mask = np.array(curr_bundle_mask, dtype=np.uint8)
    prev_labeled_mask = np.array(prev_labeled_mask, dtype=np.int32)
    curr_labeled_mask = np.array(curr_labeled_mask, dtype=np.int32)

    # Check if arrays have the same shape
    if not (prev_metric_map.shape == 
            curr_metric_map.shape == 
            prev_bundle_mask.shape == 
            curr_bundle_mask.shape == 
            prev_labeled_mask.shape == 
            curr_labeled_mask.shape):
        raise ValueError("All input maps must have the same shape")

    # Create binary masks for the bundle and lesions
    bundle_mask = np.logical_and(prev_bundle_mask > 0, curr_bundle_mask > 0)
    prev_lesion_mask = prev_labeled_mask > 0
    lesion_mask = curr_labeled_mask > 0

    mask1 = np.logical_and(bundle_mask, subtract_binary_masks(prev_lesion_mask, lesion_mask))
    mask2 = np.logical_and(bundle_mask, lesion_mask)

    # Apply bundle mask to both arrays
    masked_prev_metric_map = prev_metric_map[mask1].flatten()
    masked_curr_metric_map = curr_metric_map[mask2].flatten()
    
    # Add threshold to the arrays
    masked_prev_metric_map = masked_prev_metric_map[masked_prev_metric_map > 0]
    masked_curr_metric_map = masked_curr_metric_map[masked_curr_metric_map > 0]

    return masked_prev_metric_map, masked_curr_metric_map

def subtract_labeled_masks(mask1, mask2):
    """
    Subtracts mask1 from mask2 only between voxels with the same labels,
    keeping non-negative values.
    
    Parameters:
    mask1, mask2: numpy arrays containing integer labels
    
    Returns:
    Mask with the result of the subtraction where labels match,
    keeping original values where they don't match
    """
    
    # Convert inputs to numpy arrays if they aren't already
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    
    # Create output array initialized with mask2
    result = mask2.copy()
    
    # Get unique labels (excluding 0 if it exists)
    labels = np.unique(mask2)
    labels = labels[labels != 0]  # Remove background (0) if present
    
    # For each label
    for label in labels:
        # Create masks for current label
        mask1_label = (mask1 == label)
        mask2_label = (mask2 == label)
        
        # Where both masks have the same label
        common_region = np.logical_and(mask1_label, mask2_label)
        
        # Perform subtraction only in common region
        # and ensure non-negative values
        result[common_region] = np.maximum(0, mask2[common_region] - mask1[common_region])
    
    return result
