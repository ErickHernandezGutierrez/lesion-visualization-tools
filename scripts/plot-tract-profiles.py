import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from utils import load_group_stats, load_subject_stats

def plot_tract_profiles(ax, means, stds, xlabel='', ylabel='', color='black', add_first_variance=False, add_second_variance=False, is_cohort=False, zorder=1, label=None):
    if is_cohort:
        mean = np.average(means, axis=1)
        std = np.std(means, axis=1)
    else:
        mean = np.array(means).ravel()
        std = np.array(stds).ravel()
    dim = np.arange(1, len(mean)+1)

    # Set axis labels
    ax.set_xlabel(xlabel)
    if ylabel in ['AD', 'RD', 'MD', 'fixel-AD', 'fixel-RD', 'fixel-MD']:
        ax.set_ylabel(ylabel + r' [$\mu m^2/ms$]')
        mean *= 1e3
        std  *= 1e3
    else:
        ax.set_ylabel(ylabel)

    # Set axis ticks
    ax.set_xticks(dim)

    # Plot tract profiles
    ax.plot(dim, mean, linewidth=5, color=color, label=label, zorder=zorder)

    # Add tract profile variances
    if add_first_variance:
        ax.fill_between(dim, mean-std, mean+std, facecolor=color, alpha=0.5, zorder=zorder)
    if add_second_variance:
        ax.fill_between(dim, mean-2*std, mean+2*std, facecolor=color, alpha=0.3, zorder=zorder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot tract profiles')
    parser.add_argument('--patient', help='Subject ID')
    parser.add_argument('--patient_tractometry_results_dir', help='Path to patient tractometry results directory')
    parser.add_argument('--cohort_tractometry_results_dir', help='Path to cohort tractometry results directory')
    parser.add_argument('--metrics', nargs='+', help='Metrics to plot')
    parser.add_argument('--bundles', nargs='+', help='Bundles to plot')
    parser.add_argument('-add_first_variance', action='store_true', help='Add first variance to the plot')
    parser.add_argument('-add_second_variance', action='store_true', help='Add second variance to the plot')
    parser.add_argument('-add_legend', action='store_true', help='Add legend to the plot')
    parser.add_argument('--patient_label', default='Patient', help='Patient label in the legend')
    parser.add_argument('--cohort_label', default='Cohort', help='Cohort label in the legend')
    parser.add_argument('--patient_color', default='#D62728', help='Patient color')
    parser.add_argument('--cohort_color', default='#272829', help='Cohort color')
    parser.add_argument('--title', help='Title of the plot')
    parser.add_argument('--figsize', nargs=2, type=int, default=(12,5), help='Figure size')
    args = parser.parse_args()

    patient_stats_mean, patient_stats_std = load_subject_stats(os.path.join(
        args.patient_tractometry_results_dir, 
        args.patient,
        'Bundle_Mean_Std_Per_Point',
        f'{args.patient}__mean_std_per_point.json'
    ))

    cohort_stats_mean, cohort_stats_std = load_group_stats(os.path.join(
        args.cohort_tractometry_results_dir,
        'Statistics',
        'mean_std_per_point.json'
    ))

    n_metrics = len(args.metrics)
    n_bundles = len(args.bundles)

    fig, axes = plt.subplots(nrows=n_bundles, ncols=n_metrics, figsize=args.figsize)
    for i, bundle in enumerate(args.bundles):
        for j, metric in enumerate(args.metrics):
            if n_bundles == 1 and n_metrics == 1:
                ax = axes
            elif n_bundles > 1 and n_metrics > 1:
                ax = axes[i][j]
            elif n_metrics > 1:
                ax = axes[j]
            else:
                ax = axes[i]

            plot_tract_profiles(
                ax, 
                patient_stats_mean[(bundle, metric)],
                patient_stats_std[(bundle, metric)], 
                xlabel='Location along the bundle', 
                ylabel=metric,
                color=args.patient_color,
                label=f'{args.patient_label} {args.patient}',
                zorder=2 # Patient plot on top of cohort plot
            )
            plot_tract_profiles(
                ax, 
                cohort_stats_mean[(bundle, metric)],
                cohort_stats_std[(bundle, metric)], 
                xlabel='Location along the bundle', 
                ylabel=metric,
                color=args.cohort_color,
                label=f'{args.cohort_label}',
                add_first_variance=args.add_first_variance,
                add_second_variance=args.add_second_variance,
                is_cohort=True
            )

            ax.set_title(f'Bundle: {bundle}')
    if args.title:
        plt.suptitle(args.title)
    if args.add_legend:
        plt.legend(loc='upper right')
    plt.show()
