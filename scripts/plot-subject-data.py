import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import FormatStrFormatter
from utils import get_ax
import os

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
    34: 'E3',
    218: 'E4',
    372: 'E5',
    389: 'E6',
    426: 'E7',
}

ylims = {
    'fixel-rd': [0.0003, 0.001],
    'fixel-ad': [0.003, 0.006],
    'fixel-fa': [0.2, 0.9],
    'fixel-md': [0.0000, 0.0008],
    'rd': [0.0003, 0.001],
    'ad': [0.003, 0.006],
    'fa': [0.2, 0.9],
    'md': [0.0000, 0.0008],
    'MTsat': [0,1],
    'fw': [0.1,0.4],
    'mrds-isovf': [0.1,0.4],
}

def rename_metric(metric):
    if metric in ['fixel-ad', 'fixel-rd', 'fixel-fa', 'fixel-md']:
        return 'fixel-'+metric.replace('fixel-', '').upper()
    elif metric in ['ad', 'rd', 'fa', 'md']:
        return metric.upper()
    elif metric in ['MTsat', 'MTR', 'ihMTsat', 'ihMTR']:
        return metric
    elif metric == 'fw':
        return 'FW'
    elif metric == 'mrds-isovf':
        return 'MRDS-ISOVF'
    elif metric == 'todi-nufo':
        return 'TODI-NuFO'
    elif metric == 'afd_total':
        return 'AFD_total'
    elif metric == 'afd':
        return 'AFD'
    elif metric == 'nufo':
        return 'NuFO'
    elif metric == 'isovf':
        return 'ISOVF'
    return metric

def create_lesion_plots(
    patient,
    patient_csv_dir,
    bundle,
    bundle_sections,
    labels,
    bundles,
    metrics,
    n_sessions=5,
    add_std=False,
    add_grid=False,
    add_legend=False,
    hc_subjects=[],
    hc_csv_dir=None,
    color_lesion='#d62728',
    color_penumbra='#ff7f0e',
    color_nawm='#1f77b4',
    color_hc='#2ca02c',
    font_size=16,
    linewidth=1
):
    """
    Create grid of plots
    
    patient: patient ID. e.g. "sub-004-ms"
    bundle: name of the bundle to filter data (optional). Plot all sections of the bundle.
    labels: list of lesion labels to filter data (optional). Plot all lesion labels.
    bundles: list of bundle names to filter data (optional). Plot all bundles.
    metrics: list of metrics to plot.
    n_sessions: number of sessions to plot.
    add_std: add standard deviation to the plot.
    hc_subjects: HC subjects.
    color_lesion: Color for lesion region.
    color_penumbra: Color for penumbra region.
    color_nawm: Color for NAWM region.
    color_hc: Color for HC region.
    """

    # Determine rows
    if labels is not None:
        rows = labels
    elif bundles is not None:
        rows = bundles
    elif bundle_sections is not None:
        rows = bundle_sections
    else:
        rows = None

    if rows is None:
        raise ValueError('No rows specified')

    # Determine input CSV file (lesion-specific, bundle-specific, bundle-section-specific)
    if bundles is not None:
        file_type = 'per_bundle_data'
        y_label = ''
        title = f'Bundle-Specific Longitudinal Grid (Tool 2) | {patient}'
    elif bundle_sections is not None:
        file_type = 'per_bundle_section_data'
        y_label = 'Section '
        title = f'Bundle-Section-Specific Longitudinal Grid (Tool 3) | Bundle: {bundle} | {patient}'
    elif labels is not None:
        file_type = 'per_lesion_data'
        y_label = 'Lesion-'
        title = f'Lesion-Specific Longitudinal Grid (Tool 4) | {patient}'

    # Read the patient dataframe from the input CSV file
    df = pd.read_csv(os.path.join(
        patient_csv_dir,
        f'{patient}_{file_type}.csv'
    ))

    # Read the HC cohort dataframes from the input CSV files if provided
    if (hc_subjects is not None) and (hc_csv_dir is not None):
        hc_dfs = [pd.read_csv(os.path.join(
            hc_csv_dir,
            f'{subject}_{file_type}.csv'
        )) for subject in hc_subjects]
        hc_df = pd.concat(hc_dfs)

    # Set font size
    import matplotlib
    font = {'size' : font_size}
    matplotlib.rc('font', **font)
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Create a figure with subplots
    n_rows = len(rows)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6*n_rows), squeeze=False, sharex=True)
    
    # Colors for different regions
    colors = {
        'lesion': color_lesion,
        'penumbra': color_penumbra,
        'nawm': color_nawm,
    }
    
    # Plot for each row (lesion label, bundle or bundle section) and metric combination
    for i, row in enumerate(rows):
        for j, metric in enumerate(metrics):
            ax = get_ax(axes, i, j, n_rows, n_cols)
            
            # Filter data for current label and metric
            if bundle_sections is not None:
                mask = (df['bundle'] == bundle) & (df['label'] == row) & (df['metric'] == metric)
                if hc_subjects is not None:
                    hc_mask = (hc_df['bundle'] == bundle) & (hc_df['label'] == row) & (hc_df['metric'] == metric)
            elif labels is not None:
                mask = (df['label'] == row) & (df['metric'] == metric)
                if hc_subjects is not None:
                    hc_mask = (hc_df['metric'] == metric)
            elif bundles is not None:
                mask = (df['bundle'] == row) & (df['metric'] == metric)
                if hc_subjects is not None:
                    hc_mask = (hc_df['bundle'] == row) & (hc_df['metric'] == metric)
            data = df[mask]
            if hc_subjects is not None:
                hc_data = hc_df[hc_mask]
            
            # Plot each region (lesion, penumbra, nawm)
            for region, color in colors.items():
                region_median = data.groupby('session')[region+'_median'].mean()
                region_std = data.groupby('session')[region+'_std'].mean()
                ax.errorbar(
                    region_median.index,
                    region_median.values*1e3 if metric in ['rd', 'fixel-rd'] else region_median.values, 
                    yerr=None if not add_std else region_std.values*1e3 if metric in ['rd', 'fixel-rd'] else region_std.values,
                    fmt='o-',
                    color=color,
                    label=region,
                    capsize=3,
                    capthick=1, 
                    linewidth=linewidth
                )

            # Plot healthy controls
            if hc_subjects is not None:
                hc_median = hc_data.groupby('session')['median'].mean()
                hc_std = hc_data.groupby('session')['std'].mean()
                ax.errorbar(
                    hc_median.index,
                    hc_median.values*1e3 if metric in ['rd', 'fixel-rd'] else hc_median.values, 
                    yerr=None if not add_std else hc_std.values*1e3 if metric in ['rd', 'fixel-rd'] else hc_std.values,
                    fmt='o--',
                    color=color_hc,
                    label='HC control',
                    capsize=3,
                    capthick=1, 
                    linewidth=linewidth
                )
            
            # Format y-axis ticks to show 2 decimal places
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # Set labels and title
            if i == n_rows-1:  # Only bottom row
                ax.set_xlabel('session', fontsize=font_size)
                ax.set_xticks(np.arange(1, n_sessions+1))
            if j == 0:  # Only leftmost column
                if y_label == 'Lesion-':
                    ax.set_ylabel(f"{y_label}{row}", fontsize=font_size)
                else:
                    ax.set_ylabel(f"{y_label}{row}", fontsize=font_size)
            if i == 0:  # Only top row
                ax.set_title(rename_metric(metric))

            # Add grid
            if add_grid:
                ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(title)

    # Add legend
    if add_legend:
        handles = [
            plt.Rectangle((0,0),1,1, color=color_hc, label='HC'),
            plt.Rectangle((0,0),1,1, color=color_nawm, label='NAWM'),
            plt.Rectangle((0,0),1,1, color=color_penumbra, label='Penumbra'),
            plt.Rectangle((0,0),1,1, color=color_lesion, label='Lesion')
        ]
        fig.legend(
            handles=handles, 
            loc='lower center', 
            ncol=4, 
            bbox_to_anchor=(0.5, -0.01), 
            fontsize=font_size-4
        )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create plots for given lesion labels and metrics')
    parser.add_argument('--patient', required=True, help='Patient ID. e.g. "sub-004-ms"')
    parser.add_argument('--patient_csv_dir', required=True, help='Path to patient CSV directory')
    parser.add_argument('--metrics', nargs='+', type=str, required=True, help='List of metrics')
    parser.add_argument('--lesions', nargs='+', type=int, required=False, help='List of lesion labels')
    parser.add_argument('--bundles', nargs='+', type=str, required=False, help='List of bundles')
    parser.add_argument('--bundle', type=str, required=False, help='Bundle name to plot bundle sections. Required if bundle_sections is specified.')
    parser.add_argument('--bundle_sections', nargs='+', type=int, required=False, help='Bundle sections to plot. Required if bundle is specified.')
    parser.add_argument('--hc_subjects', nargs='+', type=str, required=False, help='HC subjects')
    parser.add_argument('--hc_csv_dir', help='Path to HC CSV directory')
    parser.add_argument('--n_sessions', type=int, default=5, help='Number of sessions')
    parser.add_argument('--color_lesion', default='#d62728', help='Color for lesion region')
    parser.add_argument('--color_penumbra', default='#ff7f0e', help='Color for penumbra region')
    parser.add_argument('--color_nawm', default='#1f77b4', help='Color for NAWM region')
    parser.add_argument('--color_hc', default='#2ca02c', help='Color for HC region')
    parser.add_argument('--linewidth', type=int, default=1, help='Line width')
    parser.add_argument('--font_size', type=int, default=16, help='Font size')
    parser.add_argument('-add_std', action='store_true', help='Add standard deviation to the plot')
    parser.add_argument('-add_grid', action='store_true', help='Add grid to the plot')
    parser.add_argument('-add_legend', action='store_true', help='Add legend to the plot')
    parser.add_argument('-save_fig', action='store_true', help='Save figure')
    args = parser.parse_args()
    
    # Create the plot   
    fig = create_lesion_plots(
        args.patient,
        args.patient_csv_dir,
        args.bundle, 
        args.bundle_sections, 
        args.lesions, 
        args.bundles,
        args.metrics,
        args.n_sessions,
        args.add_std,
        args.add_grid,
        args.add_legend,
        args.hc_subjects,
        args.hc_csv_dir,
        args.color_lesion,
        args.color_penumbra,
        args.color_nawm,
        args.color_hc,
        args.font_size,
        args.linewidth
    )

    # Save figure
    if args.save_fig:
        out_filename = f'{args.patient}_per_bundle_plots.png'
        plt.savefig(out_filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {out_filename}")

    # Show the plot
    plt.show()
