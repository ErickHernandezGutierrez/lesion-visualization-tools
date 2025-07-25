import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.ticker import FormatStrFormatter

patient_tags = {
    'sub-006-ms': 'Patient-A',
    'sub-008-ms': 'Patient-B',
    'sub-010-ms': 'Patient-C',
    'sub-018-ms': 'Patient-D',
    'sub-004-ms': 'Patient-E',
    'sub-002-ms': 'Patient-F',
    'sub-005-ms': 'Patient-G',
    'sub-020-ms': 'Patient-H',
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

def create_cohort_plots(
    patients,
    bundle,
    metrics,
    n_sessions=5,
    add_std=False,
    hc_subjects=[]
):
    """
    Create grid of plots for a cohort of patients
    
    patients: list of patient IDs. e.g. ["sub-004-ms", "sub-005-ms"]
    bundle: name of the bundle to filter data.
    metrics: list of metrics to plot.
    n_sessions: number of sessions to plot.
    add_std: add standard deviation to the plot.
    hc_subjects: HC subjects
    """

    # Read the dataframes from the input CSV files
    patient_dfs = {}
    for patient in patients:
        file_path = f'D:\\SCIL\\ms_6months\\{patient}_per_bundle_data.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            patient_dfs[patient] = df
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not patient_dfs:
        raise ValueError("No valid patient data found")
    
    # Load HC data if provided
    if hc_subjects:
        hc_dfs = []
        for subject in hc_subjects:
            hc_file_path = f'D:\\SCIL\\myelo_inferno_imk\\{subject}_per_bundle_data.csv'
            if os.path.exists(hc_file_path):
                hc_df = pd.read_csv(hc_file_path)
                hc_dfs.append(hc_df)
            else:
                print(f"Warning: HC file not found: {hc_file_path}")
        
        if hc_dfs:
            hc_df = pd.concat(hc_dfs)
            
            # Calculate mean values across HC subjects
            hc_means = {}
            for metric in metrics:
                hc_mask = (hc_df['bundle'] == bundle) & (hc_df['metric'] == metric)
                hc_data = hc_df[hc_mask]
                
                if not hc_data.empty:
                    hc_means[metric] = hc_data.groupby('session')['median'].mean()
        else:
            hc_df = None
            hc_means = {}
            print("Warning: No valid HC data found")
    else:
        hc_df = None
        hc_means = {}
    
    # Set font size
    import matplotlib
    font_size = 24  # Reduced font size for better fit in grid
    font = {'size': font_size}
    matplotlib.rc('font', **font)
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Create a figure with subplots - one row per patient
    n_rows = len(patients)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6*n_rows), squeeze=False, sharex=True, sharey='col')
    
    # Colors for different regions
    colors = {
        'lesion': '#d62728',
        'penumbra': '#ff7f0e',
        'nawm': '#1f77b4'
    }
    
    # Plot for each patient and metric
    for i, patient in enumerate(patients):
        df = patient_dfs.get(patient)
        if df is None:
            continue
            
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            
            # Filter data for current bundle and metric
            mask = (df['bundle'] == bundle) & (df['metric'] == metric)
            data = df[mask]
            
            if data.empty:
                print(f"Warning: No data found for patient '{patient}', bundle '{bundle}' and metric '{metric}'")
                continue
            
            # Plot each region (lesion, penumbra, nawm)
            for region, color in colors.items():
                # Get median values for each session
                region_values = data.groupby('session')[f'{region}_median'].mean()
                region_std = data.groupby('session')[f'{region}_std'].mean() if add_std else None
                
                # Convert values if needed
                y_values = region_values.values
                if metric in ['rd', 'fixel-rd']:
                    y_values = y_values * 1e3
                    
                # Get standard deviation if needed
                y_err = None
                if add_std and region_std is not None:
                    y_err = region_std.values
                    if metric in ['rd', 'fixel-rd']:
                        y_err = y_err * 1e3
                
                print(f'Patient {patient}, {region} {metric}: {y_values}')
                
                ax.errorbar(
                    region_values.index,
                    y_values,
                    yerr=y_err,
                    fmt='o-',
                    color=color,
                    label=region,
                    capsize=3,
                    capthick=1, 
                    elinewidth=1
                )

            # Plot healthy controls if available
            if metric in hc_means and not hc_means[metric].empty:
                hc_values = hc_means[metric].values
                if metric in ['rd', 'fixel-rd']:
                    hc_values = hc_values * 1e3
                
                ax.plot(
                    hc_means[metric].index,
                    hc_values,
                    'o--',
                    color='#2ca02c',
                    label='HC mean'
                )
            
            # Format y-axis ticks to show 2 decimal places
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # Set labels and title
            if i == n_rows-1:  # Only bottom row
                ax.set_xlabel('session', fontsize=16)
            ax.set_xticks(np.arange(1, n_sessions+1))
            
            if j == 0:  # Only leftmost column
                ax.set_ylabel(f'{patient_tags[patient]}', fontsize=20)
            
            if i == 0:  # Only top row
                ax.set_title(metric)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(f'Patient-Based Longitudinal Grid (Tool 1) | Bundle: {bundle}')

    # Add legend - only once for the entire figure
    handles = [
        plt.Rectangle((0,0),1,1, color='#2ca02c', label='HC mean'),
        plt.Rectangle((0,0),1,1, color='#1f77b4', label='NAWM'),
        plt.Rectangle((0,0),1,1, color='#ff7f0e', label='Penumbra'),
        plt.Rectangle((0,0),1,1, color='#d62728', label='Lesion')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cohort plots for given patients, bundle and metrics')
    parser.add_argument('--patients', nargs='+', type=str, required=True, help='List of patient IDs. e.g. "sub-004-ms sub-005-ms"')
    parser.add_argument('--bundle', type=str, required=True, help='Bundle name to plot')
    parser.add_argument('--metrics', nargs='+', type=str, required=True, help='List of metrics')
    parser.add_argument('--hc_subjects', nargs='+', type=str, required=False, help='HC subjects')
    parser.add_argument('--n_sessions', type=int, default=5, help='Number of sessions')
    parser.add_argument('-add_std', action='store_true', help='Add standard deviation to the plot')
    parser.add_argument('-save_fig', action='store_true', help='Save figure')
    parser.add_argument('--output', type=str, help='Output filename (without extension)')
    args = parser.parse_args()
    
    # Create the plot   
    fig = create_cohort_plots(
        args.patients,
        args.bundle,
        args.metrics,
        args.n_sessions,
        args.add_std,
        args.hc_subjects
    )

    # Save figure
    if args.save_fig:
        out_filename = f'{args.output if args.output else "cohort"}_bundle_{args.bundle}_plots.png'
        plt.savefig(out_filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {out_filename}")

    # Show the plot
    plt.show() 