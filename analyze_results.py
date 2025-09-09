import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import re

def collect_results(results_dir):
    """
    Collect results from all seed/lambda combinations.
    
    Returns:
    - results_df: DataFrame with columns [seed, lambda, mcc_final, acc_iid, acc_ood]
    """
    results = []
    
    # Find all seed directories
    seed_dirs = glob(os.path.join(results_dir, "seed_*"))
    
    for seed_dir in seed_dirs:
        # Extract seed number
        seed_match = re.search(r'seed_(\d+)', seed_dir)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        
        # Find all lambda directories within this seed
        lambda_dirs = glob(os.path.join(seed_dir, "lambda_*"))
        
        for lambda_dir in lambda_dirs:
            # Extract lambda value - handle both decimal and scientific notation
            lambda_match = re.search(r'lambda_([\d\.e\-\+]+)', lambda_dir)
            if not lambda_match:
                continue
            lambda_str = lambda_match.group(1)
            # Convert scientific notation like '1e-04' to float
            try:
                lambda_val = float(lambda_str)
            except ValueError:
                continue
            
            # Read results.csv for MCC
            results_file = os.path.join(lambda_dir, "results.csv")
            accuracy_file = os.path.join(lambda_dir, "accuracy.csv")
            
            if os.path.exists(results_file):
                try:
                    # Read MCC data
                    df_results = pd.read_csv(results_file)
                    if 'mcc' in df_results.columns and len(df_results) > 0:
                        mcc_final = df_results['mcc'].iloc[-1]  # Take final MCC value
                    else:
                        mcc_final = np.nan
                    
                    # Initialize accuracy values
                    acc_iid = np.nan
                    acc_ood = np.nan
                    
                    # Read accuracy data if available
                    if os.path.exists(accuracy_file):
                        df_accuracy = pd.read_csv(accuracy_file)
                        # Try different column name variations
                        if 'acc_iid_best' in df_accuracy.columns:
                            acc_iid = df_accuracy['acc_iid_best'].iloc[0]
                        elif 'acc_iid' in df_accuracy.columns:
                            acc_iid = df_accuracy['acc_iid'].iloc[0]
                        
                        if 'acc_ood_best' in df_accuracy.columns:
                            acc_ood = df_accuracy['acc_ood_best'].iloc[0]
                        elif 'acc_ood' in df_accuracy.columns:
                            acc_ood = df_accuracy['acc_ood'].iloc[0]
                    
                    results.append({
                        'seed': seed,
                        'lambda': lambda_val,
                        'mcc_final': mcc_final,
                        'acc_iid': acc_iid,
                        'acc_ood': acc_ood
                    })
                    
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
                    continue
    
    return pd.DataFrame(results)

def plot_metrics_vs_lambda(results_df, save_dir="plots"):
    """
    Create plots showing MCC and accuracies as a function of lambda.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['mcc_final', 'acc_iid', 'acc_ood']
    titles = ['MCC vs $\\lambda$', 'IID Accuracy vs $\\lambda$', 'OOD Accuracy vs $\\lambda$']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Filter out NaN values
        plot_data = results_df.dropna(subset=[metric])
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, f'No data available for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Create box plot
        lambda_values = sorted(plot_data['lambda'].unique())
        
        # Prepare data for box plot
        box_data = []
        box_labels = []
        
        for lambda_val in lambda_values:
            metric_values = plot_data[plot_data['lambda'] == lambda_val][metric].values
            if len(metric_values) > 0:
                box_data.append(metric_values)
                box_labels.append(f'{lambda_val:.1e}' if lambda_val < 1 else f'{lambda_val}')
        
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_xlabel('$\\lambda$', fontsize=12)
        ylabel = metric.replace('mcc_final', 'MCC').replace('acc_iid', 'Accuracy IID').replace('acc_ood', 'Accuracy OOD')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        if len(box_labels) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_vs_lambda_boxplot.pdf'), bbox_inches='tight')
    plt.show()
    
    # Also create individual plots with error bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Filter out NaN values
        plot_data = results_df.dropna(subset=[metric])
        
        if len(plot_data) == 0:
            continue
        
        # Group by lambda and compute statistics
        grouped = plot_data.groupby('lambda')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Filter out groups with only one sample (can't compute error bars)
        grouped = grouped[grouped['count'] > 1]
        
        if len(grouped) > 0:
            ax.errorbar(grouped['lambda'], grouped['mean'], yerr=grouped['std'], 
                       marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        # Also plot individual points
        for lambda_val in plot_data['lambda'].unique():
            values = plot_data[plot_data['lambda'] == lambda_val][metric].values
            x_jitter = lambda_val + np.random.normal(0, lambda_val*0.01, len(values))
            ax.scatter(x_jitter, values, alpha=0.6, s=30)
        
        ax.set_xlabel('$\\lambda$', fontsize=12)
        ylabel = metric.replace('mcc_final', 'MCC').replace('acc_iid', 'Accuracy IID').replace('acc_ood', 'Accuracy OOD')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_vs_lambda_errorbar.pdf'), bbox_inches='tight')
    plt.show()

def print_summary_statistics(results_df):
    """
    Print summary statistics for each lambda value.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for lambda_val in sorted(results_df['lambda'].unique()):
        subset = results_df[results_df['lambda'] == lambda_val]
        print(f"\n$\\lambda$ = {lambda_val}")
        print("-" * 40)
        print(f"Number of seeds: {len(subset)}")
        
        for metric in ['mcc_final', 'acc_iid', 'acc_ood']:
            values = subset[metric].dropna()
            if len(values) > 0:
                metric_name = metric.replace('mcc_final', 'MCC').replace('acc_iid', 'Accuracy IID').replace('acc_ood', 'Accuracy OOD')
                print(f"{metric_name}: {values.mean():.4f} ± {values.std():.4f} (n={len(values)})")
            else:
                metric_name = metric.replace('mcc_final', 'MCC').replace('acc_iid', 'Accuracy IID').replace('acc_ood', 'Accuracy OOD')
                print(f"{metric_name}: No data available")

def main():
    # Set the results directory
    results_dir = "results"
    
    print("Collecting results from all seeds and lambda values...")
    results_df = collect_results(results_dir)
    
    if len(results_df) == 0:
        print("No results found! Make sure the results directory exists and contains data.")
        return
    
    print(f"Found {len(results_df)} result combinations")
    print(f"Seeds: {sorted(results_df['seed'].unique())}")
    print(f"Lambda values: {sorted(results_df['lambda'].unique())}")
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    # Create plots
    print("\nCreating plots...")
    plot_metrics_vs_lambda(results_df)
    
    # Save the collected data
    results_df.to_csv("collected_results.csv", index=False)
    print(f"\nResults saved to collected_results.csv")

if __name__ == "__main__":
    main()
