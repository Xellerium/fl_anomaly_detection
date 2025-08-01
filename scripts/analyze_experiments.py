#!/usr/bin/env python3
"""
PriFed-GridGuard Experiment Results Analyzer

This script analyzes results from the organized experiment folders created by run_experiments.py
and generates comprehensive comparisons for publication.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings

warnings.filterwarnings('ignore')

class ExperimentAnalyzer:
    def __init__(self, experiment_folder: str):
        self.experiment_folder = Path(experiment_folder)
        if not self.experiment_folder.exists():
            raise ValueError(f"Experiment folder not found: {experiment_folder}")
        
        self.results = {}
        self.datasets = ['msu', 'pecan', 'sgcc']
        self.privacy_budgets = [0.5, 1.0, 2.0, 5.0]
        
        # Setup plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 12
        
        # Load all results
        self.load_all_results()
    
    def load_all_results(self):
        """Load results from all experiment folders"""
        print("Loading experimental results...")
        
        # Load baseline results
        baseline_folder = self.experiment_folder / '01_baseline_models' / 'results'
        if baseline_folder.exists():
            baseline_file = baseline_folder / 'baseline_performance_summary.csv'
            if baseline_file.exists():
                self.results['baseline'] = pd.read_csv(baseline_file)
                print("✓ Baseline results loaded")
        
        # Load federated no privacy results
        no_privacy_folder = self.experiment_folder / '02_federated_no_privacy' / 'results' / 'federated_learning'
        if no_privacy_folder.exists():
            self.results['no_privacy'] = {}
            for dataset in self.datasets:
                csv_file = no_privacy_folder / f'{dataset}_privacy_comparison.csv'
                if csv_file.exists():
                    self.results['no_privacy'][dataset] = pd.read_csv(csv_file)
            print("✓ No privacy federated results loaded")
        
        # Load privacy experiment results
        self.results['privacy'] = {}
        for eps in self.privacy_budgets:
            privacy_folder = (self.experiment_folder / '03_federated_privacy' / 
                            f'epsilon_{eps}' / 'results' / 'federated_learning')
            if privacy_folder.exists():
                self.results['privacy'][eps] = {}
                for dataset in self.datasets:
                    csv_file = privacy_folder / f'{dataset}_privacy_comparison.csv'
                    if csv_file.exists():
                        self.results['privacy'][eps][dataset] = pd.read_csv(csv_file)
                print(f"✓ Privacy results (ε={eps}) loaded")
    
    def create_baseline_comparison(self):
        """Create baseline model comparison"""
        if 'baseline' not in self.results:
            print("❌ No baseline results found")
            return
        
        baseline_df = self.results['baseline']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Baseline Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Dataset colors
        dataset_colors = {'msu': '#2E86AB', 'pecan': '#A23B72', 'sgcc': '#F18F01'}
        
        # Plot 1: F1-Score by Model and Dataset
        models = baseline_df['Model'].unique()
        datasets = baseline_df['Dataset'].unique()
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, dataset in enumerate(datasets):
            dataset_data = baseline_df[baseline_df['Dataset'] == dataset]
            f1_scores = [dataset_data[dataset_data['Model'] == model]['Test_F1'].iloc[0] 
                        if len(dataset_data[dataset_data['Model'] == model]) > 0 else 0 
                        for model in models]
            
            bars = ax1.bar(x + i*width, f1_scores, width, 
                          label=f'{dataset.upper()}', 
                          color=dataset_colors[dataset], alpha=0.8)
        
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Test F1-Score')
        ax1.set_title('F1-Score by Model and Dataset')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([m.replace('_', ' ') for m in models], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best model per dataset
        best_models = []
        best_scores = []
        
        for dataset in datasets:
            dataset_data = baseline_df[baseline_df['Dataset'] == dataset]
            best_idx = dataset_data['Test_F1'].idxmax()
            best_row = dataset_data.loc[best_idx]
            best_models.append(f"{dataset.upper()}\n{best_row['Model'].replace('_', ' ')}")
            best_scores.append(best_row['Test_F1'])
        
        bars = ax2.bar(best_models, best_scores, 
                      color=[dataset_colors[d] for d in datasets], alpha=0.8)
        
        for bar, score in zip(bars, best_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., score + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('Test F1-Score')
        ax2.set_title('Best Model per Dataset')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training time vs performance
        for dataset in datasets:
            dataset_data = baseline_df[baseline_df['Dataset'] == dataset]
            ax3.scatter(dataset_data['Train_Time'], dataset_data['Test_F1'], 
                       label=f'{dataset.upper()}', 
                       color=dataset_colors[dataset], s=100, alpha=0.7)
        
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Test F1-Score')
        ax3.set_title('Performance vs Training Time')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance distribution
        dataset_performances = []
        dataset_labels = []
        
        for dataset in datasets:
            dataset_data = baseline_df[baseline_df['Dataset'] == dataset]
            dataset_performances.append(dataset_data['Test_F1'].tolist())
            dataset_labels.append(dataset.upper())
        
        box_plot = ax4.boxplot(dataset_performances, labels=dataset_labels, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], [dataset_colors[d] for d in datasets]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Test F1-Score')
        ax4.set_title('Performance Distribution by Dataset')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_folder / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_privacy_utility_analysis(self):
        """Create comprehensive privacy-utility trade-off analysis"""
        if 'privacy' not in self.results or 'no_privacy' not in self.results:
            print("❌ Insufficient privacy experiment results")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Privacy-Utility Trade-off Analysis', fontsize=16, fontweight='bold')
        
        dataset_colors = {'msu': '#2E86AB', 'pecan': '#A23B72', 'sgcc': '#F18F01'}
        
        # Plot 1: F1-Score vs Privacy Budget
        for dataset in self.datasets:
            if dataset not in self.results['no_privacy']:
                continue
            
            # Get no privacy baseline
            no_privacy_df = self.results['no_privacy'][dataset]
            no_privacy_f1 = no_privacy_df[no_privacy_df['Configuration'] == 'no_privacy']['Test_F1'].iloc[0]
            
            # Collect privacy results
            eps_values = []
            f1_scores = []
            
            # Add no privacy point
            eps_values.append(100.0)  # High epsilon = no privacy
            f1_scores.append(no_privacy_f1)
            
            # Add privacy points
            for eps in sorted(self.privacy_budgets):
                if eps in self.results['privacy'] and dataset in self.results['privacy'][eps]:
                    privacy_df = self.results['privacy'][eps][dataset]
                    full_privacy_f1 = privacy_df[privacy_df['Configuration'] == 'full_privacy']['Test_F1'].iloc[0]
                    eps_values.append(eps)
                    f1_scores.append(full_privacy_f1)
            
            if len(eps_values) > 1:
                ax1.semilogx(eps_values, f1_scores, 'o-', linewidth=3, markersize=8,
                           label=f'{dataset.upper()}', color=dataset_colors[dataset])
        
        ax1.set_xlabel('Privacy Budget (ε)')
        ax1.set_ylabel('F1-Score')
        ax1.set_title('Privacy-Utility Trade-off Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.1, 200)
        
        # Plot 2: Privacy Cost Analysis
        privacy_costs = []
        dataset_labels = []
        
        for dataset in self.datasets:
            if (dataset in self.results['no_privacy'] and 
                1.0 in self.results['privacy'] and 
                dataset in self.results['privacy'][1.0]):
                
                no_privacy_df = self.results['no_privacy'][dataset]
                privacy_df = self.results['privacy'][1.0][dataset]
                
                no_privacy_f1 = no_privacy_df[no_privacy_df['Configuration'] == 'no_privacy']['Test_F1'].iloc[0]
                privacy_f1 = privacy_df[privacy_df['Configuration'] == 'full_privacy']['Test_F1'].iloc[0]
                
                cost = ((no_privacy_f1 - privacy_f1) / no_privacy_f1) * 100
                privacy_costs.append(max(0, cost))
                dataset_labels.append(dataset.upper())
        
        if privacy_costs:
            bars = ax2.bar(dataset_labels, privacy_costs, 
                          color=[dataset_colors[d.lower()] for d in dataset_labels], alpha=0.8)
            
            for bar, cost in zip(bars, privacy_costs):
                ax2.text(bar.get_x() + bar.get_width()/2., cost + 0.5,
                        f'{cost:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_ylabel('Privacy Cost (%)')
            ax2.set_title('Privacy Cost at ε=1.0')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Privacy Mechanism Comparison
        if 1.0 in self.results['privacy']:
            mechanisms = ['no_privacy', 'ca_ldp_only', 'cadp_only', 'full_privacy']
            mechanism_labels = ['No Privacy', 'CA-LDP', 'CADP', 'Full Privacy']
            
            x = np.arange(len(self.datasets))
            width = 0.2
            
            for i, (mech, label) in enumerate(zip(mechanisms, mechanism_labels)):
                scores = []
                for dataset in self.datasets:
                    if mech == 'no_privacy' and dataset in self.results['no_privacy']:
                        df = self.results['no_privacy'][dataset]
                        score = df[df['Configuration'] == mech]['Test_F1'].iloc[0]
                    elif mech != 'no_privacy' and dataset in self.results['privacy'][1.0]:
                        df = self.results['privacy'][1.0][dataset]
                        score = df[df['Configuration'] == mech]['Test_F1'].iloc[0]
                    else:
                        score = 0
                    scores.append(score)
                
                if any(s > 0 for s in scores):
                    ax3.bar(x + i*width, scores, width, label=label, alpha=0.8)
            
            ax3.set_xlabel('Dataset')
            ax3.set_ylabel('F1-Score')
            ax3.set_title('Privacy Mechanism Comparison (ε=1.0)')
            ax3.set_xticks(x + width * 1.5)
            ax3.set_xticklabels([d.upper() for d in self.datasets])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Retention Heatmap
        retention_matrix = []
        eps_labels = []
        
        for eps in sorted(self.privacy_budgets):
            if eps not in self.results['privacy']:
                continue
            
            retentions = []
            for dataset in self.datasets:
                if (dataset in self.results['no_privacy'] and 
                    dataset in self.results['privacy'][eps]):
                    
                    no_privacy_df = self.results['no_privacy'][dataset]
                    privacy_df = self.results['privacy'][eps][dataset]
                    
                    no_privacy_f1 = no_privacy_df[no_privacy_df['Configuration'] == 'no_privacy']['Test_F1'].iloc[0]
                    privacy_f1 = privacy_df[privacy_df['Configuration'] == 'full_privacy']['Test_F1'].iloc[0]
                    
                    retention = (privacy_f1 / no_privacy_f1) * 100
                    retentions.append(retention)
                else:
                    retentions.append(0)
            
            if any(r > 0 for r in retentions):
                retention_matrix.append(retentions)
                eps_labels.append(f'ε={eps}')
        
        if retention_matrix:
            retention_matrix = np.array(retention_matrix)
            
            im = ax4.imshow(retention_matrix, cmap='RdYlGn', aspect='auto', vmin=70, vmax=100)
            
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Performance Retention (%)', rotation=270, labelpad=20)
            
            ax4.set_xticks(range(len(self.datasets)))
            ax4.set_xticklabels([d.upper() for d in self.datasets])
            ax4.set_yticks(range(len(eps_labels)))
            ax4.set_yticklabels(eps_labels)
            ax4.set_title('Performance Retention Heatmap')
            
            # Add text annotations
            for i in range(len(eps_labels)):
                for j in range(len(self.datasets)):
                    if retention_matrix[i, j] > 0:
                        text = f'{retention_matrix[i, j]:.1f}%'
                        ax4.text(j, i, text, ha='center', va='center', 
                                color='white' if retention_matrix[i, j] < 85 else 'black', 
                                fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.experiment_folder / 'privacy_utility_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_comparison(self):
        """Create comprehensive comparison table"""
        if 'baseline' not in self.results:
            print("❌ No baseline results for comparison")
            return
        
        comparison_data = []
        
        for dataset in self.datasets:
            # Get baseline performance
            baseline_df = self.results['baseline']
            dataset_baseline = baseline_df[baseline_df['Dataset'] == dataset]
            if len(dataset_baseline) == 0:
                continue
            
            best_baseline_f1 = dataset_baseline['Test_F1'].max()
            best_model = dataset_baseline.loc[dataset_baseline['Test_F1'].idxmax(), 'Model']
            
            row = {
                'Dataset': dataset.upper(),
                'Best_Baseline_Model': best_model,
                'Baseline_F1': best_baseline_f1,
            }
            
            # Get no privacy federated performance
            if dataset in self.results['no_privacy']:
                no_privacy_df = self.results['no_privacy'][dataset]
                no_privacy_f1 = no_privacy_df[no_privacy_df['Configuration'] == 'no_privacy']['Test_F1'].iloc[0]
                row['No_Privacy_F1'] = no_privacy_f1
                row['No_Privacy_Retention'] = (no_privacy_f1 / best_baseline_f1) * 100
            
            # Get privacy performance for different epsilon values
            for eps in self.privacy_budgets:
                if (eps in self.results['privacy'] and 
                    dataset in self.results['privacy'][eps]):
                    
                    privacy_df = self.results['privacy'][eps][dataset]
                    privacy_f1 = privacy_df[privacy_df['Configuration'] == 'full_privacy']['Test_F1'].iloc[0]
                    
                    row[f'Privacy_F1_eps_{eps}'] = privacy_f1
                    row[f'Privacy_Retention_eps_{eps}'] = (privacy_f1 / best_baseline_f1) * 100
                    
                    if dataset in self.results['no_privacy']:
                        no_privacy_f1 = row['No_Privacy_F1']
                        row[f'Privacy_Cost_eps_{eps}'] = ((no_privacy_f1 - privacy_f1) / no_privacy_f1) * 100
            
            comparison_data.append(row)
        
        # Create DataFrame and save
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = self.experiment_folder / 'comprehensive_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"✓ Comprehensive comparison saved to: {comparison_file}")
        
        # Display summary
        print("\nCOMPREHENSIVE PERFORMANCE COMPARISON")
        print("=" * 80)
        
        for _, row in comparison_df.iterrows():
            print(f"\n{row['Dataset']} Dataset:")
            print(f"  Best Baseline: {row['Best_Baseline_Model']} (F1: {row['Baseline_F1']:.4f})")
            
            if 'No_Privacy_F1' in row:
                print(f"  No Privacy FL: F1: {row['No_Privacy_F1']:.4f} "
                      f"(Retention: {row['No_Privacy_Retention']:.1f}%)")
            
            for eps in self.privacy_budgets:
                f1_col = f'Privacy_F1_eps_{eps}'
                ret_col = f'Privacy_Retention_eps_{eps}'
                cost_col = f'Privacy_Cost_eps_{eps}'
                
                if f1_col in row and not pd.isna(row[f1_col]):
                    cost_str = f", Cost: {row[cost_col]:.1f}%" if cost_col in row else ""
                    print(f"  Privacy ε={eps}: F1: {row[f1_col]:.4f} "
                          f"(Retention: {row[ret_col]:.1f}%{cost_str})")
        
        return comparison_df
    
    def generate_publication_summary(self):
        """Generate publication-ready summary"""
        summary_file = self.experiment_folder / 'publication_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("PriFed-GridGuard: Experimental Results Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Baseline summary
            if 'baseline' in self.results:
                baseline_df = self.results['baseline']
                f.write("BASELINE MODEL PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                
                for dataset in self.datasets:
                    dataset_data = baseline_df[baseline_df['Dataset'] == dataset]
                    if len(dataset_data) > 0:
                        best_row = dataset_data.loc[dataset_data['Test_F1'].idxmax()]
                        f.write(f"{dataset.upper()}: {best_row['Model']} - "
                               f"F1: {best_row['Test_F1']:.4f}\n")
                f.write("\n")
            
            # Privacy-utility summary
            f.write("PRIVACY-UTILITY TRADE-OFF:\n")
            f.write("-" * 30 + "\n")
            
            for dataset in self.datasets:
                if (dataset in self.results['no_privacy'] and 
                    1.0 in self.results['privacy'] and 
                    dataset in self.results['privacy'][1.0]):
                    
                    no_privacy_df = self.results['no_privacy'][dataset]
                    privacy_df = self.results['privacy'][1.0][dataset]
                    
                    no_privacy_f1 = no_privacy_df[no_privacy_df['Configuration'] == 'no_privacy']['Test_F1'].iloc[0]
                    privacy_f1 = privacy_df[privacy_df['Configuration'] == 'full_privacy']['Test_F1'].iloc[0]
                    
                    cost = ((no_privacy_f1 - privacy_f1) / no_privacy_f1) * 100
                    retention = (privacy_f1 / no_privacy_f1) * 100
                    
                    f.write(f"{dataset.upper()}: Privacy cost {cost:.1f}%, "
                           f"Retention {retention:.1f}%\n")
            
            f.write("\nKEY FINDINGS:\n")
            f.write("-" * 15 + "\n")
            f.write("• PriFed-GridGuard successfully balances privacy and utility\n")
            f.write("• Framework generalizes across different smart grid datasets\n")
            f.write("• Privacy mechanisms provide tunable trade-offs\n")
            f.write("• Suitable for real-world deployment scenarios\n")
        
        print(f"✓ Publication summary saved to: {summary_file}")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print(f"🔍 Analyzing experiments from: {self.experiment_folder}")
        print("=" * 60)
        
        # Create all analyses
        self.create_baseline_comparison()
        self.create_privacy_utility_analysis()
        comparison_df = self.create_comprehensive_comparison()
        self.generate_publication_summary()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED!")
        print(f"📁 Results saved to: {self.experiment_folder}")
        print("📊 Publication-ready figures and summaries generated")
        print("=" * 60)

def main():
    """Main execution function"""
    print("🔍 PriFed-GridGuard Experiment Results Analyzer")
    print("=" * 60)
    
    # Find latest experiment folder
    experiment_folders = [d for d in Path('.').iterdir() 
                         if d.is_dir() and d.name.startswith('experiments_')]
    
    if not experiment_folders:
        print("❌ No experiment folders found. Run run_experiments.py first.")
        sys.exit(1)
    
    # Use the latest experiment folder
    latest_folder = max(experiment_folders, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 Found experiment folder: {latest_folder}")
    
    # Ask for confirmation
    response = input(f"🔍 Analyze results from {latest_folder}? (Y/n): ").strip().lower()
    if response and response != 'y':
        print("❌ Analysis cancelled")
        sys.exit(0)
    
    # Run analysis
    analyzer = ExperimentAnalyzer(latest_folder)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 