# baseline_models.py
"""
Enhanced Baseline Models for Multi-Dataset Smart Grid Anomaly Detection
Implements modern ML algorithms for centralized learning across three datasets

Key Models Evaluated:
1. Random Forest - Tree-based ensemble method
2. XGBoost - Gradient boosting framework  
3. LightGBM - Microsoft's gradient boosting
4. CatBoost - Yandex's gradient boosting
5. Logistic Regression - Linear baseline

Datasets:
- MSU Power System Attack Dataset
- Pecan Street Energy Consumption Dataset
- SGCC Electricity Theft Dataset

Features:
- Cross-dataset performance comparison
- Privacy vs non-privacy preprocessing comparison
- Automated hyperparameter tuning
- Publication-ready visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                           recall_score, classification_report, confusion_matrix,
                           roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
import time
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class MultiDatasetBaselineEvaluator:
    """Handles training and evaluation of baseline ML models across multiple datasets"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.dataset_names = ['msu', 'pecan', 'sgcc']
        self.datasets = {}
        self.cuda_available = self._check_cuda_available()
        
        if self.cuda_available:
            print(f"[SUCCESS] CUDA is available and will be used for supported models")
        else:
            print(f"[INFO] CUDA not available, using CPU")
    
    def _check_cuda_available(self):
        """Check if CUDA is available for GPU acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        
    def load_dataset(self, dataset_name: str, privacy_mode: bool = True, 
                    data_path: str = "data/processed") -> bool:
        """Load preprocessed data for a specific dataset"""
        data_path = Path(data_path) / dataset_name
        
        try:
            with open(data_path / 'data_splits.pkl', 'rb') as f:
                splits = pickle.load(f)
            
            with open(data_path / 'preprocessing_objects.pkl', 'rb') as f:
                prep_objects = pickle.load(f)
            
            dataset_info = {
                'X_train': splits['X_train'],
                'X_val': splits['X_val'],
                'X_test': splits['X_test'],
                'y_train': splits['y_train'],
                'y_val': splits['y_val'],
                'y_test': splits['y_test'],
                'label_encoder': prep_objects['label_encoder'],
                'class_names': list(prep_objects['label_encoder'].classes_),
                'privacy_applied': splits.get('privacy_applied', False),
                'n_features': splits['n_features'],
                'n_samples': splits['n_samples'],
                'class_distribution': splits['class_distribution']
            }
            
            # Store dataset
            key = f"{dataset_name}_{'privacy' if privacy_mode else 'no_privacy'}"
            self.datasets[key] = dataset_info
            
            print(f"\n{dataset_name.upper()} Dataset loaded successfully:")
            print(f"  Privacy mode: {'ENABLED' if privacy_mode else 'DISABLED'}")
            print(f"  Training: {len(dataset_info['X_train']):,} samples")
            print(f"  Validation: {len(dataset_info['X_val']):,} samples")
            print(f"  Test: {len(dataset_info['X_test']):,} samples")
            print(f"  Features: {dataset_info['n_features']}")
            print(f"  Classes: {dataset_info['class_names']}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Processed data for {dataset_name} not found.")
            return False
    
    def load_all_datasets(self, privacy_mode: bool = True):
        """Load all available datasets"""
        print("="*60)
        print(f"Loading All Datasets - Privacy Mode: {'ENABLED' if privacy_mode else 'DISABLED'}")
        print("="*60)
        
        for dataset_name in self.dataset_names:
            self.load_dataset(dataset_name, privacy_mode)
    
    def define_models(self, optimize_hyperparams: bool = False):
        """Define baseline models with optional hyperparameter optimization"""
        
        if optimize_hyperparams:
            # Define parameter grids for optimization
            self.param_grids = {
                'Random_Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3]
                },
                'LightGBM': {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 50, 100],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            }
        
        # Default models
        self.models = {
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15, 
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='mlogloss',
                verbosity=0,
                tree_method='gpu_hist' if self._check_cuda_available() else 'auto',
                gpu_id=0 if self._check_cuda_available() else -1
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1,
                device='gpu' if self._check_cuda_available() else 'cpu',
                gpu_platform_id=0 if self._check_cuda_available() else -1,
                gpu_device_id=0 if self._check_cuda_available() else -1
            ),
            
            'CatBoost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            ),
            
            'Logistic_Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        }
    
    def train_single_model(self, model_name: str, model, dataset_key: str) -> Dict:
        """Train a single model on a specific dataset"""
        dataset = self.datasets[dataset_key]
        
        print(f"\nTraining {model_name} on {dataset_key}...")
        start_time = time.time()
        
        # Handle model-specific parameters based on number of classes
        n_classes = len(dataset['class_names'])
        
        if model_name == 'XGBoost':
            if n_classes == 2:
                # For binary classification, recreate model with correct parameters
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    random_state=self.random_state,
                    eval_metric='logloss',
                    verbosity=0,
                    tree_method='gpu_hist' if self.cuda_available else 'auto',
                    gpu_id=0 if self.cuda_available else -1
                )
            else:
                # For multi-class, set the num_class parameter
                model.set_params(objective='multi:softprob', num_class=n_classes)
        
        elif model_name == 'LightGBM':
            if n_classes == 2:
                # For binary classification
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    num_leaves=31,
                    learning_rate=0.1,
                    objective='binary',
                    random_state=self.random_state,
                    verbosity=-1,
                    device='gpu' if self.cuda_available else 'cpu',
                    gpu_platform_id=0 if self.cuda_available else -1,
                    gpu_device_id=0 if self.cuda_available else -1
                )
            else:
                # For multi-class
                model.set_params(objective='multiclass', num_class=n_classes)
        
        # Train model
        model.fit(dataset['X_train'], dataset['y_train'])
        train_time = time.time() - start_time
        
        # Make predictions
        y_pred_val = model.predict(dataset['X_val'])
        y_pred_test = model.predict(dataset['X_test'])
        
        # Calculate metrics
        metrics = {
            'train_time': train_time,
            'val_accuracy': accuracy_score(dataset['y_val'], y_pred_val),
            'val_f1': f1_score(dataset['y_val'], y_pred_val, average='weighted'),
            'val_precision': precision_score(dataset['y_val'], y_pred_val, average='weighted'),
            'val_recall': recall_score(dataset['y_val'], y_pred_val, average='weighted'),
            'test_accuracy': accuracy_score(dataset['y_test'], y_pred_test),
            'test_f1': f1_score(dataset['y_test'], y_pred_test, average='weighted'),
            'test_precision': precision_score(dataset['y_test'], y_pred_test, average='weighted'),
            'test_recall': recall_score(dataset['y_test'], y_pred_test, average='weighted'),
        }
        
        # Try to get probability predictions for AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(dataset['X_test'])
            if len(dataset['class_names']) == 2:
                # Binary classification
                metrics['test_auc'] = roc_auc_score(dataset['y_test'], y_pred_proba[:, 1])
            else:
                # Multi-class - use one-vs-rest
                try:
                    metrics['test_auc'] = roc_auc_score(
                        dataset['y_test'], y_pred_proba, multi_class='ovr'
                    )
                except:
                    metrics['test_auc'] = None
        
        # Store additional info
        metrics['confusion_matrix'] = confusion_matrix(dataset['y_test'], y_pred_test)
        metrics['classification_report'] = classification_report(
            dataset['y_test'], y_pred_test,
            target_names=dataset['class_names']
        )
        
        return metrics
    
    def train_all_models(self, dataset_key: str) -> Dict[str, Dict]:
        """Train all models on a specific dataset"""
        results = {}
        
        print(f"\n{'='*60}")
        print(f"Training All Models on {dataset_key}")
        print(f"{'='*60}")
        
        for model_name, model in self.models.items():
            results[model_name] = self.train_single_model(model_name, model, dataset_key)
            
            # Print summary
            print(f"  Validation Accuracy: {results[model_name]['val_accuracy']:.4f}")
            print(f"  Validation F1-Score: {results[model_name]['val_f1']:.4f}")
            print(f"  Test Accuracy: {results[model_name]['test_accuracy']:.4f}")
            print(f"  Test F1-Score: {results[model_name]['test_f1']:.4f}")
            print(f"  Training Time: {results[model_name]['train_time']:.2f}s")
        
        return results
    
    def compare_across_datasets(self, privacy_mode: bool = True):
        """Train and compare models across all datasets"""
        comparison_results = {}
        
        print("\n" + "="*80)
        print("CROSS-DATASET BASELINE MODEL COMPARISON")
        print("="*80)
        
        # Train on each dataset
        for dataset_name in self.dataset_names:
            key = f"{dataset_name}_{'privacy' if privacy_mode else 'no_privacy'}"
            if key in self.datasets:
                comparison_results[dataset_name] = self.train_all_models(key)
        
        # Store results
        self.results['comparison'] = comparison_results
        self.results['privacy_mode'] = privacy_mode
        
        # Find best model per dataset
        print("\n" + "="*60)
        print("BEST PERFORMING MODELS BY DATASET")
        print("="*60)
        
        best_models = {}
        for dataset_name, dataset_results in comparison_results.items():
            best_model = max(dataset_results.items(), 
                           key=lambda x: x[1]['test_f1'])
            best_models[dataset_name] = {
                'model': best_model[0],
                'metrics': best_model[1]
            }
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Best Model: {best_model[0]}")
            print(f"  Test Accuracy: {best_model[1]['test_accuracy']:.4f}")
            print(f"  Test F1-Score: {best_model[1]['test_f1']:.4f}")
        
        self.results['best_models'] = best_models
        
        return comparison_results
    
    def compare_privacy_modes(self):
        """Compare model performance with and without privacy enhancements"""
        print("\n" + "="*80)
        print("PRIVACY VS NON-PRIVACY COMPARISON")
        print("="*80)
        
        privacy_comparison = {}
        
        # Load and train with privacy
        print("\n--- With Privacy Enhancements ---")
        self.load_all_datasets(privacy_mode=True)
        privacy_results = self.compare_across_datasets(privacy_mode=True)
        
        # Load and train without privacy
        print("\n--- Without Privacy Enhancements ---")
        self.load_all_datasets(privacy_mode=False)
        no_privacy_results = self.compare_across_datasets(privacy_mode=False)
        
        # Compare results
        print("\n" + "="*60)
        print("PRIVACY IMPACT ANALYSIS")
        print("="*60)
        
        for dataset_name in self.dataset_names:
            if dataset_name in privacy_results and dataset_name in no_privacy_results:
                print(f"\n{dataset_name.upper()} Dataset:")
                
                privacy_comparison[dataset_name] = {}
                
                for model_name in self.models.keys():
                    privacy_metrics = privacy_results[dataset_name][model_name]
                    no_privacy_metrics = no_privacy_results[dataset_name][model_name]
                    
                    accuracy_diff = (privacy_metrics['test_accuracy'] - 
                                   no_privacy_metrics['test_accuracy']) * 100
                    f1_diff = (privacy_metrics['test_f1'] - 
                             no_privacy_metrics['test_f1']) * 100
                    
                    privacy_comparison[dataset_name][model_name] = {
                        'accuracy_impact': accuracy_diff,
                        'f1_impact': f1_diff,
                        'privacy_accuracy': privacy_metrics['test_accuracy'],
                        'no_privacy_accuracy': no_privacy_metrics['test_accuracy'],
                        'privacy_f1': privacy_metrics['test_f1'],
                        'no_privacy_f1': no_privacy_metrics['test_f1']
                    }
                    
                    print(f"  {model_name}:")
                    print(f"    Accuracy Impact: {accuracy_diff:+.2f}%")
                    print(f"    F1-Score Impact: {f1_diff:+.2f}%")
        
        self.results['privacy_comparison'] = privacy_comparison
        
        return privacy_comparison
    
    def save_results(self, save_path: str = "results"):
        """Save all results and trained models"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        with open(save_path / 'baseline_comprehensive_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary as CSV
        summary_data = []
        
        if 'comparison' in self.results:
            for dataset_name, dataset_results in self.results['comparison'].items():
                for model_name, metrics in dataset_results.items():
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Privacy_Mode': self.results.get('privacy_mode', 'Unknown'),
                        'Val_Accuracy': metrics['val_accuracy'],
                        'Val_F1': metrics['val_f1'],
                        'Test_Accuracy': metrics['test_accuracy'],
                        'Test_F1': metrics['test_f1'],
                        'Train_Time': metrics['train_time']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(save_path / 'baseline_performance_summary.csv', index=False)
        
        # Save best models info
        if 'best_models' in self.results:
            best_models_data = []
            for dataset_name, info in self.results['best_models'].items():
                best_models_data.append({
                    'Dataset': dataset_name,
                    'Best_Model': info['model'],
                    'Test_Accuracy': info['metrics']['test_accuracy'],
                    'Test_F1': info['metrics']['test_f1']
                })
            
            best_df = pd.DataFrame(best_models_data)
            best_df.to_csv(save_path / 'best_models_summary.csv', index=False)
        
        print(f"\nResults saved to {save_path}")
    
    def create_visualizations(self, save_path: str = "results/figures"):
        """Create publication-ready visualizations"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Model Performance Comparison Across Datasets
        if 'comparison' in self.results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, dataset_name in enumerate(self.dataset_names):
                if dataset_name in self.results['comparison']:
                    dataset_results = self.results['comparison'][dataset_name]
                    
                    models = list(dataset_results.keys())
                    accuracies = [dataset_results[m]['test_accuracy'] for m in models]
                    f1_scores = [dataset_results[m]['test_f1'] for m in models]
                    
                    x = np.arange(len(models))
                    width = 0.35
                    
                    axes[idx].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
                    axes[idx].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
                    
                    axes[idx].set_xlabel('Model')
                    axes[idx].set_ylabel('Score')
                    axes[idx].set_title(f'{dataset_name.upper()} Dataset')
                    axes[idx].set_xticks(x)
                    axes[idx].set_xticklabels(models, rotation=45, ha='right')
                    axes[idx].legend()
                    axes[idx].set_ylim(0, 1.05)
            
            plt.tight_layout()
            plt.savefig(save_path / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Privacy Impact Visualization
        if 'privacy_comparison' in self.results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            privacy_data = self.results['privacy_comparison']
            
            # Prepare data for heatmap
            models = list(self.models.keys())
            datasets = list(privacy_data.keys())
            
            impact_matrix = np.zeros((len(models), len(datasets)))
            
            for i, model in enumerate(models):
                for j, dataset in enumerate(datasets):
                    if dataset in privacy_data and model in privacy_data[dataset]:
                        impact_matrix[i, j] = privacy_data[dataset][model]['f1_impact']
            
            # Create heatmap
            im = ax.imshow(impact_matrix, cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10)
            
            # Set ticks
            ax.set_xticks(np.arange(len(datasets)))
            ax.set_yticks(np.arange(len(models)))
            ax.set_xticklabels([d.upper() for d in datasets])
            ax.set_yticklabels(models)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('F1-Score Impact (%)', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(datasets)):
                    text = ax.text(j, i, f'{impact_matrix[i, j]:.1f}%',
                                 ha="center", va="center", color="black" if abs(impact_matrix[i, j]) < 5 else "white")
            
            ax.set_title('Privacy Enhancement Impact on Model Performance')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Model')
            
            plt.tight_layout()
            plt.savefig(save_path / 'privacy_impact_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\nVisualizations saved to {save_path}")


def main():
    """Run the enhanced baseline evaluation pipeline"""
    import argparse
    from utils.config import Config
    
    # Load configuration
    config = Config()
    
    parser = argparse.ArgumentParser(description='Enhanced Baseline Models for Smart Grid FL')
    parser.add_argument('--mode', choices=['single', 'compare', 'privacy'], 
                       default='compare', help='Evaluation mode')
    parser.add_argument('--dataset', choices=['msu', 'pecan', 'sgcc'], 
                       help='Dataset for single mode')
    parser.add_argument('--privacy', action='store_true', 
                       default=config.get('privacy.enabled', False),
                       help='Use privacy-enhanced preprocessing')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize hyperparameters (slower)')
    
    args = parser.parse_args()
    
    # Log configuration usage
    print("Using configuration values:")
    print(f"  - Privacy enabled: {args.privacy} (config default: {config.get('privacy.enabled')})")
    print(f"  - Selection metric: {config.get('model.selection_metric', 'f1_score')}")
    print(f"  - Cross-validation folds: {config.get('model.cross_validation_folds', 5)}")
    print(f"  - Random state: {config.get('model.random_state', 42)}")
    
    # Initialize evaluator
    evaluator = MultiDatasetBaselineEvaluator()
    
    print("="*80)
    print("ENHANCED BASELINE MODEL EVALUATION")
    print("="*80)
    
    # Define models
    evaluator.define_models(optimize_hyperparams=args.optimize)
    
    if args.mode == 'single':
        # Single dataset evaluation
        if not args.dataset:
            print("Error: --dataset required for single mode")
            return
        
        evaluator.load_dataset(args.dataset, args.privacy)
        key = f"{args.dataset}_{'privacy' if args.privacy else 'no_privacy'}"
        results = evaluator.train_all_models(key)
        
    elif args.mode == 'compare':
        # Compare across all datasets
        evaluator.load_all_datasets(privacy_mode=args.privacy)
        results = evaluator.compare_across_datasets(privacy_mode=args.privacy)
        
    elif args.mode == 'privacy':
        # Compare privacy vs non-privacy
        results = evaluator.compare_privacy_modes()
    
    # Save results and create visualizations
    evaluator.save_results()
    evaluator.create_visualizations()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
