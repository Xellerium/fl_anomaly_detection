# baseline_models.py
"""
Streamlined Baseline Models for Smart Grid Anomaly Detection
Implements modern ML algorithms for centralized learning performance benchmarking

Key Models Evaluated:
1. Random Forest - Tree-based ensemble method
2. XGBoost - Gradient boosting framework  
3. LightGBM - Microsoft's gradient boosting
4. CatBoost - Yandex's gradient boosting
5. Logistic Regression - Linear baseline

For PhD Supervision Discussion:
- Compares 5 state-of-the-art ML algorithms
- Uses cross-validation for robust evaluation
- Implements proper train/validation/test methodology
- Generates publication-ready performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BaselineEvaluator:
    """Handles training and evaluation of baseline ML models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.data_loaded = False
        
    def load_processed_data(self, data_path="data/processed"):
        """Load preprocessed data from data pipeline"""
        data_path = Path(data_path)
        
        try:
            with open(data_path / 'data_splits.pkl', 'rb') as f:
                splits = pickle.load(f)
            
            self.X_train = splits['X_train']
            self.X_val = splits['X_val']
            self.X_test = splits['X_test']
            self.y_train = splits['y_train']
            self.y_val = splits['y_val']
            self.y_test = splits['y_test']
            
            with open(data_path / 'label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.class_names = list(self.label_encoder.classes_)
            self.data_loaded = True
            
            print(f"Data loaded successfully:")
            print(f"  Training: {len(self.X_train):,} samples")
            print(f"  Validation: {len(self.X_val):,} samples")
            print(f"  Test: {len(self.X_test):,} samples")
            print(f"  Classes: {self.class_names}")
            
            return True
            
        except FileNotFoundError:
            print("Error: Processed data not found. Run data_pipeline.py first.")
            return False
    
    def define_models(self):
        """Define optimized baseline models for comparison"""
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
                verbosity=0
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1,
                force_col_wise=True
            ),
            
            'CatBoost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            ),
            
            'Logistic_Regression': LogisticRegression(
                max_iter=500,
                random_state=self.random_state,
                solver='liblinear'
            )
        }
        
        print(f"Defined {len(self.models)} baseline models for evaluation")
    
    def train_and_evaluate_models(self):
        """Train all models and evaluate performance"""
        if not self.data_loaded:
            print("Error: No data loaded. Call load_processed_data() first.")
            return
        
        print("\nTraining and evaluating baseline models...")
        print("="*50)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Record training time
            start_time = time.time()
            
            # Train on training data
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Evaluate on validation set
            start_time = time.time()
            y_pred = model.predict(self.X_val)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'y_pred': y_pred,
                'y_true': self.y_val
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Training Time: {training_time:.2f}s")
    
    def cross_validate_best_models(self, top_k=3):
        """Perform cross-validation on best performing models"""
        if not self.results:
            print("Error: No results available. Train models first.")
            return
        
        # Sort models by F1-score and select top performers
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        top_models = sorted_models[:top_k]
        
        print(f"\nPerforming cross-validation on top {top_k} models...")
        
        # Combine train and validation for cross-validation
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = np.concatenate([self.y_train, self.y_val])
        
        for model_name, model_data in top_models:
            model = self.models[model_name]
            
            # 5-fold cross-validation
            cv_scores = cross_val_score(model, X_combined, y_combined, cv=5, scoring='f1_weighted')
            
            self.results[model_name]['cv_mean'] = cv_scores.mean()
            self.results[model_name]['cv_std'] = cv_scores.std()
            
            print(f"  {model_name}: CV F1 = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    def evaluate_best_model_on_test(self):
        """Evaluate the best model on held-out test set"""
        if not self.results:
            print("Error: No results available. Train models first.")
            return
        
        # Find best model by F1-score
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_model = self.results[best_model_name]['model']
        
        print(f"\nEvaluating best model ({best_model_name}) on test set...")
        
        # Final evaluation on test set
        test_predictions = best_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        test_f1 = f1_score(self.y_test, test_predictions, average='weighted')
        
        self.results[best_model_name]['test_accuracy'] = test_accuracy
        self.results[best_model_name]['test_f1'] = test_f1
        
        print(f"Final test performance:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, test_predictions, target_names=self.class_names))
        
        return best_model_name, test_accuracy, test_f1
    
    def generate_performance_summary(self):
        """Create comprehensive performance summary"""
        if not self.results:
            print("Error: No results available.")
            return
        
        print(f"\n{'='*70}")
        print("BASELINE MODEL PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        # Create summary table
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name.replace('_', ' '),
                'Validation Accuracy': f"{results['accuracy']:.4f}",
                'Validation F1-Score': f"{results['f1_score']:.4f}",
                'Training Time (s)': f"{results['training_time']:.2f}",
                'CV Mean': f"{results.get('cv_mean', 'N/A'):.4f}" if results.get('cv_mean') else 'N/A'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Validation F1-Score', ascending=False)
        
        print(summary_df.to_string(index=False))
        
        # Save summary
        results_path = Path("results")
        results_path.mkdir(exist_ok=True)
        summary_df.to_csv(results_path / 'baseline_performance_summary.csv', index=False)
        
        return summary_df
    
    def save_models(self):
        """Save trained models and results"""
        models_path = Path("results/models")
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, result_data in self.results.items():
            model_file = models_path / f'{model_name.lower()}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(result_data['model'], f)
        
        # Save complete results
        with open(Path("results") / 'baseline_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Models and results saved to results/ directory")
    
    def run_complete_evaluation(self):
        """Execute complete baseline evaluation pipeline"""
        print("="*70)
        print("BASELINE MODEL EVALUATION PIPELINE")
        print("="*70)
        
        # Load data
        if not self.load_processed_data():
            return
        
        # Define models
        self.define_models()
        
        # Train and evaluate
        self.train_and_evaluate_models()
        
        # Cross-validate best models
        self.cross_validate_best_models()
        
        # Final test evaluation
        best_model, test_acc, test_f1 = self.evaluate_best_model_on_test()
        
        # Generate summary
        self.generate_performance_summary()
        
        # Save results
        self.save_models()
        
        print(f"\n{'='*70}")
        print("BASELINE EVALUATION COMPLETED")
        print(f"Best Model: {best_model}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print(f"{'='*70}")
        
        return self.results

# Main execution function
def main():
    """Run complete baseline model evaluation"""
    evaluator = BaselineEvaluator()
    results = evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
