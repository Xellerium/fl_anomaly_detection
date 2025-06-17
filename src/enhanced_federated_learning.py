"""
Enhanced Federated Learning Implementation for Smart Grid Security
Fixed issues with model aggregation, privacy mechanisms, and learning dynamics
Now integrates with best baseline model automatically

Key Improvements:
1. Automatic best baseline model integration
2. Proper federated averaging for neural networks
3. Dynamic learning rates and early stopping
4. Robust differential privacy implementation
5. Better client data heterogeneity handling
6. Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import copy
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFederatedClient:
    """Enhanced federated client with improved local training and model updates"""
    
    def __init__(self, client_id: int, model_type: str = "neural_network", random_state: int = 42, model_params: Dict = None):
        self.client_id = client_id
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.local_data = None
        self.local_labels = None
        self.data_size = 0
        self.local_epochs = 3
        self.learning_rate = 0.01
        self.performance_history = []
        self.model_params = model_params or {}
        
    def load_local_data(self, X_local: pd.DataFrame, y_local: np.ndarray):
        """Load and validate local training data"""
        self.local_data = X_local.copy()
        self.local_labels = y_local.copy()
        self.data_size = len(X_local)
        
        # Analyze local data distribution
        unique, counts = np.unique(y_local, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        logger.info(f"Client {self.client_id}: Loaded {self.data_size} samples, "
                   f"class distribution: {class_dist}")
    
    def initialize_model(self, global_model=None):
        """Initialize local model with flexible architecture and optimal parameters"""
        if global_model is None:
            if self.model_type == "neural_network":
                self.model = MLPClassifier(
                    hidden_layer_sizes=self.model_params.get('hidden_layer_sizes', (100, 50)),
                    learning_rate_init=self.learning_rate,
                    max_iter=self.local_epochs,
                    random_state=self.random_state + self.client_id,
                    early_stopping=False,
                    warm_start=True,
                    alpha=self.model_params.get('alpha', 0.01)
                )
            else:  # Random Forest with optimized parameters
                self.model = RandomForestClassifier(
                    n_estimators=self.model_params.get('n_estimators', 100),
                    max_depth=self.model_params.get('max_depth', 15),
                    min_samples_split=self.model_params.get('min_samples_split', 5),
                    min_samples_leaf=self.model_params.get('min_samples_leaf', 2),
                    random_state=self.random_state + self.client_id,
                    n_jobs=1
                )
        else:
            self.model = copy.deepcopy(global_model)
    
    def local_training(self, global_round: int) -> Dict:
        """Enhanced local training with adaptive learning"""
        if self.local_data is None or self.model is None:
            raise ValueError("Local data and model must be initialized")
        
        start_time = time.time()
        
        # Adaptive learning rate based on round
        if hasattr(self.model, 'learning_rate_init'):
            self.model.learning_rate_init = max(0.001, self.learning_rate / (1 + 0.05 * global_round))
        
        # Train model on local data
        try:
            self.model.fit(self.local_data, self.local_labels)
                
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            return {'error': str(e)}
        
        # Calculate local performance metrics
        local_predictions = self.model.predict(self.local_data)
        local_accuracy = accuracy_score(self.local_labels, local_predictions)
        local_f1 = f1_score(self.local_labels, local_predictions, average='weighted')
        
        training_time = time.time() - start_time
        
        # Store performance history
        performance = {
            'round': global_round,
            'accuracy': local_accuracy,
            'f1_score': local_f1,
            'training_time': training_time,
            'data_size': self.data_size
        }
        self.performance_history.append(performance)
        
        logger.info(f"Client {self.client_id} Round {global_round}: "
                   f"Acc={local_accuracy:.4f}, F1={local_f1:.4f}")
        
        return performance
    
    def get_model_parameters(self) -> Dict:
        """Extract model parameters for aggregation"""
        if self.model_type == "neural_network" and hasattr(self.model, 'coefs_'):
            return {
                'coefs': [coef.copy() for coef in self.model.coefs_],
                'intercepts': [intercept.copy() for intercept in self.model.intercepts_],
                'data_size': self.data_size,
                'client_id': self.client_id
            }
        else:
            # For Random Forest, return feature importances and basic params
            return {
                'model': copy.deepcopy(self.model),
                'feature_importances': self.model.feature_importances_.copy() if hasattr(self.model, 'feature_importances_') else None,
                'data_size': self.data_size,
                'client_id': self.client_id,
                'predictions': None  # Will be filled during evaluation
            }
    
    def update_model_parameters(self, aggregated_params: Dict):
        """Update local model with aggregated parameters"""
        if self.model_type == "neural_network" and 'coefs' in aggregated_params:
            if hasattr(self.model, 'coefs_'):
                self.model.coefs_ = [coef.copy() for coef in aggregated_params['coefs']]
                self.model.intercepts_ = [intercept.copy() for intercept in aggregated_params['intercepts']]
        elif 'model' in aggregated_params:
            self.model = copy.deepcopy(aggregated_params['model'])

class AdvancedFederatedServer:
    """Enhanced federated server with proper model aggregation and privacy"""
    
    def __init__(self, model_type: str = "neural_network", privacy_budget: float = 1.0, 
                 random_state: int = 42, baseline_config: Dict = None):
        self.model_type = model_type
        self.global_model = None
        self.clients = []
        self.round_results = []
        self.privacy_budget = privacy_budget
        self.privacy_used = 0.0
        self.random_state = random_state
        self.best_global_performance = 0.0
        self.patience = 5
        self.no_improvement_rounds = 0
        self.baseline_config = baseline_config or {}
        self.test_predictions = None  # Store for confusion matrix generation
        
    def initialize_global_model(self, input_size: int, num_classes: int, model_params: Dict = None):
        """Initialize global model with optimal baseline configuration"""
        params = model_params or {}
        
        if self.model_type == "neural_network":
            self.global_model = MLPClassifier(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (100, 50)),
                learning_rate_init=0.01,
                max_iter=3,
                random_state=self.random_state,
                warm_start=True,
                alpha=params.get('alpha', 0.01)
            )
            # Initialize with dummy data to set up the architecture
            dummy_X = np.random.random((100, input_size))
            dummy_y = np.random.randint(0, num_classes, 100)
            self.global_model.fit(dummy_X, dummy_y)
        else:
            # Use optimal parameters from best baseline model
            self.global_model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 15),
                min_samples_split=params.get('min_samples_split', 5),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=self.random_state,
                n_jobs=-1
            )
        
        logger.info(f"Global {self.model_type} model initialized with optimal parameters")
        if params:
            logger.info(f"Using baseline configuration: {params}")
    
    def register_client(self, client):
        """Register client with validation"""
        self.clients.append(client)
        logger.info(f"Client {client.client_id} registered. Total clients: {len(self.clients)}")
    
    def federated_averaging(self, client_updates: List[Dict], apply_privacy: bool = False) -> Dict:
        """Implement proper FedAvg algorithm with optional privacy"""
        if not client_updates:
            return {}
        
        total_samples = sum(update['data_size'] for update in client_updates)
        
        if self.model_type == "neural_network":
            # Weighted averaging for neural network parameters
            if all('coefs' in update for update in client_updates):
                # Initialize aggregated parameters
                aggregated_coefs = []
                aggregated_intercepts = []
                
                # Average each layer's weights
                for layer_idx in range(len(client_updates[0]['coefs'])):
                    layer_coefs = np.zeros_like(client_updates[0]['coefs'][layer_idx])
                    layer_intercepts = np.zeros_like(client_updates[0]['intercepts'][layer_idx])
                    
                    for update in client_updates:
                        weight = update['data_size'] / total_samples
                        layer_coefs += weight * update['coefs'][layer_idx]
                        layer_intercepts += weight * update['intercepts'][layer_idx]
                    
                    # Add differential privacy noise if requested
                    if apply_privacy and self.privacy_budget > self.privacy_used:
                        noise_scale = 0.1 / (self.privacy_budget - self.privacy_used)
                        layer_coefs += np.random.laplace(0, noise_scale, layer_coefs.shape)
                        layer_intercepts += np.random.laplace(0, noise_scale, layer_intercepts.shape)
                        self.privacy_used += 0.1
                    
                    aggregated_coefs.append(layer_coefs)
                    aggregated_intercepts.append(layer_intercepts)
                
                return {
                    'coefs': aggregated_coefs,
                    'intercepts': aggregated_intercepts
                }
        
        # For Random Forest, use weighted model selection with ensemble approach
        weights = np.array([update['data_size'] for update in client_updates])
        weights = weights / np.sum(weights)
        
        if apply_privacy and self.privacy_budget > self.privacy_used:
            noise_scale = 0.1 / (self.privacy_budget - self.privacy_used)
            weights += np.random.laplace(0, noise_scale, len(weights))
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weights = weights / np.sum(weights)  # Renormalize
            self.privacy_used += 0.1
        
        # Select best performing client as global model (improved aggregation)
        performance_scores = []
        for update in client_updates:
            # Weight by both data size and performance
            score = update['f1_score'] * (update['data_size'] / total_samples)
            performance_scores.append(score)
        
        best_idx = np.argmax(performance_scores)
        return client_updates[best_idx]
    
    def federated_round(self, round_num: int, apply_privacy: bool = False) -> Dict:
        """Execute enhanced federated learning round"""
        logger.info(f"\n--- Federated Learning Round {round_num} ---")
        
        # Distribute global model to clients
        for client in self.clients:
            client.update_model_parameters(self.get_global_parameters())
        
        # Collect client updates after local training
        client_updates = []
        for client in self.clients:
            performance = client.local_training(round_num)
            if 'error' not in performance:
                model_params = client.get_model_parameters()
                model_params.update(performance)
                client_updates.append(model_params)
        
        if not client_updates:
            logger.error("No successful client updates received")
            return {}
        
        # Aggregate models
        aggregated_params = self.federated_averaging(client_updates, apply_privacy)
        self.update_global_model(aggregated_params)
        
        # Calculate round statistics
        total_samples = sum(update['data_size'] for update in client_updates)
        weighted_accuracy = sum(
            update['accuracy'] * update['data_size'] for update in client_updates
        ) / total_samples
        weighted_f1 = sum(
            update['f1_score'] * update['data_size'] for update in client_updates
        ) / total_samples
        
        # Check for improvement
        current_performance = weighted_f1
        improvement = False
        if current_performance > self.best_global_performance:
            self.best_global_performance = current_performance
            self.no_improvement_rounds = 0
            improvement = True
        else:
            self.no_improvement_rounds += 1
        
        round_stats = {
            'round': round_num,
            'avg_accuracy': weighted_accuracy,
            'avg_f1_score': weighted_f1,
            'total_samples': total_samples,
            'privacy_used': self.privacy_used,
            'privacy_applied': apply_privacy,
            'improvement': improvement
        }
        
        self.round_results.append(round_stats)
        
        logger.info(f"Round {round_num} Results:")
        logger.info(f"  Average Accuracy: {weighted_accuracy:.4f}")
        logger.info(f"  Average F1-Score: {weighted_f1:.4f}")
        logger.info(f"  Best F1 So Far: {self.best_global_performance:.4f}")
        if apply_privacy:
            logger.info(f"  Privacy Budget Used: {self.privacy_used:.3f}/{self.privacy_budget}")
        
        return round_stats
    
    def get_global_parameters(self) -> Dict:
        """Get global model parameters for distribution"""
        if self.model_type == "neural_network" and hasattr(self.global_model, 'coefs_'):
            return {
                'coefs': [coef.copy() for coef in self.global_model.coefs_],
                'intercepts': [intercept.copy() for intercept in self.global_model.intercepts_]
            }
        else:
            return {'model': copy.deepcopy(self.global_model)}
    
    def update_global_model(self, aggregated_params: Dict):
        """Update global model with aggregated parameters"""
        if self.model_type == "neural_network" and 'coefs' in aggregated_params:
            if hasattr(self.global_model, 'coefs_'):
                self.global_model.coefs_ = [coef.copy() for coef in aggregated_params['coefs']]
                self.global_model.intercepts_ = [intercept.copy() for intercept in aggregated_params['intercepts']]
        elif 'model' in aggregated_params:
            self.global_model = copy.deepcopy(aggregated_params['model'])
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early"""
        return self.no_improvement_rounds >= self.patience
    
    def evaluate_global_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """Evaluate global model on test set and store predictions"""
        try:
            if self.global_model is None:
                logger.error("Global model not initialized")
                return {'test_accuracy': 0.0, 'test_f1': 0.0}
            
            # If neural network needs fitting, train on combined client data
            if self.model_type == "neural_network" and not hasattr(self.global_model, 'coefs_'):
                all_X = []
                all_y = []
                for client in self.clients:
                    all_X.append(client.local_data)
                    all_y.append(client.local_labels)
                
                X_combined = pd.concat(all_X, ignore_index=True)
                y_combined = np.concatenate(all_y)
                self.global_model.fit(X_combined, y_combined)
            
            # Generate predictions and store for confusion matrix
            test_predictions = self.global_model.predict(X_test)
            self.test_predictions = test_predictions  # Store for later use
            
            test_accuracy = accuracy_score(y_test, test_predictions)
            test_f1 = f1_score(y_test, test_predictions, average='weighted')
            
            return {
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'predictions': test_predictions
            }
            
        except Exception as e:
            logger.error(f"Global model evaluation failed: {e}")
            return {'test_accuracy': 0.0, 'test_f1': 0.0}

class EnhancedFederatedExperiment:
    """Enhanced federated learning experiment with baseline model integration"""
    
    def __init__(self, model_type: str = "neural_network", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.server = None
        self.clients = []
        self.results = {}
        self.best_baseline_model = None
        self.model_params = {}
        
    def load_data(self, data_path: str = "data/processed") -> bool:
        """Load preprocessed data for federated learning"""
        data_path = Path(data_path)
        
        try:
            with open(data_path / 'data_splits.pkl', 'rb') as f:
                splits = pickle.load(f)
            
            # Combine train and validation for federated learning
            self.X_fed = pd.concat([splits['X_train'], splits['X_val']])
            self.y_fed = np.concatenate([splits['y_train'], splits['y_val']])
            self.X_test = splits['X_test']
            self.y_test = splits['y_test']
            
            logger.info(f"Federated learning data loaded:")
            logger.info(f"  Training data: {len(self.X_fed)} samples")
            logger.info(f"  Test data: {len(self.X_test)} samples")
            logger.info(f"  Features: {self.X_fed.shape[1]}")
            logger.info(f"  Classes: {len(np.unique(self.y_fed))}")
            
            return True
            
        except FileNotFoundError:
            logger.error("Processed data not found. Run data_pipeline.py first.")
            return False
    
    def load_best_baseline_model(self) -> bool:
        """Load the best performing baseline model configuration"""
        results_path = Path("results")
        
        try:
            # Load baseline performance summary
            baseline_df = pd.read_csv(results_path / 'baseline_performance_summary.csv')
            
            # Load baseline model objects
            with open(results_path / 'baseline_results.pkl', 'rb') as f:
                baseline_models = pickle.load(f)
            
            # Identify best model by F1-score
            best_model_name = baseline_df.loc[baseline_df['Validation F1-Score'].idxmax(), 'Model']
            best_model_key = best_model_name.replace(' ', '_')
            best_model_obj = baseline_models[best_model_key]['model']
            
            self.best_baseline_model = {
                'name': best_model_name,
                'model': best_model_obj,
                'performance': {
                    'accuracy': baseline_df.loc[baseline_df['Model'] == best_model_name, 'Validation Accuracy'].iloc[0],
                    'f1_score': baseline_df.loc[baseline_df['Model'] == best_model_name, 'Validation F1-Score'].iloc[0]
                }
            }
            
            # Extract optimal parameters
            if hasattr(best_model_obj, 'get_params'):
                self.model_params = best_model_obj.get_params()
            
            logger.info(f"Best baseline model loaded: {best_model_name}")
            logger.info(f"Performance: Acc={self.best_baseline_model['performance']['accuracy']:.4f}, "
                       f"F1={self.best_baseline_model['performance']['f1_score']:.4f}")
            
            return True
            
        except FileNotFoundError:
            logger.warning("Baseline results not found. Using default parameters.")
            return False
    
    def create_client_data_splits(self, num_clients: int = 5, non_iid: bool = False) -> List[Tuple]:
        """Create federated data splits with optional non-IID distribution"""
        if non_iid:
            # Create non-IID distribution
            client_data = []
            classes = np.unique(self.y_fed)
            
            for i in range(num_clients):
                # Each client gets 2 dominant classes
                dominant_classes = np.random.choice(classes, 2, replace=False)
                
                client_indices = []
                for cls in dominant_classes:
                    cls_indices = np.where(self.y_fed == cls)[0]
                    n_samples = len(cls_indices) // num_clients
                    selected = np.random.choice(cls_indices, n_samples, replace=False)
                    client_indices.extend(selected)
                
                # Add some samples from other classes
                other_classes = [c for c in classes if c not in dominant_classes]
                for cls in other_classes:
                    cls_indices = np.where(self.y_fed == cls)[0]
                    n_samples = len(cls_indices) // (num_clients * 3)
                    if n_samples > 0:
                        selected = np.random.choice(cls_indices, n_samples, replace=False)
                        client_indices.extend(selected)
                
                client_indices = np.array(client_indices)
                X_client = self.X_fed.iloc[client_indices].reset_index(drop=True)
                y_client = self.y_fed[client_indices]
                client_data.append((X_client, y_client))
                
        else:
            # IID distribution
            indices = np.random.RandomState(self.random_state).permutation(len(self.X_fed))
            client_indices = np.array_split(indices, num_clients)
            
            client_data = []
            for i, client_idx in enumerate(client_indices):
                X_client = self.X_fed.iloc[client_idx].reset_index(drop=True)
                y_client = self.y_fed[client_idx]
                client_data.append((X_client, y_client))
        
        return client_data
    
    def setup_federated_system_with_best_baseline(self, num_clients: int = 5, privacy_budget: float = 1.0, 
                                                 non_iid: bool = False):
        """Initialize federated system using the best baseline model configuration"""
        logger.info(f"Setting up federated learning system with baseline integration...")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Data distribution: {'Non-IID' if non_iid else 'IID'}")
        
        # Load best baseline model configuration
        self.load_best_baseline_model()
        
        # Initialize server with baseline configuration
        self.server = AdvancedFederatedServer(
            model_type=self.model_type,
            privacy_budget=privacy_budget, 
            random_state=self.random_state,
            baseline_config=self.model_params
        )
        
        # Create client data splits
        client_data_splits = self.create_client_data_splits(num_clients, non_iid)
        
        # Initialize global model with optimal configuration
        num_features = self.X_fed.shape[1]
        num_classes = len(np.unique(self.y_fed))
        self.server.initialize_global_model(num_features, num_classes, self.model_params)
        
        # Initialize and register clients with optimal parameters
        self.clients = []
        for i, (X_client, y_client) in enumerate(client_data_splits):
            client = AdvancedFederatedClient(
                client_id=i, 
                model_type=self.model_type,
                random_state=self.random_state,
                model_params=self.model_params
            )
            client.load_local_data(X_client, y_client)
            client.initialize_model()
            
            self.server.register_client(client)
            self.clients.append(client)
        
        logger.info(f"Federated system initialized with {len(self.clients)} clients using optimal baseline configuration")
    
    def run_federated_training(self, num_rounds: int = 15, apply_privacy: bool = False) -> List[Dict]:
        """Execute enhanced federated learning training"""
        logger.info(f"\nStarting federated training for {num_rounds} rounds...")
        logger.info(f"Privacy protection: {'Enabled' if apply_privacy else 'Disabled'}")
        if self.best_baseline_model:
            logger.info(f"Using configuration from: {self.best_baseline_model['name']}")
        
        for round_num in range(1, num_rounds + 1):
            round_results = self.server.federated_round(round_num, apply_privacy)
            
            # Evaluate on test set every 3 rounds or on final round
            if round_num % 3 == 0 or round_num == num_rounds:
                global_eval = self.server.evaluate_global_model(self.X_test, self.y_test)
                round_results.update(global_eval)
                logger.info(f"  Global Test Accuracy: {global_eval['test_accuracy']:.4f}")
                logger.info(f"  Global Test F1-Score: {global_eval['test_f1']:.4f}")
            
            # Early stopping check
            if self.server.should_stop_early():
                logger.info(f"Early stopping triggered after {round_num} rounds")
                break
        
        return self.server.round_results
    
    def run_comparative_analysis(self) -> Dict:
        """Run both federated approaches and compare with baseline"""
        logger.info("="*70)
        logger.info("COMPARATIVE FEDERATED LEARNING ANALYSIS")
        logger.info("="*70)
        
        results = {}
        
        # Standard federated learning with best baseline model
        logger.info("\n1. STANDARD FEDERATED LEARNING (IID) WITH BEST BASELINE CONFIG")
        logger.info("="*60)
        
        self.setup_federated_system_with_best_baseline(num_clients=5, privacy_budget=1.0, non_iid=False)
        standard_results = self.run_federated_training(num_rounds=15, apply_privacy=False)
        results['standard_iid_optimized'] = standard_results
        
        # Non-IID federated learning
        logger.info("\n2. NON-IID FEDERATED LEARNING WITH BEST BASELINE CONFIG")
        logger.info("="*60)
        
        self.setup_federated_system_with_best_baseline(num_clients=5, privacy_budget=1.0, non_iid=True)
        non_iid_results = self.run_federated_training(num_rounds=15, apply_privacy=False)
        results['non_iid_optimized'] = non_iid_results
        
        # Privacy-preserving experiments
        logger.info("\n3. PRIVACY-PRESERVING FEDERATED LEARNING")
        logger.info("="*60)
        
        privacy_results = {}
        for epsilon in [0.5, 1.0, 5.0]:
            logger.info(f"\n--- Privacy Experiment: ε = {epsilon} ---")
            
            self.setup_federated_system_with_best_baseline(num_clients=5, privacy_budget=epsilon, non_iid=False)
            privacy_training = self.run_federated_training(num_rounds=10, apply_privacy=True)
            privacy_results[f"epsilon_{epsilon}"] = privacy_training
        
        results['privacy_experiments'] = privacy_results
        
        # Generate comparison with baseline
        comparison = self.compare_with_baseline()
        results['baseline_comparison'] = comparison
        
        # Save results
        self.save_results(results)
        
        logger.info("\n" + "="*70)
        logger.info("COMPARATIVE ANALYSIS COMPLETED")
        logger.info("="*70)
        
        return results
    
    def compare_with_baseline(self) -> Dict:
        """Compare federated results with best baseline model"""
        if not self.best_baseline_model:
            logger.warning("No baseline model available for comparison")
            return {}
        
        # Get final federated performance
        final_eval = self.server.evaluate_global_model(self.X_test, self.y_test)
        
        baseline_acc = self.best_baseline_model['performance']['accuracy']
        baseline_f1 = self.best_baseline_model['performance']['f1_score']
        federated_acc = final_eval['test_accuracy']
        federated_f1 = final_eval['test_f1']
        
        # Calculate performance gaps
        accuracy_gap = baseline_acc - federated_acc
        f1_gap = baseline_f1 - federated_f1
        
        comparison = {
            'baseline': {
                'model': self.best_baseline_model['name'],
                'accuracy': baseline_acc,
                'f1_score': baseline_f1
            },
            'federated': {
                'accuracy': federated_acc,
                'f1_score': federated_f1,
                'predictions': final_eval.get('predictions')
            },
            'performance_gaps': {
                'accuracy_gap': accuracy_gap,
                'f1_gap': f1_gap,
                'accuracy_retention': (federated_acc / baseline_acc) * 100,
                'f1_retention': (federated_f1 / baseline_f1) * 100
            }
        }
        
        logger.info(f"\nPerformance Comparison:")
        logger.info(f"  Baseline ({self.best_baseline_model['name']}): Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
        logger.info(f"  Federated Learning: Acc={federated_acc:.4f}, F1={federated_f1:.4f}")
        logger.info(f"  Performance Retention: Acc={comparison['performance_gaps']['accuracy_retention']:.1f}%, F1={comparison['performance_gaps']['f1_retention']:.1f}%")
        
        return comparison
    
    def save_results(self, results: Dict):
        """Save comprehensive experiment results"""
        results_path = Path("results/federated_learning")
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        with open(results_path / 'enhanced_federated_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Save training history separately for easier access
        if 'standard_iid_optimized' in results:
            with open(results_path / 'training_history.pkl', 'wb') as f:
                pickle.dump(results['standard_iid_optimized'], f)
        
        # Save comparison results for visualization
        if 'baseline_comparison' in results:
            with open(results_path / 'federated_comparison.pkl', 'wb') as f:
                pickle.dump(results['baseline_comparison'], f)
        
        # Generate summary text
        summary = self.generate_experiment_summary(results)
        with open(results_path / 'experiment_summary.txt', 'w') as f:
            f.write(summary)
        
        logger.info(f"Enhanced federated learning results saved to {results_path}")
    
    def generate_experiment_summary(self, results: Dict) -> str:
        """Generate comprehensive experiment summary"""
        summary = "ENHANCED FEDERATED LEARNING EXPERIMENT SUMMARY\n"
        summary += "="*60 + "\n\n"
        
        if self.best_baseline_model:
            summary += f"BASELINE MODEL INTEGRATION:\n"
            summary += f"- Best Model: {self.best_baseline_model['name']}\n"
            summary += f"- Baseline Performance: Acc={self.best_baseline_model['performance']['accuracy']:.4f}, "
            summary += f"F1={self.best_baseline_model['performance']['f1_score']:.4f}\n"
            summary += f"- Configuration Applied: {len(self.model_params)} parameters\n\n"
        
        for experiment_name, experiment_results in results.items():
            if isinstance(experiment_results, list) and experiment_results:
                final_result = experiment_results[-1]
                summary += f"{experiment_name.upper().replace('_', ' ')}:\n"
                summary += f"  Rounds Completed: {len(experiment_results)}\n"
                summary += f"  Final Accuracy: {final_result.get('avg_accuracy', 'N/A'):.4f}\n"
                summary += f"  Final F1-Score: {final_result.get('avg_f1_score', 'N/A'):.4f}\n"
                if 'test_accuracy' in final_result:
                    summary += f"  Test Accuracy: {final_result['test_accuracy']:.4f}\n"
                    summary += f"  Test F1-Score: {final_result['test_f1']:.4f}\n"
                summary += "\n"
            elif experiment_name == 'baseline_comparison' and experiment_results:
                summary += "BASELINE COMPARISON:\n"
                comp = experiment_results
                summary += f"  Performance Retention: {comp['performance_gaps']['accuracy_retention']:.1f}% accuracy, {comp['performance_gaps']['f1_retention']:.1f}% F1\n"
                summary += f"  Privacy Cost: {comp['performance_gaps']['accuracy_gap']:.4f} accuracy, {comp['performance_gaps']['f1_gap']:.4f} F1\n\n"
        
        return summary
    
    def run_complete_experiment(self):
        """Execute complete enhanced federated learning experiment pipeline"""
        logger.info("="*70)
        logger.info("ENHANCED FEDERATED LEARNING EXPERIMENT FOR SMART GRID SECURITY")
        logger.info("="*70)
        
        # Load data
        if not self.load_data():
            return {}
        
        # Run comparative analysis
        results = self.run_comparative_analysis()
        
        return results

    # Main execution function
def main():
    """Run enhanced federated learning experiment with baseline integration"""
    # Run with neural networks (recommended for better aggregation)
    experiment_nn = EnhancedFederatedExperiment(model_type="neural_network")
    results_nn = experiment_nn.run_complete_experiment()
    
    # Run with Random Forest using best baseline configuration
    experiment_rf = EnhancedFederatedExperiment(model_type="random_forest")
    results_rf = experiment_rf.run_complete_experiment()

if __name__ == "__main__":
    main()