"""
PriFed-GridGuard: Privacy-Enhanced Federated Learning for Smart Grid Security
Implements advanced privacy mechanisms for federated learning with multi-dataset support

Privacy Mechanisms:
1. Context-Aware Local Differential Privacy (CA-LDP)
2. Cluster-Adaptive Differential Privacy (CADP) 
3. Selective Homomorphic Encryption (S-HE)
4. Utility-Aware Noise Scheduler (UANS)

Features:
- Multi-dataset support (MSU, Pecan Street, SGCC)
- Automatic best baseline model integration
- Privacy-utility trade-off analysis
- Publication-ready results generation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import copy
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict
import hashlib

# Try to import PyTorch for GPU support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"[SUCCESS] CUDA is available - GPU acceleration enabled for neural networks")
        DEVICE = torch.device('cuda')
    else:
        print(f"✗ CUDA not available - using CPU for neural networks")
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = None
    print("✗ PyTorch not installed - using sklearn MLPClassifier")

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# PyTorch Neural Network for GPU acceleration
if TORCH_AVAILABLE:
    class SmartGridNN(nn.Module):
        """PyTorch neural network for smart grid classification with GPU support"""
        
        def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...], num_classes: int):
            super(SmartGridNN, self).__init__()
            
            layers = []
            prev_size = input_size
            
            # Build hidden layers
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, num_classes))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)


class PrivacyMechanisms:
    """Implements privacy mechanisms for PriFed-GridGuard"""
    
    @staticmethod
    def add_ca_ldp_noise(data: np.ndarray, feature_sensitivities: Dict[int, float], 
                        epsilon: float) -> np.ndarray:
        """Context-Aware Local Differential Privacy - adds varying noise based on feature sensitivity"""
        noisy_data = data.copy()
        
        for i in range(data.shape[1]):
            sensitivity = feature_sensitivities.get(i, 0.5)  # Default medium sensitivity
            
            # Scale noise based on sensitivity (higher sensitivity = more noise)
            noise_scale = sensitivity / epsilon
            
            # Add Laplace noise
            noise = np.random.laplace(0, noise_scale, size=data.shape[0])
            noisy_data[:, i] += noise
        
        return noisy_data
    
    @staticmethod
    def selective_homomorphic_encryption(params: Dict, sensitive_dims: List[int], 
                                       key: str = "default") -> Dict:
        """Selective Homomorphic Encryption - encrypts only sensitive dimensions"""
        encrypted_params = copy.deepcopy(params)
        
        # Simple encryption simulation (in practice, use real HE library like TenSEAL)
        for dim in sensitive_dims:
            if 'coefs' in params:  # Neural network
                for i, coef in enumerate(params['coefs']):
                    if dim < coef.shape[0]:
                        # Simulate encryption by hashing
                        encrypted_params['coefs'][i][dim] = hash(str(coef[dim]) + key) % 1e6
            elif 'feature_importances' in params:  # Random Forest
                if dim < len(params['feature_importances']):
                    encrypted_params['feature_importances'][dim] = hash(
                        str(params['feature_importances'][dim]) + key
                    ) % 1e6
        
        encrypted_params['encrypted_dims'] = sensitive_dims
        return encrypted_params
    
    @staticmethod
    def decrypt_selective_params(encrypted_params: Dict, key: str = "default") -> Dict:
        """Decrypt selectively encrypted parameters (simulation)"""
        # In practice, would use real HE decryption
        # For simulation, we'll just mark as decrypted
        decrypted_params = copy.deepcopy(encrypted_params)
        if 'encrypted_dims' in decrypted_params:
            del decrypted_params['encrypted_dims']
        return decrypted_params


class PrivacyEnhancedFederatedClient:
    """Enhanced federated client with privacy mechanisms"""
    
    def __init__(self, client_id: int, cluster_id: int = 0, model_type: str = "neural_network", 
                 random_state: int = 42, privacy_config: Dict = None):
        self.client_id = client_id
        self.cluster_id = cluster_id
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.local_data = None
        self.local_labels = None
        self.data_size = 0
        self.local_epochs = 30
        self.learning_rate = 0.01
        self.performance_history = []
        
        # Privacy configuration
        self.privacy_config = privacy_config or {
            'ca_ldp_enabled': False,
            'epsilon': 1.0,
            's_he_enabled': False,
            'sensitive_features_ratio': 0.3
        }
        
        # Feature sensitivities (will be computed based on data)
        self.feature_sensitivities = {}
        
    def load_local_data(self, X_local: pd.DataFrame, y_local: np.ndarray):
        """Load local data and compute feature sensitivities"""
        self.local_data = X_local.copy()
        self.local_labels = y_local.copy()
        self.data_size = len(X_local)
        
        # Compute feature sensitivities based on variance and correlation with labels
        self._compute_feature_sensitivities()
        
        # Log data distribution
        unique, counts = np.unique(y_local, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Client {self.client_id} (Cluster {self.cluster_id}): "
                   f"{self.data_size} samples, distribution: {class_dist}")
    
    def _compute_feature_sensitivities(self):
        """Compute sensitivity scores for each feature"""
        X = self.local_data.values if isinstance(self.local_data, pd.DataFrame) else self.local_data
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute correlation with labels
        correlations = []
        for i in range(X_scaled.shape[1]):
            corr = np.corrcoef(X_scaled[:, i], self.local_labels)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # Compute variance
        variances = np.var(X_scaled, axis=0)
        
        # Combine into sensitivity scores (high correlation + high variance = high sensitivity)
        max_corr = max(correlations) if correlations else 1
        max_var = max(variances) if len(variances) > 0 else 1
        
        for i in range(X.shape[1]):
            corr_score = correlations[i] / max_corr if max_corr > 0 else 0
            var_score = variances[i] / max_var if max_var > 0 else 0
            self.feature_sensitivities[i] = (corr_score + var_score) / 2
    
    def initialize_model(self, global_model=None, model_params: Dict = None):
        """Initialize model with privacy-aware configuration"""
        if global_model is None:
            if self.model_type == "neural_network":
                # Check if we have all classes in our local data
                unique_classes = np.unique(self.local_labels) if hasattr(self, 'local_labels') else []
                warm_start = len(unique_classes) == 3 if len(unique_classes) > 0 else False
                
                self.model = MLPClassifier(
                    hidden_layer_sizes=model_params.get('hidden_layer_sizes', (100, 50)),
                    learning_rate_init=self.learning_rate,
                    max_iter=self.local_epochs,
                    random_state=self.random_state + self.client_id,
                    early_stopping=False,
                    warm_start=warm_start,
                    alpha=model_params.get('alpha', 0.01)
                )
            else:  # Random Forest
                self.model = RandomForestClassifier(
                    n_estimators=model_params.get('n_estimators', 100),
                    max_depth=model_params.get('max_depth', 15),
                    min_samples_split=model_params.get('min_samples_split', 5),
                    random_state=self.random_state + self.client_id,
                    n_jobs=1
                )
        else:
            self.model = copy.deepcopy(global_model)
    
    def local_training_with_privacy(self, global_round: int, cluster_epsilon: float = None) -> Dict:
        """Local training with CA-LDP privacy enhancement"""
        if self.local_data is None or self.model is None:
            raise ValueError("Local data and model must be initialized")
        
        start_time = time.time()
        
        # Use cluster-specific epsilon if provided (CADP)
        epsilon = cluster_epsilon if cluster_epsilon else self.privacy_config['epsilon']
        
        # Apply CA-LDP if enabled
        if self.privacy_config.get('ca_ldp_enabled', False):
            X_train = PrivacyMechanisms.add_ca_ldp_noise(
                self.local_data.values, 
                self.feature_sensitivities,
                epsilon
            )
        else:
            X_train = self.local_data.values
        
        # Adaptive learning rate
        if hasattr(self.model, 'learning_rate_init'):
            self.model.learning_rate_init = max(0.001, self.learning_rate / (1 + 0.05 * global_round))
        
        # Train model
        try:
            self.model.fit(X_train, self.local_labels)
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            return {'error': str(e)}
        
        # Evaluate on original data (without noise)
        predictions = self.model.predict(self.local_data)
        accuracy = accuracy_score(self.local_labels, predictions)
        f1 = f1_score(self.local_labels, predictions, average='weighted')
        
        training_time = time.time() - start_time
        
        performance = {
            'round': global_round,
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'data_size': self.data_size,
            'cluster_id': self.cluster_id,
            'epsilon_used': epsilon
        }
        
        self.performance_history.append(performance)
        logger.info(f"Client {self.client_id} Round {global_round}: "
                   f"Acc={accuracy:.4f}, F1={f1:.4f}, ε={epsilon:.2f}")
        
        return performance
    
    def get_encrypted_parameters(self) -> Dict:
        """Get model parameters with selective encryption"""
        params = self._extract_model_parameters()
        
        if self.privacy_config.get('s_he_enabled', False):
            # Determine sensitive dimensions
            n_features = len(self.feature_sensitivities)
            n_sensitive = int(n_features * self.privacy_config.get('sensitive_features_ratio', 0.3))
            
            # Select most sensitive features
            sorted_features = sorted(self.feature_sensitivities.items(), 
                                   key=lambda x: x[1], reverse=True)
            sensitive_dims = [feat[0] for feat in sorted_features[:n_sensitive]]
            
            # Apply selective encryption
            params = PrivacyMechanisms.selective_homomorphic_encryption(
                params, sensitive_dims, key=f"client_{self.client_id}"
            )
        
        return params
    
    def _extract_model_parameters(self) -> Dict:
        """Extract model parameters for aggregation"""
        if self.model_type == "neural_network" and hasattr(self.model, 'coefs_'):
            return {
                'coefs': [coef.copy() for coef in self.model.coefs_],
                'intercepts': [intercept.copy() for intercept in self.model.intercepts_],
                'data_size': self.data_size,
                'client_id': self.client_id,
                'cluster_id': self.cluster_id
            }
        else:
            return {
                'model': copy.deepcopy(self.model),
                'feature_importances': self.model.feature_importances_.copy() if hasattr(self.model, 'feature_importances_') else None,
                'data_size': self.data_size,
                'client_id': self.client_id,
                'cluster_id': self.cluster_id
            }
    
    def update_model_parameters(self, aggregated_params: Dict):
        """Update local model with aggregated parameters"""
        if self.model_type == "neural_network" and 'coefs' in aggregated_params:
                self.model.coefs_ = [coef.copy() for coef in aggregated_params['coefs']]
                self.model.intercepts_ = [intercept.copy() for intercept in aggregated_params['intercepts']]
        elif 'model' in aggregated_params:
            self.model = copy.deepcopy(aggregated_params['model'])


class ClusterAdaptivePrivacyManager:
    """Implements Cluster-Adaptive Differential Privacy (CADP)"""
    
    def __init__(self, n_clusters: int = 3, base_epsilon: float = 1.0):
        self.n_clusters = n_clusters
        self.base_epsilon = base_epsilon
        self.cluster_epsilons = {}
        self.cluster_characteristics = {}
    
    def cluster_clients(self, client_data_stats: List[Dict]) -> Dict[int, int]:
        """Cluster clients based on data characteristics"""
        # Extract features for clustering
        features = []
        client_ids = []
        
        for stats in client_data_stats:
            client_ids.append(stats['client_id'])
            features.append([
                stats['data_size'],
                stats['n_classes'],
                stats.get('avg_feature_sensitivity', 0.5)
            ])
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Create client-cluster mapping
        client_clusters = dict(zip(client_ids, cluster_labels))
        
        # Analyze cluster characteristics
        for cluster_id in range(self.n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_features = [features[i] for i in cluster_indices]
            
            if cluster_features:
                avg_features = np.mean(cluster_features, axis=0)
                self.cluster_characteristics[cluster_id] = {
                    'avg_data_size': avg_features[0],
                    'avg_n_classes': avg_features[1],
                    'avg_sensitivity': avg_features[2],
                    'n_clients': len(cluster_indices)
                }
        
        # Assign adaptive epsilon values
        self._assign_cluster_epsilons()
        
        return client_clusters
    
    def _assign_cluster_epsilons(self):
        """Assign epsilon values based on cluster characteristics"""
        for cluster_id, chars in self.cluster_characteristics.items():
            # Higher privacy budget for clusters with:
            # - Larger datasets (more robust to noise)
            # - Lower sensitivity scores
            # - Better class balance
            
            size_factor = min(2.0, chars['avg_data_size'] / 1000)  # Normalize by 1000 samples
            sensitivity_factor = 1.0 - chars['avg_sensitivity']
            
            # Weighted combination
            adjustment = (size_factor * 0.4 + sensitivity_factor * 0.6) # Adjusted weights
            
            # Scale epsilon
            self.cluster_epsilons[cluster_id] = self.base_epsilon * max(0.5, min(2.0, adjustment))
            
            logger.info(f"Cluster {cluster_id}: ε = {self.cluster_epsilons[cluster_id]:.2f}, "
                       f"clients = {chars['n_clients']}")


class UtilityAwareNoiseScheduler:
    """Implements Utility-Aware Noise Scheduler (UANS)"""
    
    def __init__(self, total_rounds: int, base_epsilon: float, 
                 privacy_budget: float = 10.0):
        self.total_rounds = total_rounds
        self.base_epsilon = base_epsilon
        self.privacy_budget = privacy_budget
        self.round_epsilons = {}
        self.performance_history = []
        
        # Initialize round-specific epsilons
        self._initialize_schedule()
    
    def _initialize_schedule(self):
        """Initialize privacy budget schedule across rounds"""
        # Allocate more budget to early rounds (critical for convergence)
        # Use exponential decay
        
        weights = []
        for r in range(self.total_rounds):
            weight = np.exp(-0.1 * r)  # Exponential decay
            weights.append(weight)
        
        # Normalize weights to sum to privacy budget
        total_weight = sum(weights)
        normalized_weights = [w / total_weight * self.privacy_budget for w in weights]
        
        # Assign epsilon values
        for r in range(self.total_rounds):
            self.round_epsilons[r] = max(0.1, normalized_weights[r])
    
    def get_round_epsilon(self, round_num: int, performance_delta: float = None) -> float:
        """Get epsilon for current round with optional performance-based adjustment"""
        base_epsilon = self.round_epsilons.get(round_num, self.base_epsilon)
        
        # Adjust based on performance improvement
        if performance_delta is not None and len(self.performance_history) > 2:
            # If performance is plateauing, we can use less privacy budget
            if abs(performance_delta) < 0.001:  # Less than 0.1% improvement
                base_epsilon *= 0.8
            elif performance_delta > 0.01:  # More than 1% improvement
                base_epsilon *= 1.1  # Allow slightly more budget for momentum
        
        # Record actual epsilon used
        self.performance_history.append({
            'round': round_num,
            'epsilon': base_epsilon,
            'performance_delta': performance_delta
        })
        
        return base_epsilon
    
    def get_remaining_budget(self, current_round: int) -> float:
        """Calculate remaining privacy budget"""
        used_budget = sum(self.round_epsilons[r] for r in range(current_round))
        return max(0, self.privacy_budget - used_budget)


class PriFedGridGuardServer:
    """Privacy-Enhanced Federated Server with all privacy mechanisms"""
    
    def __init__(self, dataset_name: str, n_clients: int = 10, 
                 privacy_config: Dict = None, random_state: int = 42):
        self.dataset_name = dataset_name
        self.n_clients = n_clients
        self.random_state = random_state
        self.global_model = None
        self.clients = []
        self.round_results = []
        self.best_baseline_config = None
        
        # Privacy configuration
        self.privacy_config = privacy_config or {
            'ca_ldp': True,
            'cadp': True,
            's_he': True,
            'uans': True,
            'base_epsilon': 1.0,
            'total_privacy_budget': 10.0,
            'n_clusters': 3
        }
        
        # Initialize privacy managers
        self.cluster_manager = ClusterAdaptivePrivacyManager(
            n_clusters=self.privacy_config['n_clusters'],
            base_epsilon=self.privacy_config['base_epsilon']
        ) if self.privacy_config['cadp'] else None
        
        self.noise_scheduler = UtilityAwareNoiseScheduler(
            total_rounds=20,  # Will be updated
            base_epsilon=self.privacy_config['base_epsilon'],
            privacy_budget=self.privacy_config['total_privacy_budget']
        ) if self.privacy_config['uans'] else None
        
        logger.info(f"Initialized PriFed-GridGuard Server for {dataset_name}")
    
    def load_dataset_and_baseline(self, data_path: str = "data/processed"):
        """Load dataset and best baseline model configuration"""
        # Load processed data
        dataset_path = Path(data_path) / self.dataset_name
        
        with open(dataset_path / 'data_splits.pkl', 'rb') as f:
            self.data_splits = pickle.load(f)
        
        # Load best baseline results if available
        baseline_path = Path("results") / "best_models_summary.csv"
        if baseline_path.exists():
            import pandas as pd
            best_models_df = pd.read_csv(baseline_path)
            
            # Find best model for this dataset
            dataset_row = best_models_df[best_models_df['Dataset'] == self.dataset_name]
            if not dataset_row.empty:
                self.best_baseline_config = {
                    'model_type': dataset_row.iloc[0]['Best_Model'],
                    'accuracy': dataset_row.iloc[0]['Test_Accuracy'],
                    'f1_score': dataset_row.iloc[0]['Test_F1']
                }
                logger.info(f"Loaded best baseline: {self.best_baseline_config['model_type']} "
                           f"(Acc: {self.best_baseline_config['accuracy']:.4f})")
    
    def create_federated_clients(self, data_distribution: str = "iid"):
        """Create federated clients with data partitioning"""
        X_train = self.data_splits['X_train']
        y_train = self.data_splits['y_train']
        
        # Analyze client data for CADP
        client_data_stats = []
        
        if data_distribution == "iid":
            # IID distribution
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            
            split_indices = np.array_split(indices, self.n_clients)
            
            for i in range(self.n_clients):
                client_indices = split_indices[i]
                X_client = X_train.iloc[client_indices]
                y_client = y_train[client_indices]
                
                # Compute client statistics
                unique, counts = np.unique(y_client, return_counts=True)
                class_imbalance = 1 - (min(counts) / max(counts)) if len(counts) > 1 else 0
                
                client_data_stats.append({
                    'client_id': i,
                    'data_size': len(client_indices),
                    'n_classes': len(unique),
                    'class_imbalance': class_imbalance,
                    'indices': client_indices
                })
        else:
            # Non-IID distribution (label-based partitioning)
            # Sort by labels
            sorted_indices = np.argsort(y_train)
            
            # Create shards with dominant labels
            n_shards = self.n_clients * 2
            shard_size = len(y_train) // n_shards
            shards = []
            
            for i in range(n_shards):
                start_idx = i * shard_size
                end_idx = start_idx + shard_size if i < n_shards - 1 else len(y_train)
                shards.append(sorted_indices[start_idx:end_idx])
            
            # Assign shards to clients
            np.random.shuffle(shards)
            for i in range(self.n_clients):
                # Each client gets 2 shards
                client_indices = np.concatenate([
                    shards[i * 2], 
                    shards[i * 2 + 1] if i * 2 + 1 < len(shards) else []
                ])
                
                X_client = X_train.iloc[client_indices]
                y_client = y_train[client_indices]
                
                # Compute statistics
                unique, counts = np.unique(y_client, return_counts=True)
                class_imbalance = 1 - (min(counts) / max(counts)) if len(counts) > 1 else 0
                
                client_data_stats.append({
                    'client_id': i,
                    'data_size': len(client_indices),
                    'n_classes': len(unique),
                    'class_imbalance': class_imbalance,
                    'indices': client_indices
                })
        
        # Perform client clustering if CADP is enabled
        if self.cluster_manager:
            client_clusters = self.cluster_manager.cluster_clients(client_data_stats)
        else:
            client_clusters = {i: 0 for i in range(self.n_clients)}  # All in same cluster
        
        # Create client objects
        self.clients = []
        for i, stats in enumerate(client_data_stats):
            client = PrivacyEnhancedFederatedClient(
                client_id=i,
                cluster_id=client_clusters[i],
                model_type="neural_network",  # Will use NN for FL
                privacy_config=self.privacy_config
            )
            
            # Load client data
            X_client = X_train.iloc[stats['indices']]
            y_client = y_train[stats['indices']]
            client.load_local_data(X_client, y_client)
            
            self.clients.append(client)
        
        logger.info(f"Created {self.n_clients} federated clients with {data_distribution} distribution")
    
    def initialize_global_model(self):
        """Initialize global model based on best baseline"""
        # Use neural network for federated learning
        model_params = {
            'hidden_layer_sizes': (100, 50),
            'alpha': 0.01
        }
        
        # Initialize with a dummy model
        self.global_model = MLPClassifier(
            hidden_layer_sizes=model_params['hidden_layer_sizes'],
            learning_rate_init=0.01,
            max_iter=1,
            random_state=self.random_state,
            warm_start=True,
            alpha=model_params['alpha']
        )
        
        # Fit on small sample to initialize structure
        sample_size = min(100, len(self.data_splits['X_train']))
        self.global_model.fit(
            self.data_splits['X_train'].iloc[:sample_size],
            self.data_splits['y_train'][:sample_size]
        )
        
        # Initialize all clients
        for client in self.clients:
            client.initialize_model(self.global_model, model_params)
    
    def aggregate_parameters(self, client_params: List[Dict]) -> Dict:
        """Federated averaging with privacy-aware aggregation"""
        if not client_params:
            return {}
        
        # Decrypt parameters if needed
        if self.privacy_config['s_he']:
            decrypted_params = []
            for params in client_params:
                decrypted = PrivacyMechanisms.decrypt_selective_params(params)
                decrypted_params.append(decrypted)
            client_params = decrypted_params
        
        # Calculate total data size for weighted averaging
        total_size = sum(p['data_size'] for p in client_params)
        
        # Initialize aggregated parameters
        first_params = client_params[0]
        
        if 'coefs' in first_params:  # Neural network
            n_layers = len(first_params['coefs'])
            aggregated_coefs = []
            aggregated_intercepts = []
            
            for layer in range(n_layers):
                # Weighted average of coefficients
                layer_coef = np.zeros_like(first_params['coefs'][layer])
                layer_intercept = np.zeros_like(first_params['intercepts'][layer])
                
                for params in client_params:
                    weight = params['data_size'] / total_size
                    layer_coef += params['coefs'][layer] * weight
                    layer_intercept += params['intercepts'][layer] * weight
                
                aggregated_coefs.append(layer_coef)
                aggregated_intercepts.append(layer_intercept)
            
            return {
                'coefs': aggregated_coefs,
                'intercepts': aggregated_intercepts
            }
        
        else:  # Other model types
            # For Random Forest, we can't easily aggregate
            # Return the model from the client with most data
            best_client = max(client_params, key=lambda x: x['data_size'])
            return {'model': best_client['model']}
    
    def run_federated_training(self, n_rounds: int = 20, n_clients_per_round: int = None):
        """Run privacy-enhanced federated training"""
        if n_clients_per_round is None:
            n_clients_per_round = max(2, int(0.5 * self.n_clients))
        
        # Update noise scheduler with actual rounds
        if self.noise_scheduler:
            self.noise_scheduler.total_rounds = n_rounds
            self.noise_scheduler._initialize_schedule()
        
        logger.info(f"\nStarting Federated Training: {n_rounds} rounds, "
                   f"{n_clients_per_round} clients/round")
        
        best_val_accuracy = 0
        best_round = 0
        consecutive_no_improvement = 0
        
        for round_num in range(n_rounds):
            logger.info(f"\n--- Round {round_num + 1}/{n_rounds} ---")
            
            # Select clients for this round
            selected_clients = np.random.choice(
                self.clients, 
                size=n_clients_per_round, 
                replace=False
            )
            
            # Get round-specific epsilon if UANS is enabled
            if self.noise_scheduler and round_num > 0:
                performance_delta = self.round_results[-1]['val_accuracy'] - \
                                  (self.round_results[-2]['val_accuracy'] if len(self.round_results) > 1 else 0)
                round_epsilon = self.noise_scheduler.get_round_epsilon(round_num, performance_delta)
            else:
                round_epsilon = self.privacy_config.get('base_epsilon', 1.0)
            
            # Client training
            client_params = []
            client_performances = []
            
            for client in selected_clients:
                # Get cluster-specific epsilon if CADP is enabled
                if self.cluster_manager:
                    cluster_epsilon = self.cluster_manager.cluster_epsilons.get(
                        client.cluster_id, round_epsilon
                    )
                else:
                    cluster_epsilon = round_epsilon
                
                # Train client
                performance = client.local_training_with_privacy(round_num, cluster_epsilon)
                
                if 'error' not in performance:
                    client_performances.append(performance)
                    
                    # Get encrypted parameters
                    params = client.get_encrypted_parameters()
                    client_params.append(params)
            
            # Aggregate parameters
            if client_params:
                aggregated_params = self.aggregate_parameters(client_params)
                
                # Update global model
                self.global_model.coefs_ = aggregated_params['coefs']
                self.global_model.intercepts_ = aggregated_params['intercepts']
                
                # Update all clients
                for client in self.clients:
                    client.update_model_parameters(aggregated_params)
            
            # Evaluate global model
            val_metrics = self.evaluate_global_model()
            
            # Record round results
            round_result = {
                'round': round_num + 1,
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_score'],
                'client_performances': client_performances,
                'epsilon_used': round_epsilon,
                'n_clients': len(client_params)
            }
            
            if self.noise_scheduler:
                round_result['remaining_budget'] = self.noise_scheduler.get_remaining_budget(round_num + 1)
            
            self.round_results.append(round_result)
            
            logger.info(f"Round {round_num + 1} - Val Acc: {val_metrics['accuracy']:.4f}, "
                       f"Val F1: {val_metrics['f1_score']:.4f}, ε: {round_epsilon:.2f}")
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_round = round_num + 1
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
            
            if consecutive_no_improvement >= 5:
                logger.info(f"Early stopping at round {round_num + 1}")
                break
        
        # Final evaluation
        test_metrics = self.evaluate_global_model(on_test=True)
        
        logger.info(f"\nTraining Complete!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f} at round {best_round}")
        logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Final test F1-score: {test_metrics['f1_score']:.4f}")
        
        return {
            'round_results': self.round_results,
            'test_metrics': test_metrics,
            'best_val_accuracy': best_val_accuracy,
            'best_round': best_round
        }
    
    def evaluate_global_model(self, on_test: bool = False):
        """Evaluate global model on validation or test set"""
        if on_test:
            X_eval = self.data_splits['X_test']
            y_eval = self.data_splits['y_test']
        else:
            X_eval = self.data_splits['X_val']
            y_eval = self.data_splits['y_val']
        
        predictions = self.global_model.predict(X_eval)
        
        metrics = {
            'accuracy': accuracy_score(y_eval, predictions),
            'f1_score': f1_score(y_eval, predictions, average='weighted'),
            'precision': precision_score(y_eval, predictions, average='weighted'),
            'recall': recall_score(y_eval, predictions, average='weighted')
        }
        
        if on_test:
            metrics['confusion_matrix'] = confusion_matrix(y_eval, predictions)
            metrics['classification_report'] = classification_report(y_eval, predictions)
        
        return metrics


class PrivacyEnhancedFederatedExperiment:
    """Manages complete privacy-enhanced federated learning experiments"""
    
    def __init__(self, dataset_name: str, experiment_config: Dict = None):
        self.dataset_name = dataset_name
        self.experiment_config = experiment_config or {
            'n_clients': 10,
            'n_rounds': 20,
            'data_distribution': 'non-iid',
            'privacy_configs': [
                {'name': 'no_privacy', 'ca_ldp': False, 'cadp': False, 's_he': False, 'uans': False},
                {'name': 'ca_ldp_only', 'ca_ldp': True, 'cadp': False, 's_he': False, 'uans': False, 'base_epsilon': 1.0},
                {'name': 'full_privacy', 'ca_ldp': True, 'cadp': True, 's_he': True, 'uans': True, 'base_epsilon': 1.0}
            ]
        }
        self.results = {}
    
    def run_privacy_comparison(self):
        """Run experiments comparing different privacy configurations"""
        logger.info(f"\n{'='*80}")
        logger.info(f"PRIVACY-ENHANCED FEDERATED LEARNING EXPERIMENT: {self.dataset_name.upper()}")
        logger.info(f"{'='*80}")
        
        for privacy_config in self.experiment_config['privacy_configs']:
            config_name = privacy_config['name']
            logger.info(f"\n\n--- Running configuration: {config_name} ---")
            
            # Create server with privacy configuration
            server = PriFedGridGuardServer(
                dataset_name=self.dataset_name,
                n_clients=self.experiment_config['n_clients'],
                privacy_config=privacy_config
            )
            
            # Setup
            server.load_dataset_and_baseline()
            server.create_federated_clients(self.experiment_config['data_distribution'])
            server.initialize_global_model()
            
            # Run training
            training_results = server.run_federated_training(
                n_rounds=self.experiment_config['n_rounds']
            )
            
            # Store results
            self.results[config_name] = {
                'training_results': training_results,
                'privacy_config': privacy_config,
                'baseline_comparison': server.best_baseline_config
            }
        
        # Generate comparison report
        self._generate_comparison_report()
        
        return self.results
    
    def _generate_comparison_report(self):
        """Generate detailed comparison report"""
        logger.info(f"\n\n{'='*80}")
        logger.info("PRIVACY COMPARISON RESULTS")
        logger.info(f"{'='*80}")
        
        # Summary table
        summary_data = []
        
        for config_name, results in self.results.items():
            test_metrics = results['training_results']['test_metrics']
            
            summary_data.append({
                'Configuration': config_name,
                'Test_Accuracy': test_metrics['accuracy'],
                'Test_F1': test_metrics['f1_score'],
                'Best_Val_Acc': results['training_results']['best_val_accuracy'],
                'Best_Round': results['training_results']['best_round']
            })
        
        # Print summary
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Compare with baseline
        if self.results and 'baseline_comparison' in list(self.results.values())[0]:
            baseline = list(self.results.values())[0]['baseline_comparison']
            if baseline:
                print(f"\nBaseline {baseline['model_type']} Performance:")
                print(f"  Accuracy: {baseline['accuracy']:.4f}")
                print(f"  F1-Score: {baseline['f1_score']:.4f}")
        
        # Privacy-utility trade-off analysis
        if 'no_privacy' in self.results and 'full_privacy' in self.results:
            no_privacy_acc = self.results['no_privacy']['training_results']['test_metrics']['accuracy']
            full_privacy_acc = self.results['full_privacy']['training_results']['test_metrics']['accuracy']
            
            accuracy_cost = (no_privacy_acc - full_privacy_acc) * 100
            print(f"\nPrivacy Cost Analysis:")
            print(f"  Accuracy reduction with full privacy: {accuracy_cost:.2f}%")
            
            if baseline:
                fl_improvement = (full_privacy_acc - baseline['accuracy']) * 100
                print(f"  FL with privacy vs centralized baseline: {fl_improvement:+.2f}%")
    
    def save_results(self, save_path: str = "results/federated_learning"):
        """Save experiment results"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(save_path / f'{self.dataset_name}_privacy_enhanced_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary CSV
        summary_data = []
        for config_name, results in self.results.items():
            test_metrics = results['training_results']['test_metrics']
            summary_data.append({
                'Dataset': self.dataset_name,
                'Configuration': config_name,
                'Test_Accuracy': test_metrics['accuracy'],
                'Test_F1': test_metrics['f1_score'],
                'Test_Precision': test_metrics['precision'],
                'Test_Recall': test_metrics['recall']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path / f'{self.dataset_name}_privacy_comparison.csv', index=False)
        
        logger.info(f"\nResults saved to {save_path}")


def main():
    """Run privacy-enhanced federated learning experiments"""
    import argparse
    from utils.config import Config
    
    # Load configuration
    config = Config()
    
    parser = argparse.ArgumentParser(description='PriFed-GridGuard: Privacy-Enhanced FL for Smart Grid')
    parser.add_argument('--dataset', choices=['msu', 'pecan', 'sgcc', 'all'], 
                       default='all', help='Dataset to use')
    parser.add_argument('--n_clients', type=int, 
                       default=config.get('federated_learning.num_clients', 10), 
                       help='Number of federated clients')
    parser.add_argument('--n_rounds', type=int, 
                       default=config.get('federated_learning.num_rounds', 20),
                       help='Number of federated rounds')
    parser.add_argument('--distribution', choices=['iid', 'non-iid'], 
                       default=config.get('federated_learning.data_distribution', 'non-iid'), 
                       help='Data distribution')
    parser.add_argument('--epsilon', type=float, 
                       default=config.get('privacy.default_epsilon', 1.0),
                       help='Base epsilon for differential privacy')
    
    args = parser.parse_args()
    
    # Experiment configuration with config file integration
    experiment_config = {
        'n_clients': args.n_clients,
        'n_rounds': args.n_rounds,
        'data_distribution': args.distribution,
        'privacy_configs': [
            {
                'name': 'no_privacy',
                'ca_ldp': False, 'cadp': False, 's_he': False, 'uans': False
            },
            {
                'name': 'ca_ldp_only',
                'ca_ldp': True, 'cadp': False, 's_he': False, 'uans': False,
                'base_epsilon': args.epsilon
            },
            {
                'name': 'cadp_only', 
                'ca_ldp': False, 'cadp': True, 's_he': False, 'uans': False,
                'base_epsilon': args.epsilon, 'n_clusters': 3
            },
            {
                'name': 'full_privacy',
                'ca_ldp': True, 'cadp': True, 's_he': True, 'uans': True,
                'base_epsilon': args.epsilon, 'n_clusters': 3,
                'total_privacy_budget': config.get('privacy.accounting.max_privacy_loss', 10.0)
            }
        ]
    }
    
    # Log configuration usage
    logger.info("Using configuration values:")
    logger.info(f"  - Clients: {args.n_clients} (config default: {config.get('federated_learning.num_clients')})")
    logger.info(f"  - Rounds: {args.n_rounds} (config default: {config.get('federated_learning.num_rounds')})")
    logger.info(f"  - Distribution: {args.distribution} (config default: {config.get('federated_learning.data_distribution')})")
    logger.info(f"  - Epsilon: {args.epsilon} (config default: {config.get('privacy.default_epsilon')})")
    
    # Run experiments
    datasets = ['msu', 'pecan', 'sgcc'] if args.dataset == 'all' else [args.dataset]
    
    all_results = {}
    
    for dataset_name in datasets:
        try:
            experiment = PrivacyEnhancedFederatedExperiment(
                dataset_name=dataset_name,
                experiment_config=experiment_config
            )
            
            results = experiment.run_privacy_comparison()
            experiment.save_results()
            
            all_results[dataset_name] = results
            
        except Exception as e:
            logger.error(f"Error running experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate cross-dataset comparison if multiple datasets
    if len(all_results) > 1:
        logger.info(f"\n\n{'='*80}")
        logger.info("CROSS-DATASET COMPARISON")
        logger.info(f"{'='*80}")
        
        comparison_data = []
        for dataset_name, dataset_results in all_results.items():
            for config_name, config_results in dataset_results.items():
                test_metrics = config_results['training_results']['test_metrics']
                comparison_data.append({
                    'Dataset': dataset_name.upper(),
                    'Configuration': config_name,
                    'Accuracy': f"{test_metrics['accuracy']:.4f}",
                    'F1-Score': f"{test_metrics['f1_score']:.4f}"
                })
        
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()