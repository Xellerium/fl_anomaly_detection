"""
Flower-based Federated Learning Implementation for Smart Grid Security
Advanced federated learning using the Flower framework

Requirements:
pip install flwr[simulation] torch torchvision
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

class SmartGridNN(nn.Module):
    """Neural Network for Smart Grid Security Classification"""
    
    def __init__(self, input_size: int, num_classes: int):
        super(SmartGridNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class SmartGridClient(fl.client.NumPyClient):
    """Flower client for smart grid federated learning"""
    
    def __init__(self, client_id: int, train_data: Tuple, test_data: Tuple, 
                 input_size: int, num_classes: int):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = SmartGridNN(input_size, num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Prepare data
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Convert to tensors
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.LongTensor(y_train)
        )
        self.test_dataset = TensorDataset(
            torch.FloatTensor(X_test.values),
            torch.LongTensor(y_test)
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        print(f"Client {client_id}: {len(self.train_dataset)} train, {len(self.test_dataset)} test samples")
    def get_parameters(self, config):
       """Return model parameters as a list of NumPy ndarrays"""
       return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
       """Set model parameters from a list of NumPy ndarrays"""
       params_dict = zip(self.model.state_dict().keys(), parameters)
       state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
       self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
       """Train the model on the locally held training set"""
       self.set_parameters(parameters)
       
       # Training configuration
       epochs = config.get("local_epochs", 3)
       learning_rate = config.get("learning_rate", 0.001)
       
       # Update optimizer learning rate
       for param_group in self.optimizer.param_groups:
           param_group['lr'] = learning_rate
       
       self.model.train()
       total_loss = 0.0
       correct = 0
       total = 0
       
       for epoch in range(epochs):
           for batch_idx, (data, targets) in enumerate(self.train_loader):
               data, targets = data.to(self.device), targets.to(self.device)
               
               self.optimizer.zero_grad()
               outputs = self.model(data)
               loss = self.criterion(outputs, targets)
               loss.backward()
               self.optimizer.step()
               
               total_loss += loss.item()
               _, predicted = outputs.max(1)
               total += targets.size(0)
               correct += predicted.eq(targets).sum().item()
       
       accuracy = correct / total
       avg_loss = total_loss / len(self.train_loader) / epochs
       
       print(f"Client {self.client_id}: Local training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
       
       return self.get_parameters(config={}), len(self.train_dataset), {
           "loss": avg_loss,
           "accuracy": accuracy
       }

    def evaluate(self, parameters, config):
       """Evaluate the model on the locally held test set"""
       self.set_parameters(parameters)
       
       self.model.eval()
       total_loss = 0.0
       correct = 0
       total = 0
       all_predictions = []
       all_targets = []
       
       with torch.no_grad():
           for data, targets in self.test_loader:
               data, targets = data.to(self.device), targets.to(self.device)
               outputs = self.model(data)
               loss = self.criterion(outputs, targets)
               
               total_loss += loss.item()
               _, predicted = outputs.max(1)
               total += targets.size(0)
               correct += predicted.eq(targets).sum().item()
               
               all_predictions.extend(predicted.cpu().numpy())
               all_targets.extend(targets.cpu().numpy())
       
       accuracy = correct / total
       avg_loss = total_loss / len(self.test_loader)
       f1 = f1_score(all_targets, all_predictions, average='weighted')
       
       return avg_loss, len(self.test_dataset), {
           "accuracy": accuracy,
           "f1_score": f1
       }

def client_fn(cid: str) -> SmartGridClient:
   """Create a Flower client instance"""
   client_id = int(cid)
   
   # Load client data (you'll need to implement data distribution logic)
   train_data, test_data, input_size, num_classes = load_client_data(client_id)
   
   return SmartGridClient(client_id, train_data, test_data, input_size, num_classes)

def load_client_data(client_id: int, num_clients: int = 5):
   """Load and distribute data for specific client"""
   # Load preprocessed data
   data_path = Path("data/processed")
   with open(data_path / 'data_splits.pkl', 'rb') as f:
       splits = pickle.load(f)
   
   # Combine train and validation for federated learning
   X_fed = pd.concat([splits['X_train'], splits['X_val']])
   y_fed = np.concatenate([splits['y_train'], splits['y_val']])
   X_test = splits['X_test']
   y_test = splits['y_test']
   
   # Split data among clients
   indices = np.random.RandomState(42).permutation(len(X_fed))
   client_indices = np.array_split(indices, num_clients)
   
   # Get data for specific client
   client_idx = client_indices[client_id]
   X_client = X_fed.iloc[client_idx].reset_index(drop=True)
   y_client = y_fed[client_idx]
   
   return (X_client, y_client), (X_test, y_test), X_fed.shape[1], len(np.unique(y_fed))

def get_evaluate_fn(test_data):
   """Return an evaluation function for server-side evaluation"""
   X_test, y_test = test_data
   
   def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
       # Create model and set parameters
       input_size = X_test.shape[1]
       num_classes = len(np.unique(y_test))
       model = SmartGridNN(input_size, num_classes)
       
       params_dict = zip(model.state_dict().keys(), parameters)
       state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
       model.load_state_dict(state_dict, strict=True)
       
       # Evaluate
       model.eval()
       test_dataset = TensorDataset(
           torch.FloatTensor(X_test.values),
           torch.LongTensor(y_test)
       )
       test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
       
       criterion = nn.CrossEntropyLoss()
       total_loss = 0.0
       correct = 0
       total = 0
       all_predictions = []
       all_targets = []
       
       with torch.no_grad():
           for data, targets in test_loader:
               outputs = model(data)
               loss = criterion(outputs, targets)
               
               total_loss += loss.item()
               _, predicted = outputs.max(1)
               total += targets.size(0)
               correct += predicted.eq(targets).sum().item()
               
               all_predictions.extend(predicted.numpy())
               all_targets.extend(targets.numpy())
       
       accuracy = correct / total
       avg_loss = total_loss / len(test_loader)
       f1 = f1_score(all_targets, all_predictions, average='weighted')
       
       print(f"Server Round {server_round}: Global Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
       
       return avg_loss, {"accuracy": accuracy, "f1_score": f1}
   
   return evaluate

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
   """Aggregate evaluation metrics using weighted average"""
   # Calculate weighted averages
   total_examples = sum(num_examples for num_examples, _ in metrics)
   
   weighted_accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
   weighted_f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
   
   aggregated_accuracy = sum(weighted_accuracies) / total_examples
   aggregated_f1 = sum(weighted_f1_scores) / total_examples
   
   return {
       "accuracy": aggregated_accuracy,
       "f1_score": aggregated_f1
   }

def run_flower_federated_learning():
   """Run federated learning experiment using Flower framework"""
   print("="*70)
   print("FLOWER-BASED FEDERATED LEARNING FOR SMART GRID SECURITY")
   print("="*70)
   
   # Load test data for server-side evaluation
   data_path = Path("data/processed")
   with open(data_path / 'data_splits.pkl', 'rb') as f:
       splits = pickle.load(f)
   
   X_test = splits['X_test']
   y_test = splits['y_test']
   
   # Configure federated learning strategy
   strategy = fl.server.strategy.FedAvg(
       fraction_fit=1.0,  # Use all clients for training
       fraction_evaluate=1.0,  # Use all clients for evaluation
       min_fit_clients=5,  # Minimum clients for training
       min_evaluate_clients=5,  # Minimum clients for evaluation
       min_available_clients=5,  # Minimum available clients
       evaluate_fn=get_evaluate_fn((X_test, y_test)),  # Server-side evaluation
       evaluate_metrics_aggregation_fn=weighted_average,  # Metrics aggregation
       on_fit_config_fn=lambda server_round: {
           "local_epochs": 3,
           "learning_rate": 0.001 / (1 + 0.1 * server_round)  # Decay learning rate
       }
   )
   
   # Start Flower simulation
   print("Starting Flower federated learning simulation...")
   
   hist = fl.simulation.start_simulation(
       client_fn=client_fn,
       num_clients=5,
       config=fl.server.ServerConfig(num_rounds=15),
       strategy=strategy,
       client_resources={"num_cpus": 1, "num_gpus": 0}
   )
   
   # Print final results
   if hist.metrics_centralized:
       final_accuracy = hist.metrics_centralized["accuracy"][-1][1]
       final_f1 = hist.metrics_centralized["f1_score"][-1][1]
       print(f"\nFinal Global Performance:")
       print(f"  Test Accuracy: {final_accuracy:.4f}")
       print(f"  Test F1-Score: {final_f1:.4f}")
   
   # Save results
   results_path = Path("results/federated_learning")
   results_path.mkdir(parents=True, exist_ok=True)
   
   with open(results_path / 'flower_federated_results.pkl', 'wb') as f:
       pickle.dump(hist, f)
   
   print(f"Flower federated learning results saved to {results_path}")
   
   return hist

# Global data storage for client function
_client_data_cache = {}

def load_client_data(client_id: int, num_clients: int = 5):
   """Load and distribute data for specific client with caching"""
   global _client_data_cache
   
   if not _client_data_cache:
       # Load preprocessed data once
       data_path = Path("data/processed")
       with open(data_path / 'data_splits.pkl', 'rb') as f:
           splits = pickle.load(f)
       
       # Combine train and validation for federated learning
       X_fed = pd.concat([splits['X_train'], splits['X_val']])
       y_fed = np.concatenate([splits['y_train'], splits['y_val']])
       X_test = splits['X_test']
       y_test = splits['y_test']
       
       # Split data among clients
       indices = np.random.RandomState(42).permutation(len(X_fed))
       client_indices = np.array_split(indices, num_clients)
       
       # Cache client data
       for i in range(num_clients):
           client_idx = client_indices[i]
           X_client = X_fed.iloc[client_idx].reset_index(drop=True)
           y_client = y_fed[client_idx]
           
           _client_data_cache[i] = {
               'train_data': (X_client, y_client),
               'test_data': (X_test, y_test),
               'input_size': X_fed.shape[1],
               'num_classes': len(np.unique(y_fed))
           }
   
   client_info = _client_data_cache[client_id]
   return (client_info['train_data'], client_info['test_data'], 
           client_info['input_size'], client_info['num_classes'])

if __name__ == "__main__":
   # Run Flower-based federated learning
   flower_results = run_flower_federated_learning()