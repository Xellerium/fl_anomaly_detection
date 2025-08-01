"""
Quick Start Example for DPFed-GridGuard

This minimal example shows how to quickly get started with the framework.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import the main components
from src.data_pipeline import SmartGridDataProcessor
from src.enhanced_federated_learning import EnhancedFederatedExperiment

# Process data
print("1. Processing MSU dataset...")
processor = SmartGridDataProcessor(dataset_name='msu')
processor.run_complete_pipeline()

# Run federated learning with privacy
print("\n2. Running federated learning (ε=1.0)...")
experiment = EnhancedFederatedExperiment(
    dataset_name='msu',
    n_clients=5,
    privacy_budget=1.0
)

results = experiment.run_experiment(n_rounds=10)

# Show results
print(f"\n✓ Federated F1-Score: {results['federated_performance']['f1_score']:.3f}")
print(f"✓ Utility Retention: {results['utility_retention']:.1f}%")
print(f"✓ Privacy Budget Used: ε = {experiment.privacy_budget}")
