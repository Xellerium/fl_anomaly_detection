"""
Example: Running DPFed-GridGuard with Custom Configuration

This example demonstrates how to run a federated learning experiment
with specific privacy settings and dataset selection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_pipeline import SmartGridDataProcessor
from src.baseline_models import BaselineEvaluator
from src.enhanced_federated_learning import EnhancedFederatedExperiment

def run_custom_experiment():
    """Run a custom federated learning experiment"""
    
    print("DPFed-GridGuard Custom Experiment Example")
    print("=" * 50)
    
    # Step 1: Process data (if not already done)
    print("\n1. Processing data...")
    processor = SmartGridDataProcessor(dataset_name='msu')
    
    # Check if data already processed
    if not processor.check_processed_data_exists():
        processor.run_complete_pipeline()
        print("✓ Data processing complete")
    else:
        print("✓ Using existing processed data")
    
    # Step 2: Train baseline model (optional but recommended)
    print("\n2. Training baseline model...")
    evaluator = BaselineEvaluator(dataset_name='msu')
    best_model_info = evaluator.run_complete_evaluation()
    print(f"✓ Best baseline model: {best_model_info['model_name']}")
    
    # Step 3: Run federated learning with custom settings
    print("\n3. Running federated learning...")
    
    # Custom configuration
    config = {
        'dataset': 'msu',
        'n_clients': 5,
        'n_rounds': 10,
        'epsilon': 1.0,  # Privacy budget
        'distribution': 'non-iid',
        'client_fraction': 0.5,  # 50% clients per round
        'local_epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.01
    }
    
    # Run experiment
    experiment = EnhancedFederatedExperiment(
        dataset_name=config['dataset'],
        n_clients=config['n_clients'],
        distribution_type=config['distribution'],
        privacy_budget=config['epsilon']
    )
    
    results = experiment.run_experiment(n_rounds=config['n_rounds'])
    
    # Print results
    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS")
    print("=" * 50)
    
    print(f"\nDataset: {config['dataset'].upper()}")
    print(f"Privacy Budget (ε): {config['epsilon']}")
    print(f"Clients: {config['n_clients']} ({config['distribution']})")
    print(f"Rounds: {config['n_rounds']}")
    
    print(f"\nPerformance:")
    print(f"  Baseline F1-Score: {results['baseline_performance']['f1_score']:.4f}")
    print(f"  Federated F1-Score: {results['federated_performance']['f1_score']:.4f}")
    print(f"  Utility Retention: {results['utility_retention']:.1f}%")
    
    print(f"\nConvergence: Round {results['convergence_round']}")
    print(f"Total Training Time: {results['total_time']:.1f} seconds")
    
    return results

def run_privacy_comparison():
    """Compare performance across different privacy budgets"""
    
    print("\nPrivacy-Utility Trade-off Analysis")
    print("=" * 50)
    
    privacy_budgets = [0.5, 1.0, 2.0, 5.0]
    results = {}
    
    for epsilon in privacy_budgets:
        print(f"\nTesting with ε = {epsilon}...")
        
        experiment = EnhancedFederatedExperiment(
            dataset_name='msu',
            n_clients=5,
            distribution_type='non-iid',
            privacy_budget=epsilon
        )
        
        result = experiment.run_experiment(n_rounds=10)
        results[epsilon] = result['utility_retention']
        
        print(f"  Utility Retention: {result['utility_retention']:.1f}%")
    
    # Plot results
    print("\nSummary:")
    print("Epsilon | Utility Retention")
    print("-" * 25)
    for eps, retention in results.items():
        print(f"{eps:7.1f} | {retention:6.1f}%")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPFed-GridGuard Example")
    parser.add_argument('--mode', choices=['custom', 'privacy'], 
                       default='custom',
                       help='Example mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'custom':
        run_custom_experiment()
    else:
        run_privacy_comparison()
