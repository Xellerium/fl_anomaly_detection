# config.py
"""
Simple Configuration Management for Smart Grid Federated Learning Research
Handles YAML configuration loading with sensible defaults

For PhD Supervision Discussion:
- Centralized configuration management
- Default values for all experiments
- Easy parameter modification for different scenarios
"""

import yaml
from pathlib import Path

class Config:
    """Simple configuration manager for research experiments"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file with defaults"""
        # Default configuration for all experiments
        default_config = {
            'data': {
                'raw_path': 'data/raw/',
                'processed_path': 'data/processed/', 
                'test_size': 0.2,
                'validation_size': 0.1
            },
            'model': {
                'random_state': 42,
                'cross_validation_folds': 5
            },
            'federated_learning': {
                'num_clients': 5,
                'num_rounds': 10,
                'data_distribution': 'iid'
            },
            'privacy': {
                'default_epsilon': 1.0,
                'privacy_budgets': [0.1, 1.0, 10.0]
            },
            'output': {
                'save_models': True,
                'save_figures': True,
                'figure_dpi': 300
            }
        }
        
        # Try to load from file, use defaults if file doesn't exist
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                # Merge file config with defaults
                self._deep_update(default_config, file_config)
                print(f"Configuration loaded from {self.config_path}")
                
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
        else:
            print("No config file found, using defaults")
        
        return default_config
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key, default=None):
        """Get configuration value using dot notation (e.g., 'data.raw_path')"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def save(self, output_path=None):
        """Save current configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to {output_path}")
    
    def display(self):
        """Display current configuration"""
        print("Current Configuration:")
        print("-" * 30)
        self._print_nested_dict(self.config)
    
    def _print_nested_dict(self, d, indent=0):
        """Recursively print nested dictionary"""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_nested_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")

# Simple usage example
if __name__ == "__main__":
    config = Config()
    config.display()
    
    # Example usage
    print(f"\nExample usage:")
    print(f"Raw data path: {config.get('data.raw_path')}")
    print(f"Number of FL clients: {config.get('federated_learning.num_clients')}")
    print(f"Random state: {config.get('model.random_state')}")
