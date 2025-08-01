#!/usr/bin/env python3
"""
PriFed-GridGuard Comprehensive Experiment Runner

This script runs the complete experimental suite for PriFed-GridGuard:
1. Baseline models on all datasets
2. Federated learning without privacy
3. Federated learning with different privacy configurations
4. Analysis and visualization generation

Results are organized in timestamped folders for easy comparison.
"""

import os
import sys
import time
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.utils.config import Config

class ExperimentRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_root = Path(f"experiments_{self.timestamp}")
        self.results_backup = Path("results_backup")
        
        # Load enhanced configuration
        self.config = Config()
        
        # Experiment configurations from config file
        self.privacy_budgets = self.config.get('privacy.privacy_budgets', [0.5, 1.0, 2.0, 5.0])
        self.federated_config = {
            'n_clients': self.config.get('federated_learning.num_clients', 5),
            'n_rounds': self.config.get('federated_learning.num_rounds', 15),
            'distribution': self.config.get('federated_learning.data_distribution', 'non-iid')
        }
        
        # Create experiment directories first
        self.create_directories()
        
        # Setup logging after directories exist
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for experiment tracking"""
        log_file = self.experiment_root / 'experiment_log.txt'
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler without emojis for Windows compatibility
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        # Set formatters
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def create_directories(self):
        """Create organized directory structure for experiments"""
        self.experiment_root.mkdir(exist_ok=True)
        
        # Main experiment folders
        self.folders = {
            'baseline': self.experiment_root / '01_baseline_models',
            'federated_no_privacy': self.experiment_root / '02_federated_no_privacy',
            'federated_privacy': self.experiment_root / '03_federated_privacy',
            'analysis': self.experiment_root / '04_analysis_results',
            'comparison': self.experiment_root / '05_final_comparison'
        }
        
        # Privacy sub-folders
        for eps in self.privacy_budgets:
            privacy_folder = self.folders['federated_privacy'] / f'epsilon_{eps}'
            privacy_folder.mkdir(parents=True, exist_ok=True)
        
        # Create all directories
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Created experiment directories in: {self.experiment_root}")
    
    def backup_existing_results(self):
        """Backup existing results before running new experiments"""
        results_path = Path("results")
        if results_path.exists():
            backup_path = self.results_backup / f"backup_{self.timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.copytree(results_path, backup_path / "results")
                self.logger.info(f"Backed up existing results to: {backup_path}")
            except Exception as e:
                self.logger.warning(f"Could not backup results: {e}")
    
    def run_command(self, command, experiment_name, output_folder):
        """Run a command and handle results"""
        self.logger.info(f"Starting: {experiment_name}")
        self.logger.info(f"Command: {command}")
        
        start_time = time.time()
        
        try:
            # Run the command
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Save command output
            output_file = output_folder / f"{experiment_name}_output.txt"
            with open(output_file, 'w') as f:
                f.write(f"Command: {command}\n")
                f.write(f"Runtime: {runtime:.2f} seconds\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            # Copy results to experiment folder
            self.copy_results_to_folder(output_folder, experiment_name)
            
            self.logger.info(f"[SUCCESS] Completed: {experiment_name} in {runtime:.2f}s")
            return True
            
        except subprocess.CalledProcessError as e:
            runtime = time.time() - start_time
            self.logger.error(f"[FAILED] {experiment_name} after {runtime:.2f}s")
            self.logger.error(f"Error: {e}")
            
            # Save error info
            error_file = output_folder / f"{experiment_name}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Command: {command}\n")
                f.write(f"Runtime: {runtime:.2f} seconds\n")
                f.write(f"Return code: {e.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(e.stdout if e.stdout else "No stdout")
                f.write("\nSTDERR:\n")
                f.write(e.stderr if e.stderr else "No stderr")
            
            return False
    
    def copy_results_to_folder(self, output_folder, experiment_name):
        """Copy results from main results folder to experiment folder"""
        results_path = Path("results")
        if not results_path.exists():
            return
        
        try:
            # Copy entire results directory
            experiment_results = output_folder / "results"
            if experiment_results.exists():
                shutil.rmtree(experiment_results)
            shutil.copytree(results_path, experiment_results)
            
            # Create metadata file
            metadata = {
                'experiment_name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'results_location': str(experiment_results)
            }
            
            metadata_file = output_folder / "experiment_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not copy results for {experiment_name}: {e}")
    
    def run_baseline_experiments(self):
        """Run baseline model experiments"""
        self.logger.info("="*60)
        self.logger.info("PHASE 1: BASELINE MODEL EVALUATION")
        self.logger.info("="*60)
        
        # Clear previous results
        self.clear_results()
        
        # Run baseline comparison
        command = "python src/baseline_models.py --mode compare --optimize"
        success = self.run_command(
            command, 
            "baseline_models", 
            self.folders['baseline']
        )
        
        if success:
            self.logger.info("[SUCCESS] Baseline experiments completed successfully")
        else:
            self.logger.error("[FAILED] Baseline experiments failed")
            return False
        
        return True
    
    def run_federated_no_privacy(self):
        """Run federated learning without privacy"""
        self.logger.info("="*60)
        self.logger.info("PHASE 2: FEDERATED LEARNING (NO PRIVACY)")
        self.logger.info("="*60)
        
        # Clear previous results
        self.clear_results()
        
        # Run federated learning with high epsilon (effectively no privacy)
        command = (f"python src/enhanced_federated_learning.py "
                  f"--dataset all "
                  f"--n_clients {self.federated_config['n_clients']} "
                  f"--n_rounds {self.federated_config['n_rounds']} "
                  f"--distribution {self.federated_config['distribution']} "
                  f"--epsilon 100.0")
        
        success = self.run_command(
            command,
            "federated_no_privacy",
            self.folders['federated_no_privacy']
        )
        
        if success:
            self.logger.info("[SUCCESS] Federated (no privacy) experiments completed successfully")
        else:
            self.logger.error("[FAILED] Federated (no privacy) experiments failed")
            return False
        
        return True
    
    def run_federated_privacy_experiments(self):
        """Run federated learning with different privacy configurations"""
        self.logger.info("="*60)
        self.logger.info("PHASE 3: FEDERATED LEARNING (WITH PRIVACY)")
        self.logger.info("="*60)
        
        for eps in self.privacy_budgets:
            self.logger.info(f"Running privacy experiment with epsilon = {eps}")
            
            # Clear previous results
            self.clear_results()
            
            # Run federated learning with specific epsilon
            command = (f"python src/enhanced_federated_learning.py "
                      f"--dataset all "
                      f"--n_clients {self.federated_config['n_clients']} "
                      f"--n_rounds {self.federated_config['n_rounds']} "
                      f"--distribution {self.federated_config['distribution']} "
                      f"--epsilon {eps}")
            
            output_folder = self.folders['federated_privacy'] / f'epsilon_{eps}'
            success = self.run_command(
                command,
                f"federated_privacy_eps_{eps}",
                output_folder
            )
            
            if not success:
                self.logger.error(f"[FAILED] Privacy experiment with epsilon = {eps} failed")
                return False
            
            time.sleep(2)  # Brief pause between experiments
        
        self.logger.info("[SUCCESS] All privacy experiments completed successfully")
        return True
    
    def run_analysis(self):
        """Run comprehensive analysis"""
        self.logger.info("="*60)
        self.logger.info("PHASE 4: ANALYSIS AND VISUALIZATION")
        self.logger.info("="*60)
        
        # We'll consolidate results first, then run analysis
        self.consolidate_results()
        
        # Run analysis
        command = "python src/analysis_and_evaluation.py"
        success = self.run_command(
            command,
            "comprehensive_analysis",
            self.folders['analysis']
        )
        
        if success:
            self.logger.info("[SUCCESS] Analysis completed successfully")
        else:
            self.logger.error("[FAILED] Analysis failed")
            return False
        
        return True
    
    def consolidate_results(self):
        """Consolidate all experimental results for final comparison"""
        self.logger.info("Consolidating results from all experiments...")
        
        consolidated_folder = self.folders['comparison']
        
        # Copy key results from each experiment
        experiments = [
            ('baseline', self.folders['baseline']),
            ('no_privacy', self.folders['federated_no_privacy']),
        ]
        
        # Add privacy experiments
        for eps in self.privacy_budgets:
            exp_name = f'privacy_eps_{eps}'
            exp_folder = self.folders['federated_privacy'] / f'epsilon_{eps}'
            experiments.append((exp_name, exp_folder))
        
        # Create consolidated structure
        for exp_name, exp_folder in experiments:
            if (exp_folder / 'results').exists():
                dest_folder = consolidated_folder / exp_name
                dest_folder.mkdir(exist_ok=True)
                
                try:
                    shutil.copytree(
                        exp_folder / 'results', 
                        dest_folder / 'results',
                        dirs_exist_ok=True
                    )
                    # Copy metadata
                    if (exp_folder / 'experiment_metadata.json').exists():
                        shutil.copy2(
                            exp_folder / 'experiment_metadata.json',
                            dest_folder / 'experiment_metadata.json'
                        )
                except Exception as e:
                    self.logger.warning(f"Could not consolidate {exp_name}: {e}")
    
    def clear_results(self):
        """Clear the results directory before running new experiment"""
        results_path = Path("results")
        if results_path.exists():
            try:
                shutil.rmtree(results_path)
                time.sleep(1)  # Give filesystem time to update
            except Exception as e:
                self.logger.warning(f"Could not clear results directory: {e}")
    
    def generate_experiment_summary(self):
        """Generate a comprehensive experiment summary"""
        self.logger.info("Generating experiment summary...")
        
        summary = {
            'experiment_timestamp': self.timestamp,
            'total_runtime': 0,
            'configurations': {
                'federated_learning': self.federated_config,
                'privacy_budgets': self.privacy_budgets
            },
            'experiments_run': [],
            'results_locations': {}
        }
        
        # Save summary
        summary_file = self.experiment_root / 'experiment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create README
        readme_content = f"""# PriFed-GridGuard Experiment Results
        
Experiment Run: {self.timestamp}

## Experiment Structure

1. **Baseline Models** (`01_baseline_models/`)
   - Comprehensive evaluation of centralized models
   - All models (RF, XGBoost, LightGBM, CatBoost, Logistic Regression) on all datasets (MSU, Pecan, SGCC)

2. **Federated Learning (No Privacy)** (`02_federated_no_privacy/`)
   - Standard federated learning without privacy mechanisms
   - Configuration: {self.federated_config['n_clients']} clients, {self.federated_config['n_rounds']} rounds, {self.federated_config['distribution']} distribution

3. **Federated Learning (Privacy)** (`03_federated_privacy/`)
   - Privacy-enhanced federated learning with different epsilon values
   - Tested privacy budgets: {', '.join(map(str, self.privacy_budgets))}
   - Each configuration includes: CA-LDP, CADP, S-HE, UANS mechanisms

4. **Analysis Results** (`04_analysis_results/`)
   - Comprehensive analysis and visualization
   - Publication-ready figures

5. **Final Comparison** (`05_final_comparison/`)
   - Consolidated results from all experiments
   - Ready for article analysis

## Usage

Each folder contains:
- `results/` - Complete experimental results
- `experiment_metadata.json` - Experiment configuration and metadata
- `*_output.txt` - Command output and runtime information

## Analysis

Run the new notebook `prifred_gridguard_analysis.ipynb` with these results to generate 
publication-ready comparisons and figures.
"""
        
        readme_file = self.experiment_root / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    
    def run_all_experiments(self):
        """Run the complete experimental suite"""
        start_time = time.time()
        
        self.logger.info("Starting PriFed-GridGuard Comprehensive Experiments")
        self.logger.info(f"Results will be saved to: {self.experiment_root}")
        
        # Backup existing results
        self.backup_existing_results()
        
        try:
            # Phase 1: Baseline
            if not self.run_baseline_experiments():
                raise Exception("Baseline experiments failed")
            
            # Phase 2: Federated (No Privacy)
            if not self.run_federated_no_privacy():
                raise Exception("Federated (no privacy) experiments failed")
            
            # Phase 3: Federated (Privacy)
            if not self.run_federated_privacy_experiments():
                raise Exception("Privacy experiments failed")
            
            # Phase 4: Analysis (skip for now - results are already organized)
            self.logger.info("Skipping analysis phase - results are organized and ready for manual analysis")
            
            # Generate summary
            self.generate_experiment_summary()
            
            total_time = time.time() - start_time
            
            self.logger.info("="*60)
            self.logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total runtime: {total_time/60:.1f} minutes")
            self.logger.info(f"Results location: {self.experiment_root}")
            self.logger.info("="*60)
            
            print(f"\n{'='*60}")
            print("[SUCCESS] EXPERIMENT SUITE COMPLETED!")
            print(f"Results saved to: {self.experiment_root}")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Ready for analysis with prifred_gridguard_analysis.ipynb")
            print(f"{'='*60}")
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"[FAILED] Experiment suite failed after {total_time/60:.1f} minutes: {e}")
            print(f"\n[FAILED] Experiments failed: {e}")
            print(f"Partial results in: {self.experiment_root}")
            sys.exit(1)

def main():
    """Main execution function"""
    print("PriFed-GridGuard Comprehensive Experiment Runner")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("src/enhanced_federated_learning.py").exists():
        print("[ERROR] Please run this script from the project root directory")
        print("   (The directory containing src/ folder)")
        sys.exit(1)
    
    # Create and run experiments
    runner = ExperimentRunner()
    
    # Ask for confirmation
    print(f"Results will be saved to: {runner.experiment_root}")
    print(f"Configuration:")
    print(f"   - Federated clients: {runner.federated_config['n_clients']}")
    print(f"   - Federated rounds: {runner.federated_config['n_rounds']}")
    print(f"   - Data distribution: {runner.federated_config['distribution']}")
    print(f"   - Privacy budgets: {runner.privacy_budgets}")
    print(f"Estimated runtime: 2.5-3.5 hours")
    print()
    
    response = input("Proceed with experiments? (y/N): ").strip().lower()
    if response != 'y':
        print("[CANCELLED] Experiments cancelled")
        sys.exit(0)
    
    # Run all experiments
    runner.run_all_experiments()

if __name__ == "__main__":
    main() 