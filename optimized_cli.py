"""
Enhanced CLI Interface for Smart Grid Federated Learning Research
Integrated with enhanced federated learning capabilities and optimized code reuse
Now includes automatic best baseline model integration

Navigation:
1. Data Pipeline - Load and preprocess ARFF files
2. Baseline Models - Train centralized ML models  
3. Enhanced Federated Learning - Advanced distributed learning with baseline integration
4. Flower Federated Learning - Industry-standard federated learning framework
5. Analysis & Evaluation - Generate comprehensive results and figures
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util
import warnings

warnings.filterwarnings('ignore')

# Color codes for better UX
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class EnhancedFederatedLearningCLI:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.completed_steps = set()
        
        # Enhanced pipeline steps with new federated learning options
        self.pipeline_steps = [
            {
                "id": 1, 
                "name": "Data Pipeline", 
                "script": "src/data_pipeline.py", 
                "desc": "Load and preprocess smart grid dataset",
                "module": None
            },
            {
                "id": 2, 
                "name": "Baseline Models", 
                "script": "src/baseline_models.py", 
                "desc": "Train centralized ML models and identify optimal configuration",
                "module": None
            },
            {
                "id": 3, 
                "name": "Enhanced Federated Learning", 
                "script": "src/enhanced_federated_learning.py", 
                "desc": "Advanced distributed learning with automatic baseline integration",
                "module": None
            },
            {
                "id": 4, 
                "name": "Flower Federated Learning", 
                "script": "src/flower_federated_learning.py", 
                "desc": "Industry-standard federated learning framework validation",
                "module": None,
                "optional": True
            },
            {
                "id": 5, 
                "name": "Analysis & Evaluation", 
                "script": "src/analysis_and_evaluation.py", 
                "desc": "Generate comprehensive results and publication figures",
                "module": None
            }
        ]
        
        # Enhanced federated learning configurations
        self.federated_configs = {
            "standard": {
                "name": "Standard Federated Learning (IID)",
                "desc": "IID data distribution with best baseline model configuration",
                "params": {"experiment_type": "standard_iid_optimized"}
            },
            "non_iid": {
                "name": "Non-IID Federated Learning", 
                "desc": "Heterogeneous data distribution with baseline optimization",
                "params": {"experiment_type": "non_iid_optimized"}
            },
            "privacy": {
                "name": "Privacy-Preserving Federated Learning",
                "desc": "Differential privacy with multiple epsilon values",
                "params": {"experiment_type": "privacy_experiments"}
            },
            "comprehensive": {
                "name": "Comprehensive Analysis",
                "desc": "All configurations with baseline comparison and detailed analysis",
                "params": {"experiment_type": "comprehensive"}
            },
            "comparative": {
                "name": "Dual Framework Comparison",
                "desc": "Compare custom implementation with Flower framework",
                "params": {"experiment_type": "comparative_analysis"}
            }
        }
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        self.clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("╔" + "═" * 85 + "╗")
        print("║" + " " * 10 + "ENHANCED SMART GRID FEDERATED LEARNING WITH BASELINE INTEGRATION " + " " * 10 + "║")
        print("╚" + "═" * 85 + "╝")
        print(f"{Colors.ENDC}")
        
    def print_pipeline_status(self):
        print(f"{Colors.CYAN}Enhanced Experimental Pipeline Status:{Colors.ENDC}")
        print("─" * 70)
        
        for step in self.pipeline_steps:
            status = "✓" if step["id"] in self.completed_steps else "○"
            color = Colors.GREEN if step["id"] in self.completed_steps else Colors.WARNING
            optional_marker = " (Optional)" if step.get("optional", False) else ""
            
            print(f"{color}{status}{Colors.ENDC} {step['id']}. {step['name']}{optional_marker}")
            print(f"   {step['desc']}")
            
        print("─" * 70)
        required_steps = len([s for s in self.pipeline_steps if not s.get("optional", False)])
        completed_required = len([s for s in self.pipeline_steps if s["id"] in self.completed_steps and not s.get("optional", False)])
        print(f"Progress: {completed_required}/{required_steps} core steps completed ({len(self.completed_steps)}/{len(self.pipeline_steps)} total)")
        
        # Show baseline integration status
        self.show_baseline_integration_status()
        print()
        
    def show_baseline_integration_status(self):
        """Display baseline model integration status"""
        results_path = self.project_root / "results"
        baseline_file = results_path / "baseline_performance_summary.csv"
        
        if baseline_file.exists():
            try:
                import pandas as pd
                baseline_df = pd.read_csv(baseline_file)
                best_model = baseline_df.loc[baseline_df['Validation F1-Score'].idxmax(), 'Model']
                best_f1 = baseline_df['Validation F1-Score'].max()
                print(f"\n{Colors.GREEN}📊 Baseline Integration Ready:{Colors.ENDC}")
                print(f"   Best Model: {best_model} (F1: {best_f1:.4f})")
                print(f"   Configuration will be automatically applied to federated learning")
            except Exception:
                print(f"\n{Colors.WARNING}📊 Baseline Integration: Available but not parsed{Colors.ENDC}")
        else:
            print(f"\n{Colors.WARNING}📊 Baseline Integration: Run Baseline Models first{Colors.ENDC}")
    
    def load_module_dynamically(self, script_path):
        """Dynamically load a Python module for direct execution"""
        try:
            spec = importlib.util.spec_from_file_location("dynamic_module", script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["dynamic_module"] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"{Colors.FAIL}Error loading module {script_path}: {str(e)}{Colors.ENDC}")
            return None
    
    def execute_step(self, step_id, config_params=None):
        """Execute a specific pipeline step with optional configuration"""
        step = next((s for s in self.pipeline_steps if s["id"] == step_id), None)
        if not step:
            print(f"{Colors.FAIL}Invalid step ID{Colors.ENDC}")
            return False
        
        script_path = self.project_root / step["script"]
        if not script_path.exists():
            print(f"{Colors.FAIL}Script not found: {step['script']}{Colors.ENDC}")
            return False
        
        print(f"\n{Colors.BLUE}Executing: {step['name']}{Colors.ENDC}")
        print(f"Description: {step['desc']}")
        print(f"Script: {step['script']}")
        if config_params:
            print(f"Configuration: {config_params}")
        print("─" * 70)
        
        start_time = time.time()
        
        try:
            # For enhanced federated learning, use direct module execution with parameters
            if step_id == 3 and config_params:
                return self.execute_enhanced_federated_learning(script_path, config_params)
            
            # For other steps, use subprocess execution
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=False, text=True, cwd=self.project_root)
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                self.completed_steps.add(step_id)
                print(f"\n{Colors.GREEN}✓ Step completed successfully in {execution_time:.1f} seconds{Colors.ENDC}")
                return True
            else:
                print(f"\n{Colors.FAIL}✗ Step failed with return code {result.returncode}{Colors.ENDC}")
                return False
                
        except Exception as e:
            print(f"\n{Colors.FAIL}✗ Error executing step: {str(e)}{Colors.ENDC}")
            return False
    
    def execute_enhanced_federated_learning(self, script_path, config_params):
        """Execute enhanced federated learning with specific configuration"""
        try:
            # Add src to path for imports
            sys.path.insert(0, str(self.project_root / "src"))
            
            # Import the enhanced federated learning module
            module = self.load_module_dynamically(script_path)
            if not module:
                return False
            
            # Check if baseline models are available
            results_path = self.project_root / "results"
            baseline_available = (results_path / "baseline_performance_summary.csv").exists()
            
            if not baseline_available:
                print(f"{Colors.WARNING}Warning: Baseline models not found. Run step 2 first for optimal performance.{Colors.ENDC}")
                proceed = input(f"{Colors.WARNING}Continue with default parameters? (y/N): {Colors.ENDC}").strip().lower()
                if proceed != 'y':
                    return False
            
            # Execute based on configuration
            experiment_type = config_params.get("experiment_type", "comprehensive")
            
            print(f"{Colors.CYAN}Executing {experiment_type} with baseline integration...{Colors.ENDC}")
            
            if experiment_type == "comprehensive":
                # Run comprehensive experiment with baseline integration
                experiment = module.EnhancedFederatedExperiment(model_type="neural_network")
                results = experiment.run_complete_experiment()
                
                # Also run Random Forest for comparison
                print(f"\n{Colors.CYAN}Running Random Forest comparison...{Colors.ENDC}")
                experiment_rf = module.EnhancedFederatedExperiment(model_type="random_forest")
                results_rf = experiment_rf.run_complete_experiment()
                
            elif experiment_type == "comparative_analysis":
                # Run both custom and Flower implementations
                experiment = module.EnhancedFederatedExperiment(model_type="neural_network")
                results = experiment.run_comparative_analysis()
                
                # Try to run Flower comparison if available
                try:
                    flower_script = self.project_root / "src/flower_federated_learning.py"
                    if flower_script.exists():
                        print(f"\n{Colors.CYAN}Running Flower framework comparison...{Colors.ENDC}")
                        flower_module = self.load_module_dynamically(flower_script)
                        if flower_module:
                            flower_results = flower_module.run_flower_federated_learning()
                except Exception as e:
                    print(f"{Colors.WARNING}Flower comparison failed: {e}{Colors.ENDC}")
                
            else:
                # Run specific experiment type
                experiment = module.EnhancedFederatedExperiment(model_type="neural_network")
                
                if not experiment.load_data():
                    return False
                
                # Setup with baseline integration
                if experiment_type == "standard_iid_optimized":
                    experiment.setup_federated_system_with_best_baseline(
                        num_clients=5, privacy_budget=1.0, non_iid=False)
                elif experiment_type == "non_iid_optimized":
                    experiment.setup_federated_system_with_best_baseline(
                        num_clients=5, privacy_budget=1.0, non_iid=True)
                elif experiment_type == "privacy_experiments":
                    # Run multiple privacy experiments
                    privacy_results = {}
                    for epsilon in [0.5, 1.0, 5.0]:
                        print(f"\n{Colors.CYAN}Privacy experiment: ε={epsilon}{Colors.ENDC}")
                        experiment.setup_federated_system_with_best_baseline(
                            num_clients=5, privacy_budget=epsilon, non_iid=False)
                        privacy_training = experiment.run_federated_training(
                            num_rounds=10, apply_privacy=True)
                        privacy_results[f"epsilon_{epsilon}"] = privacy_training
                    
                    experiment.save_results({"privacy_experiments": privacy_results})
                else:
                    # Default to standard experiment
                    experiment.setup_federated_system_with_best_baseline(
                        num_clients=5, privacy_budget=1.0, non_iid=False)
                    results = experiment.run_federated_training(num_rounds=15, apply_privacy=False)
                    experiment.save_results({"experiment_results": results})
            
            self.completed_steps.add(3)
            print(f"\n{Colors.GREEN}✓ Enhanced federated learning with baseline integration completed successfully{Colors.ENDC}")
            
            # Show baseline integration summary
            if baseline_available:
                self.show_integration_summary()
            
            return True
            
        except Exception as e:
            print(f"\n{Colors.FAIL}✗ Enhanced federated learning failed: {str(e)}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up sys.path
            if str(self.project_root / "src") in sys.path:
                sys.path.remove(str(self.project_root / "src"))
    
    def show_integration_summary(self):
        """Show summary of baseline integration results"""
        try:
            import pandas as pd
            results_path = self.project_root / "results"
            
            # Load baseline performance
            baseline_df = pd.read_csv(results_path / "baseline_performance_summary.csv")
            best_model = baseline_df.loc[baseline_df['Validation F1-Score'].idxmax(), 'Model']
            best_acc = baseline_df['Validation Accuracy'].max()
            best_f1 = baseline_df['Validation F1-Score'].max()
            
            print(f"\n{Colors.GREEN}📊 BASELINE INTEGRATION SUMMARY:{Colors.ENDC}")
            print(f"   Optimal Model Applied: {best_model}")
            print(f"   Baseline Performance: Acc={best_acc:.4f}, F1={best_f1:.4f}")
            print(f"   Federated learning now uses these optimal parameters")
            
        except Exception as e:
            print(f"{Colors.WARNING}Could not load integration summary: {e}{Colors.ENDC}")
    
    def show_federated_learning_menu(self):
        """Interactive menu for enhanced federated learning configurations"""
        while True:
            self.print_header()
            print(f"{Colors.BOLD}Enhanced Federated Learning Configurations:{Colors.ENDC}")
            print("─" * 70)
            
            for i, (key, config) in enumerate(self.federated_configs.items(), 1):
                print(f"{Colors.CYAN}{i}.{Colors.ENDC} {config['name']}")
                print(f"     {config['desc']}")
                print()
            
            print(f"{Colors.WARNING}0. Return to Main Menu{Colors.ENDC}")
            
            try:
                choice = int(input(f"\n{Colors.BLUE}Select configuration (0-{len(self.federated_configs)}): {Colors.ENDC}"))
                
                if choice == 0:
                    break
                elif 1 <= choice <= len(self.federated_configs):
                    config_key = list(self.federated_configs.keys())[choice - 1]
                    config = self.federated_configs[config_key]
                    
                    print(f"\n{Colors.CYAN}Selected: {config['name']}{Colors.ENDC}")
                    print(f"Description: {config['desc']}")
                    
                    # Check baseline availability
                    baseline_available = (self.project_root / "results" / "baseline_performance_summary.csv").exists()
                    if baseline_available:
                        print(f"{Colors.GREEN}✓ Baseline models available - optimal configuration will be applied{Colors.ENDC}")
                    else:
                        print(f"{Colors.WARNING}⚠ Baseline models not found - default parameters will be used{Colors.ENDC}")
                        print(f"   Recommendation: Run 'Baseline Models' first for optimal performance")
                    
                    confirm = input(f"\n{Colors.WARNING}Execute this configuration? (y/N): {Colors.ENDC}").strip().lower()
                    if confirm == 'y':
                        success = self.execute_step(3, config['params'])
                        if success:
                            print(f"\n{Colors.GREEN}🎯 Configuration completed successfully!{Colors.ENDC}")
                            print(f"Results saved to results/federated_learning/")
                        self.wait_for_input()
                else:
                    print(f"{Colors.FAIL}Invalid choice{Colors.ENDC}")
                    time.sleep(1)
                    
            except ValueError:
                print(f"{Colors.FAIL}Please enter a valid number{Colors.ENDC}")
                time.sleep(1)
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        dependencies = {
            "pandas": "Data manipulation and analysis",
            "numpy": "Numerical computing",
            "sklearn": "Machine learning algorithms",
            "matplotlib": "Plotting and visualization"
        }
        
        optional_dependencies = {
            "torch": "PyTorch for neural networks",
            "flwr": "Flower framework for FL"
        }
        
        missing_required = []
        missing_optional = []
        
        # Check required dependencies
        for dep, desc in dependencies.items():
            try:
                __import__(dep)
            except ImportError:
                missing_required.append((dep, desc))
        
        # Check optional dependencies
        for dep, desc in optional_dependencies.items():
            try:
                __import__(dep)
            except ImportError:
                missing_optional.append((dep, desc))
        
        if missing_required:
            print(f"{Colors.FAIL}Missing Required Dependencies:{Colors.ENDC}")
            for dep, desc in missing_required:
                print(f"  - {dep}: {desc}")
            print(f"\nInstall with: pip install pandas numpy scikit-learn matplotlib")
            return False
        
        if missing_optional:
            print(f"{Colors.WARNING}Optional Dependencies Not Available:{Colors.ENDC}")
            for dep, desc in missing_optional:
                print(f"  - {dep}: {desc}")
            print(f"\nFor full functionality: pip install torch flwr")
        
        return True
    
    def run_complete_pipeline(self):
        """Execute complete enhanced pipeline with baseline integration"""
        print(f"\n{Colors.BOLD}Running Complete Enhanced Research Pipeline{Colors.ENDC}")
        print("This pipeline automatically integrates the best baseline model with federated learning.")
        print("\nPipeline Overview:")
        for step in self.pipeline_steps:
            optional_marker = " (Optional)" if step.get("optional", False) else ""
            print(f"  {step['id']}. {step['name']}{optional_marker}")
            print(f"      → {step['desc']}")
        
        print(f"\n{Colors.CYAN}Key Enhancement: Baseline Integration{Colors.ENDC}")
        print("  → Step 2 identifies the optimal model configuration")
        print("  → Step 3 automatically applies this configuration to federated learning")
        print("  → Results show performance with optimal vs. default parameters")
        
        confirm = input(f"\n{Colors.WARNING}Continue with complete pipeline? (y/N): {Colors.ENDC}").strip().lower()
        if confirm != 'y':
            return
        
        # Check dependencies
        if not self.check_dependencies():
            print(f"{Colors.FAIL}Please install missing dependencies before proceeding{Colors.ENDC}")
            self.wait_for_input()
            return
        
        print(f"\n{Colors.CYAN}Starting complete enhanced pipeline execution...{Colors.ENDC}")
        
        failed_steps = []
        for step in self.pipeline_steps:
            # Skip optional steps if they fail
            if step.get("optional", False):
                print(f"\n{Colors.WARNING}Note: {step['name']} is optional and may be skipped if dependencies are missing{Colors.ENDC}")
            
            print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
            print(f"{Colors.HEADER}STEP {step['id']}: {step['name'].upper()}{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
            
            # Use comprehensive configuration for enhanced federated learning
            if step["id"] == 3:
                success = self.execute_step(step["id"], self.federated_configs["comprehensive"]["params"])
            else:
                success = self.execute_step(step["id"])
            
            if not success:
                if step.get("optional", False):
                    print(f"{Colors.WARNING}Optional step failed, continuing with pipeline...{Colors.ENDC}")
                else:
                    failed_steps.append(step["name"])
                    break
            
            time.sleep(2)
        
        print(f"\n{Colors.BOLD}Enhanced Pipeline Execution Summary:{Colors.ENDC}")
        if not failed_steps:
            print(f"{Colors.GREEN}✓ All core steps completed successfully!{Colors.ENDC}")
            print(f"{Colors.GREEN}✓ Baseline integration applied automatically{Colors.ENDC}")
            print(f"{Colors.CYAN}📁 Check the results/ directory for comprehensive outputs{Colors.ENDC}")
            print(f"{Colors.CYAN}📊 Run the visualization notebook for publication-quality figures{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ Pipeline stopped at: {failed_steps[0]}{Colors.ENDC}")
        
        self.wait_for_input()
    
    def show_results_summary(self):
        """Enhanced results summary including baseline integration status"""
        print(f"\n{Colors.BOLD}ENHANCED EXPERIMENTAL RESULTS SUMMARY{Colors.ENDC}")
        print("─" * 70)
        
        results_dir = self.project_root / "results"
        if not results_dir.exists():
            print(f"{Colors.WARNING}No results directory found. Run experiments first.{Colors.ENDC}")
            self.wait_for_input()
            return
        
        # Enhanced result files check
        result_files = {
            "Baseline Performance": results_dir / "baseline_performance_summary.csv",
            "Enhanced Federated Results": results_dir / "federated_learning" / "enhanced_federated_results.pkl",
            "Training History": results_dir / "federated_learning" / "training_history.pkl",
            "Baseline Comparison": results_dir / "federated_learning" / "federated_comparison.pkl",
            "Flower Results": results_dir / "federated_learning" / "flower_federated_results.pkl",
            "Analysis Results": results_dir / "analysis" / "performance_comparison_table.csv",
            "Experiment Summary": results_dir / "federated_learning" / "experiment_summary.txt"
        }
        
        print(f"{Colors.CYAN}Available Enhanced Results:{Colors.ENDC}")
        available_count = 0
        for name, path in result_files.items():
            if path.exists():
                available_count += 1
                size_mb = path.stat().st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"{Colors.GREEN}✓{Colors.ENDC} {name}: {size_mb:.1f}MB (Modified: {mod_time})")
            else:
                print(f"{Colors.WARNING}○{Colors.ENDC} {name}: Not available")
        
        print(f"\n{Colors.CYAN}Results Summary: {available_count}/{len(result_files)} files available{Colors.ENDC}")
        
        # Show baseline integration status and performance
        baseline_file = results_dir / "baseline_performance_summary.csv"
        if baseline_file.exists():
            try:
                import pandas as pd
                baseline_df = pd.read_csv(baseline_file)
                best_model = baseline_df.loc[baseline_df['Validation F1-Score'].idxmax(), 'Model']
                best_f1 = baseline_df['Validation F1-Score'].max()
                best_acc = baseline_df['Validation Accuracy'].max()
                
                print(f"\n{Colors.GREEN}📊 BASELINE INTEGRATION STATUS:{Colors.ENDC}")
                print(f"   Best Model: {best_model}")
                print(f"   Performance: Acc={best_acc:.4f}, F1={best_f1:.4f}")
                print(f"   Status: {'✓ Applied to federated learning' if 3 in self.completed_steps else '○ Ready for application'}")
            except Exception as e:
                print(f"\n{Colors.WARNING}Could not parse baseline results: {e}{Colors.ENDC}")
        
        # Show experiment summary if available
        experiment_summary = results_dir / "federated_learning" / "experiment_summary.txt"
        if experiment_summary.exists():
            print(f"\n{Colors.CYAN}Latest Experiment Preview:{Colors.ENDC}")
            try:
                with open(experiment_summary, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:15]  # Show first 15 lines
                    for line in lines:
                        if line.strip() and not line.startswith("="):
                            print(f"  {line.strip()}")
            except Exception as e:
                print(f"  Error reading experiment summary: {e}")
        
        self.wait_for_input()
    
    def wait_for_input(self):
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
    
    def run_main_menu(self):
        """Enhanced main application loop"""
        while True:
            self.print_header()
            self.print_pipeline_status()
            
            print(f"{Colors.BOLD}Enhanced Main Menu Options:{Colors.ENDC}")
            print("─" * 50)
            print(f"{Colors.GREEN}1.{Colors.ENDC} Run Complete Enhanced Pipeline")
            print(f"{Colors.BLUE}2.{Colors.ENDC} Execute Individual Steps")
            print(f"{Colors.CYAN}3.{Colors.ENDC} Enhanced Federated Learning Menu")
            print(f"{Colors.HEADER}4.{Colors.ENDC} View Enhanced Results Summary")
            print(f"{Colors.WARNING}5.{Colors.ENDC} Enhanced Project Overview")
            print(f"{Colors.HEADER}6.{Colors.ENDC} Code Structure Explanation")
            print(f"{Colors.FAIL}0.{Colors.ENDC} Exit")
            
            try:
                choice = int(input(f"\n{Colors.BLUE}Select option (0-6): {Colors.ENDC}"))
                
                if choice == 0:
                    print(f"\n{Colors.CYAN}Thank you for using the Enhanced Smart Grid FL Research Framework!{Colors.ENDC}")
                    break
                elif choice == 1:
                    self.run_complete_pipeline()
                elif choice == 2:
                    self.show_individual_steps_menu()
                elif choice == 3:
                    self.show_federated_learning_menu()
                elif choice == 4:
                    self.show_results_summary()
                elif choice == 5:
                    self.show_project_overview()
                elif choice == 6:
                    self.show_code_explanation()
                else:
                    print(f"{Colors.FAIL}Invalid choice{Colors.ENDC}")
                    time.sleep(1)
                    
            except ValueError:
                print(f"{Colors.FAIL}Please enter a valid number{Colors.ENDC}")
                time.sleep(1)
            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}Operation cancelled by user{Colors.ENDC}")
                break
    
    def show_individual_steps_menu(self):
        """Enhanced menu for executing individual pipeline steps"""
        while True:
            self.print_header()
            self.print_pipeline_status()
            
            print(f"{Colors.BOLD}Individual Step Execution:{Colors.ENDC}")
            print("─" * 60)
            
            for step in self.pipeline_steps:
                status = "✓" if step["id"] in self.completed_steps else " "
                color = Colors.GREEN if step["id"] in self.completed_steps else Colors.ENDC
                optional_marker = " (Optional)" if step.get("optional", False) else ""
                
                print(f"{color}{step['id']}.{status} {step['name']}{optional_marker}{Colors.ENDC}")
                print(f"     {step['desc']}")
            
            print(f"\n{Colors.CYAN}Special Options:{Colors.ENDC}")
            print(f"{Colors.CYAN}6. Enhanced Federated Learning Menu{Colors.ENDC}")
            print(f"     Configure and run specific federated learning experiments")
            print(f"\n{Colors.WARNING}0. Return to Main Menu{Colors.ENDC}")
            
            try:
                choice = int(input(f"\n{Colors.BLUE}Select option (0-6): {Colors.ENDC}"))
                
                if choice == 0:
                    break
                elif choice == 6:
                    self.show_federated_learning_menu()
                elif 1 <= choice <= len(self.pipeline_steps):
                    if choice == 3:  # Enhanced federated learning
                        self.show_federated_learning_menu()
                    else:
                        self.execute_step(choice)
                        self.wait_for_input()
                else:
                    print(f"{Colors.FAIL}Invalid choice{Colors.ENDC}")
                    time.sleep(1)
                    
            except ValueError:
                print(f"{Colors.FAIL}Please enter a valid number{Colors.ENDC}")
                time.sleep(1)
   
    def show_project_overview(self):
       """Enhanced project overview including baseline integration capabilities"""
       print(f"\n{Colors.BOLD}ENHANCED PROJECT OVERVIEW FOR PhD SUPERVISION{Colors.ENDC}")
       print("─" * 70)
       
       print(f"{Colors.CYAN}Research Objectives:{Colors.ENDC}")
       objectives = [
           "Develop advanced federated learning framework for smart grid anomaly detection",
           "Integrate optimal centralized model configurations with distributed learning", 
           "Compare centralized vs multiple distributed learning approaches",
           "Implement comprehensive privacy-preserving mechanisms with differential privacy",
           "Analyze privacy-utility trade-offs across different federated configurations",
           "Evaluate custom implementations against industry-standard frameworks (Flower)",
           "Generate publication-ready experimental results with statistical significance"
       ]
       for obj in objectives:
           print(f"  • {obj}")
       
       print(f"\n{Colors.CYAN}Enhanced Technical Approach:{Colors.ENDC}")
       print("  • Baseline Model Selection: Automatic identification of optimal configuration")
       print("  • Model Integration: Best baseline parameters applied to federated learning")
       print("  • Neural Network Architecture: Multi-layer perceptron with adaptive learning")
       print("  • Tree-based Models: Random Forest, XGBoost, LightGBM, CatBoost optimization")
       print("  • Federated Configurations: IID, Non-IID, Privacy-preserving, Comprehensive")
       print("  • Aggregation Methods: FedAvg, weighted averaging, differential privacy")
       print("  • Privacy Mechanisms: Configurable epsilon budgets, Laplace noise injection")
       print("  • Framework Validation: Custom implementation vs. Flower comparison")
       
       print(f"\n{Colors.CYAN}Baseline Integration Innovation:{Colors.ENDC}")
       print("  • Automatic Best Model Detection: F1-score based selection from baseline results")
       print("  • Parameter Transfer: Optimal hyperparameters applied to federated clients")
       print("  • Performance Inheritance: Federated learning starts with proven configurations")
       print("  • Dynamic Optimization: Adaptive learning rates based on centralized performance")
       print("  • Comparative Analysis: Direct comparison of optimized vs. default parameters")
       
       print(f"\n{Colors.CYAN}Enhanced Experimental Design:{Colors.ENDC}")
       print("  • Baseline-Informed Federated Learning: Uses optimal centralized configuration")
       print("  • Multiple federated learning configurations for comprehensive analysis")
       print("  • Comparison between custom implementation and industry-standard Flower")
       print("  • Statistical validation through cross-validation and multiple runs")
       print("  • Performance metrics: Accuracy, F1-score, training time, privacy cost")
       print("  • Dynamic visualization: Publication-quality figures that update with experiments")
       
       print(f"\n{Colors.CYAN}Expected Enhanced Contributions:{Colors.ENDC}")
       contributions = [
           "Novel baseline-federated integration methodology for optimal performance",
           "First comprehensive FL study on power system attack dataset with baseline optimization",
           "Privacy-preserving anomaly detection with mathematical performance guarantees", 
           "Performance comparison of modern ML algorithms in optimized FL settings",
           "Framework-agnostic validation across custom and industry-standard implementations",
           "Dynamic experimental pipeline with automatic configuration optimization"
       ]
       for contrib in contributions:
           print(f"  • {contrib}")
       
       self.wait_for_input()
   
    def show_code_explanation(self):
       """Enhanced code structure explanation including baseline integration"""
       print(f"\n{Colors.BOLD}ENHANCED CODE STRUCTURE FOR SUPERVISION{Colors.ENDC}")
       print("─" * 70)
       
       explanations = {
           "src/enhanced_federated_learning.py": [
               "Purpose: Advanced federated learning with automatic baseline integration",
               "Key Enhancement: load_best_baseline_model() function for optimal configuration",
               "Key Classes:",
               "  • AdvancedFederatedClient - Enhanced local training with baseline parameters",
               "  • AdvancedFederatedServer - Proper FedAvg with baseline-informed initialization",
               "  • EnhancedFederatedExperiment - Comprehensive experimental framework",
               "Baseline Integration: setup_federated_system_with_best_baseline() method",
               "Academic Significance: Bridges centralized and federated learning optimally"
           ],
           
           "src/flower_federated_learning.py": [
               "Purpose: Industry-standard federated learning using Flower framework",
               "Integration: Can utilize baseline model configurations for validation",
               "Key Components:",
               "  • SmartGridNN - PyTorch neural network for smart grid classification",
               "  • SmartGridClient - Flower client with baseline-informed parameters", 
               "  • Server strategy - FedAvg with configurable baseline integration",
               "Key Features: Simulation support, advanced aggregation, scalability testing",
               "Academic Significance: Validates custom implementation against industry standard"
           ],
           
           "visualization_notebook.ipynb": [
               "Purpose: Dynamic publication-quality visualization generation",
               "Key Enhancement: Automatic integration with actual experimental results",
               "Dynamic Features:",
               "  • load_experimental_data() - Detects and loads all result formats",
               "  • generate_dynamic_confusion_matrices() - Uses actual model predictions",
               "  • extract_dynamic_training_history() - Real federated learning convergence",
               "Baseline Integration: Automatically shows performance comparisons",
               "Academic Significance: Eliminates manual data entry, ensures accuracy"
           ]
       }
       
       for script, details in explanations.items():
           print(f"\n{Colors.CYAN}{script}:{Colors.ENDC}")
           for detail in details:
               print(f"  {detail}")
       
       print(f"\n{Colors.CYAN}Enhanced Quality Features:{Colors.ENDC}")
       features = [
           "Automatic baseline model detection and parameter extraction",
           "Seamless integration between centralized and federated learning phases",
           "Proper federated averaging algorithms with mathematical rigor",
           "Comprehensive privacy mechanisms with differential privacy guarantees",
           "Multiple experimental configurations for thorough evaluation",
           "Industry-standard framework integration for validation",
           "Dynamic visualization pipeline that updates with experimental results",
           "Robust error handling and comprehensive logging throughout the pipeline"
       ]
       for feature in features:
           print(f"  • {feature}")
       
       print(f"\n{Colors.CYAN}Integration Workflow:{Colors.ENDC}")
       workflow = [
           "1. Baseline Models: Train centralized models and identify optimal configuration",
           "2. Parameter Extraction: Extract hyperparameters from best performing model",
           "3. Federated Setup: Initialize federated system with optimal baseline parameters",
           "4. Enhanced Training: Run federated learning with baseline-informed configuration",
           "5. Dynamic Comparison: Automatically compare federated vs. centralized performance",
           "6. Visualization: Generate figures showing baseline integration benefits"
       ]
       for step in workflow:
           print(f"  {step}")
       
       self.wait_for_input()

def main():
   """Enhanced application entry point"""
   if sys.version_info < (3, 7):
       print(f"{Colors.FAIL}Error: Python 3.7 or higher is required{Colors.ENDC}")
       sys.exit(1)
   
   cli = EnhancedFederatedLearningCLI()
   
   try:
       cli.run_main_menu()
   except Exception as e:
       print(f"{Colors.FAIL}Unexpected error: {str(e)}{Colors.ENDC}")
       sys.exit(1)

if __name__ == "__main__":
   main()