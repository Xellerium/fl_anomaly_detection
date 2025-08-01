"""
PriFed-GridGuard CLI Interface
Enhanced interface for multi-dataset federated learning with privacy mechanisms

Features:
- Multi-dataset support (MSU, Pecan Street, SGCC)
- Privacy-enhanced preprocessing options
- Privacy mechanism selection for federated learning
- Automated pipeline execution
- Publication-ready results generation
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
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

class PriFedGridGuardCLI:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.completed_steps = set()
        self.selected_datasets = []
        self.privacy_enabled = True
        
        # Available datasets
        self.datasets = {
            'msu': 'Mississippi State University Power System Attack Dataset',
            'pecan': 'Pecan Street Energy Consumption Dataset', 
            'sgcc': 'State Grid Corporation of China Electricity Theft Dataset'
        }
        
        # Privacy mechanisms
        self.privacy_mechanisms = {
            'ca_ldp': 'Context-Aware Local Differential Privacy',
            'cadp': 'Cluster-Adaptive Differential Privacy',
            's_he': 'Selective Homomorphic Encryption',
            'uans': 'Utility-Aware Noise Scheduler',
            'full': 'All Privacy Mechanisms Combined'
        }
        
        # Pipeline steps
        self.pipeline_steps = [
            {
                "id": 1, 
                "name": "Data Pipeline", 
                "desc": "Process datasets with privacy-enhanced preprocessing",
                "command": "python src/data_pipeline.py"
            },
            {
                "id": 2, 
                "name": "Baseline Models", 
                "desc": "Train and compare centralized models across datasets",
                "command": "python src/baseline_models.py"
            },
            {
                "id": 3, 
                "name": "Enhanced Federated Learning", 
                "desc": "Run privacy-enhanced federated learning experiments",
                "command": "python src/enhanced_federated_learning.py"
            },
            {
                "id": 4, 
                "name": "Analysis & Evaluation", 
                "desc": "Generate comprehensive results and publication figures",
                "command": "python src/analysis_and_evaluation.py"
            }
        ]
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print CLI header"""
        self.clear_screen()
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}PriFed-GridGuard: Privacy-Enhanced Federated Learning for Smart Grid Security{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print()
    
    def print_menu(self):
        """Display main menu"""
        self.print_header()
        
        print(f"{Colors.CYAN}Selected Datasets:{Colors.ENDC}", end=" ")
        if self.selected_datasets:
            print(", ".join([d.upper() for d in self.selected_datasets]))
        else:
            print("None (will use all datasets)")
        
        print(f"{Colors.CYAN}Privacy Mode:{Colors.ENDC} {'ENABLED' if self.privacy_enabled else 'DISABLED'}")
        print()
        
        print(f"{Colors.BOLD}MAIN MENU:{Colors.ENDC}")
        print("1. Configure Datasets")
        print("2. Configure Privacy Settings")
        print("3. Run Complete Pipeline")
        print("4. Run Individual Steps")
        print("5. View Results Summary")
        print("6. Clean Project Data")
        print("0. Exit")
        print()
    
    def configure_datasets(self):
        """Configure which datasets to use"""
        self.print_header()
        print(f"{Colors.BOLD}DATASET CONFIGURATION{Colors.ENDC}")
        print()
        
        print("Available Datasets:")
        for key, desc in self.datasets.items():
            selected = "✓" if key in self.selected_datasets else " "
            print(f"  [{selected}] {key.upper()}: {desc}")
        
        print("\nOptions:")
        print("1. Use all datasets (default)")
        print("2. Select specific datasets")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self.selected_datasets = []
            print(f"{Colors.GREEN}✓ Will use all datasets{Colors.ENDC}")
            time.sleep(1)
        elif choice == '2':
            print("\nEnter dataset codes separated by commas (e.g., msu,pecan):")
            selection = input().strip().lower()
            
            selected = []
            for ds in selection.split(','):
                ds = ds.strip()
                if ds in self.datasets:
                    selected.append(ds)
            
            if selected:
                self.selected_datasets = selected
                print(f"{Colors.GREEN}✓ Selected: {', '.join([d.upper() for d in selected])}{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}✗ No valid datasets selected{Colors.ENDC}")
            time.sleep(1)
    
    def configure_privacy(self):
        """Configure privacy settings"""
        self.print_header()
        print(f"{Colors.BOLD}PRIVACY CONFIGURATION{Colors.ENDC}")
        print()
        
        print("Privacy Options:")
        print(f"1. Enable privacy enhancements (current: {'ON' if self.privacy_enabled else 'OFF'})")
        print(f"2. Disable privacy enhancements")
        print("3. Configure epsilon value (differential privacy)")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
            
        if choice == '1':
            self.privacy_enabled = True
            print(f"{Colors.GREEN}✓ Privacy enhancements ENABLED{Colors.ENDC}")
            time.sleep(1)
        elif choice == '2':
            self.privacy_enabled = False
            print(f"{Colors.WARNING}⚠ Privacy enhancements DISABLED{Colors.ENDC}")
            time.sleep(1)
        elif choice == '3':
            print("\nEnter epsilon value (default: 1.0, smaller = more privacy):")
            try:
                epsilon = float(input().strip())
                if 0.1 <= epsilon <= 10.0:
                    self.epsilon = epsilon
                    print(f"{Colors.GREEN}✓ Epsilon set to {epsilon}{Colors.ENDC}")
                else:
                    print(f"{Colors.FAIL}✗ Epsilon must be between 0.1 and 10.0{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.FAIL}✗ Invalid epsilon value{Colors.ENDC}")
            time.sleep(1)
    
    def run_pipeline_step(self, step, extra_args=""):
        """Run a single pipeline step"""
        print(f"\n{Colors.CYAN}Running: {step['name']}{Colors.ENDC}")
        print(f"Description: {step['desc']}")
        print("-" * 60)
        
        # Build command with arguments
        cmd = step['command']
        
        # Add dataset selection if applicable
        if self.selected_datasets and step['id'] in [1, 2, 3]:
            if len(self.selected_datasets) == 1:
                cmd += f" --dataset {self.selected_datasets[0]}"
            # For multiple datasets, the scripts handle 'all' by default
        
        # Add privacy settings
        if step['id'] in [1, 2, 3]:
            if not self.privacy_enabled:
                cmd += " --no-privacy"
            elif hasattr(self, 'epsilon'):
                cmd += f" --epsilon {self.epsilon}"
        
        # Add any extra arguments
        if extra_args:
            cmd += f" {extra_args}"
        
        print(f"Command: {cmd}")
        print()
        
        # Run the command
        try:
            start_time = time.time()
            result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
            
            elapsed = time.time() - start_time
            print(f"\n{Colors.GREEN}✓ {step['name']} completed in {elapsed:.1f}s{Colors.ENDC}")
            
            self.completed_steps.add(step['id'])
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n{Colors.FAIL}✗ Error running {step['name']}{Colors.ENDC}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        self.print_header()
        print(f"{Colors.BOLD}RUNNING COMPLETE PIPELINE{Colors.ENDC}")
        print()
        
        datasets_str = ", ".join([d.upper() for d in self.selected_datasets]) if self.selected_datasets else "ALL"
        print(f"Datasets: {datasets_str}")
        print(f"Privacy: {'ENABLED' if self.privacy_enabled else 'DISABLED'}")
        print()
        
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != 'y':
            return
        
        start_time = time.time()
        
        # Run each step
        for step in self.pipeline_steps:
            if not self.run_pipeline_step(step):
                print(f"\n{Colors.FAIL}Pipeline stopped due to error{Colors.ENDC}")
                input("\nPress Enter to continue...")
                return
            time.sleep(1)
        
        total_time = time.time() - start_time
        print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.GREEN}✓ PIPELINE COMPLETED SUCCESSFULLY{Colors.ENDC}")
        print(f"{Colors.GREEN}Total time: {total_time/60:.1f} minutes{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_individual_steps(self):
        """Run individual pipeline steps"""
        while True:
            self.print_header()
            print(f"{Colors.BOLD}RUN INDIVIDUAL STEPS{Colors.ENDC}")
            print()
            
            for step in self.pipeline_steps:
                status = "✓" if step['id'] in self.completed_steps else " "
                print(f"{step['id']}. [{status}] {step['name']}")
                print(f"      {step['desc']}")
            
            print("\n0. Back to main menu")
            
            choice = input("\nSelect step to run: ").strip()
                
            if choice == '0':
                    break
            
            try:
                step_id = int(choice)
                step = next((s for s in self.pipeline_steps if s['id'] == step_id), None)
                
                if step:
                    # Check dependencies
                    if step_id == 2 and 1 not in self.completed_steps:
                        print(f"{Colors.WARNING}⚠ Warning: Data pipeline should be run first{Colors.ENDC}")
                        if input("Continue anyway? (y/n): ").strip().lower() != 'y':
                            continue
                    
                    if step_id == 3 and 2 not in self.completed_steps:
                        print(f"{Colors.WARNING}⚠ Warning: Baseline models should be run first{Colors.ENDC}")
                        if input("Continue anyway? (y/n): ").strip().lower() != 'y':
                            continue
                    
                    # Get additional arguments if needed
                    extra_args = ""
                    if step_id == 3:
                        print("\nFederated Learning Options:")
                        print("1. Standard configuration")
                        print("2. Custom configuration")
                        fl_choice = input("Select option (1): ").strip() or '1'
                        
                        if fl_choice == '2':
                            n_clients = input("Number of clients (10): ").strip() or '10'
                            n_rounds = input("Number of rounds (20): ").strip() or '20'
                            distribution = input("Distribution (iid/non-iid) [non-iid]: ").strip() or 'non-iid'
                            
                            extra_args = f"--n_clients {n_clients} --n_rounds {n_rounds} --distribution {distribution}"
                    
                    self.run_pipeline_step(step, extra_args)
                    input("\nPress Enter to continue...")
                else:
                    print(f"{Colors.FAIL}Invalid selection{Colors.ENDC}")
                    time.sleep(1)
                    
            except ValueError:
                print(f"{Colors.FAIL}Invalid input{Colors.ENDC}")
                time.sleep(1)
    
    def view_results_summary(self):
        """View results summary"""
        self.print_header()
        print(f"{Colors.BOLD}RESULTS SUMMARY{Colors.ENDC}")
        print()
        
        results_path = Path("results")
        
        # Check for result files
        files_to_check = [
            ("Baseline Results", "baseline_performance_summary.csv"),
            ("Best Models", "best_models_summary.csv"),
            ("Performance Comparison", "comprehensive_performance_comparison.csv"),
            ("Statistical Analysis", "statistical_analysis.csv"),
            ("Research Summary", "research_summary.txt")
        ]
        
        found_files = []
        for name, filename in files_to_check:
            filepath = results_path / filename
            if filepath.exists():
                found_files.append((name, filepath))
                print(f"{Colors.GREEN}✓{Colors.ENDC} {name}: {filename}")
            else:
                print(f"{Colors.WARNING}✗{Colors.ENDC} {name}: Not found")
        
        # Check for figures
        figures_path = results_path / "publication_figures"
        if figures_path.exists():
            figures = list(figures_path.glob("*.png"))
            print(f"\n{Colors.CYAN}Publication Figures:{Colors.ENDC}")
            for fig in figures[:5]:  # Show first 5
                print(f"  • {fig.name}")
            if len(figures) > 5:
                print(f"  ... and {len(figures)-5} more")
        
        print("\nOptions:")
        print("1. View research summary")
        print("2. Open results folder")
        print("0. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            summary_path = results_path / "research_summary.txt"
            if summary_path.exists():
                print(f"\n{Colors.CYAN}Research Summary:{Colors.ENDC}\n")
                with open(summary_path, 'r') as f:
                    content = f.read()
                    # Show first 50 lines
                    lines = content.split('\n')[:50]
                    print('\n'.join(lines))
                    if len(content.split('\n')) > 50:
                        print("\n... (truncated, see full file for complete summary)")
            else:
                print(f"{Colors.FAIL}Research summary not found. Run analysis first.{Colors.ENDC}")
            
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            # Open results folder in file explorer
            if sys.platform == 'win32':
                os.startfile(results_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', results_path])
            else:
                subprocess.run(['xdg-open', results_path])
    
    def clean_project_data(self):
        """Clean generated data and results"""
        self.print_header()
        print(f"{Colors.BOLD}CLEAN PROJECT DATA{Colors.ENDC}")
        print()
        
        print("This will remove:")
        print("• Processed datasets (data/processed/)")
        print("• Results (results/)")
        print("• Temporary files")
        print()
        print(f"{Colors.WARNING}⚠ This action cannot be undone!{Colors.ENDC}")
        
        confirm = input("\nProceed? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled")
            time.sleep(1)
            return
        
        # Clean directories
        dirs_to_clean = [
            Path("data/processed"),
            Path("results"),
            Path("catboost_info")
        ]
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                print(f"Cleaning {dir_path}...")
                import shutil
                shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Reset completed steps
        self.completed_steps.clear()
        
        print(f"\n{Colors.GREEN}✓ Project data cleaned{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main CLI loop"""
        while True:
            self.print_menu()
            
            choice = input("Select option: ").strip()
                
            if choice == '0':
                print(f"\n{Colors.CYAN}Thank you for using PriFed-GridGuard!{Colors.ENDC}")
                break
            elif choice == '1':
                self.configure_datasets()
            elif choice == '2':
                self.configure_privacy()
            elif choice == '3':
                self.run_complete_pipeline()
            elif choice == '4':
                self.run_individual_steps()
            elif choice == '5':
                self.view_results_summary()
            elif choice == '6':
                self.clean_project_data()
            else:
                print(f"{Colors.FAIL}Invalid option{Colors.ENDC}")
                time.sleep(1)
   

def main():
    """Run the CLI interface"""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check if running from project root
    if not Path("src").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
   
    # Create necessary directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # Run CLI
    cli = PriFedGridGuardCLI()
   
    try:
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
   main()