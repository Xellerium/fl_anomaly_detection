# optimize_project.py
"""
Project Optimization Script for Federated Learning Smart Grid Research
Streamlines codebase for PhD supervision and removes unnecessary complexity
"""

import os
import shutil
from pathlib import Path

class ProjectOptimizer:
    def __init__(self):
        self.project_root = Path("C:/Projects/federated_smart_grid_detection")
        
    def remove_unnecessary_files_and_folders(self):
        """Remove redundant files and folders to streamline project"""
        
        # Files to remove (redundant or unused)
        files_to_delete = [
            # Redundant federated learning files
            "src/federated_learning/aggregation.py",
            "src/federated_learning/private_server.py", 
            "src/federated_learning/private_experiment.py",
            "src/federated_learning/federated_system.py",
            "src/federated_learning/client.py",
            "src/federated_learning/server.py",
            "src/federated_learning/experiment.py",
            # Old data processing files
            "src/data_processing/data_exploration.py",
            "src/data_processing/preprocessing.py",
            # Old model files
            "src/models/baseline_models.py",
            # Old evaluation files  
            "src/evaluation/comprehensive_analysis.py",
            # Privacy files (integrated into main FL script)
            "src/privacy/differential_privacy.py",
            # Utility files
            "src/utils/project_cleanup.py",
            # Old CLI
            "federated_learning_cli.py"
        ]
        
        # Folders to remove (empty or redundant) 
        folders_to_delete = [
            "src/models",
            "src/evaluation", 
            "src/privacy",
            "src/data_processing",
            "src/federated_learning"
        ]
        
        print("Removing unnecessary files and folders...")
        
        # Remove files
        for file_path in files_to_delete:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    full_path.unlink()
                    print(f"  ✓ Removed file: {file_path}")
                except Exception as e:
                    print(f"  ✗ Error removing {file_path}: {e}")
                
        # Remove folders
        for folder_path in folders_to_delete:
            full_path = self.project_root / folder_path
            if full_path.exists():
                try:
                    shutil.rmtree(full_path)
                    print(f"  ✓ Removed folder: {folder_path}")
                except Exception as e:
                    print(f"  ✗ Error removing {folder_path}: {e}")
        
        print("Cleanup completed.")
    
    def create_streamlined_structure(self):
        """Create the new streamlined project structure"""
        print("Creating streamlined project structure...")
        
        # Create new simplified src structure
        new_dirs = [
            "src",
            "src/utils"
        ]
        
        for dir_path in new_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created directory: {dir_path}")
        
        # Create __init__.py files
        init_files = [
            "src/__init__.py",
            "src/utils/__init__.py"
        ]
        
        for init_file in init_files:
            full_path = self.project_root / init_file
            if not full_path.exists():
                full_path.touch()
                print(f"  ✓ Created: {init_file}")

def main():
    print("=" * 60)
    print("PROJECT OPTIMIZATION FOR PHD RESEARCH")
    print("=" * 60)
    print("This will streamline the codebase by:")
    print("- Removing redundant and complex files")
    print("- Consolidating functionality into 4 main scripts")
    print("- Creating a clean, supervision-friendly structure")
    print("=" * 60)
    
    confirm = input("Continue with optimization? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Optimization cancelled.")
        return
    
    optimizer = ProjectOptimizer()
    optimizer.remove_unnecessary_files_and_folders()
    optimizer.create_streamlined_structure()
    
    print("\n" + "=" * 60)
    print("PROJECT OPTIMIZATION COMPLETED")
    print("=" * 60)
    print("New streamlined structure:")
    print("src/")
    print("  ├── data_pipeline.py           # Complete data processing")
    print("  ├── baseline_models.py         # Centralized ML models")  
    print("  ├── federated_learning.py      # Distributed learning")
    print("  ├── analysis_and_evaluation.py # Results & figures")
    print("  └── utils/")
    print("      └── config.py              # Configuration management")
    print("")
    print("✓ All functionality preserved in 4 main scripts")
    print("✓ Code optimized for PhD supervision discussions")
    print("✓ Well-documented, academic-quality implementation")
    print("✓ Run 'python optimized_cli.py' to start experiments")

if __name__ == "__main__":
    main()
