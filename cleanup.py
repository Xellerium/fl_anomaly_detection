"""
Enhanced Clean Results Script for Smart Grid Federated Learning Pipeline
Removes all generated files including new enhanced federated learning outputs

This script provides complete cleanup while preserving the enhanced pipeline structure
"""

import os
import shutil
from pathlib import Path
import glob

def clean_pipeline_results():
    """Remove all generated files from the enhanced federated learning pipeline"""
    
    project_root = Path.cwd()
    
    # Enhanced cleanup targets including new file types
    cleanup_targets = {
        'results': [
            '*.csv',
            '*.pkl', 
            '*.png',
            '*.pdf',
            '*.txt'
        ],
        'data/processed': [
            '*.pkl',
            '*.csv'
        ],
        'data/splits': [
            '*.pkl'
        ]
    }
    
    # Additional specific files and directories including enhanced outputs
    specific_cleanup = [
        'results/baseline_performance_summary.csv',
        'results/baseline_results.pkl',
        'results/models',
        'results/federated_learning',
        'results/analysis',
        'results/figures',
        'results/publication_figures',
        'results/private_federated_learning',
        'results/final_analysis',
        'results/logs',
        'catboost_info'
    ]
    
    deleted_files = []
    deleted_dirs = []
    
    print("Starting enhanced pipeline cleanup...")
    print("=" * 50)
    
    # Clean files by pattern in specific directories
    for directory, patterns in cleanup_targets.items():
        dir_path = project_root / directory
        
        if dir_path.exists():
            print(f"\nCleaning directory: {directory}")
            
            for pattern in patterns:
                files_to_delete = list(dir_path.glob(pattern))
                
                for file_path in files_to_delete:
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            deleted_files.append(str(file_path.relative_to(project_root)))
                            print(f"  Deleted: {file_path.name}")
                        except Exception as e:
                            print(f"  Error deleting {file_path.name}: {e}")
                
                # Also check subdirectories
                subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                for subdir in subdirs:
                    subdir_files = list(subdir.glob(pattern))
                    for file_path in subdir_files:
                        if file_path.is_file():
                            try:
                                file_path.unlink()
                                deleted_files.append(str(file_path.relative_to(project_root)))
                                print(f"  Deleted: {file_path.relative_to(dir_path)}")
                            except Exception as e:
                                print(f"  Error deleting {file_path.relative_to(dir_path)}: {e}")
    
    # Clean specific files and directories
    print(f"\nCleaning specific targets...")
    
    for target in specific_cleanup:
        target_path = project_root / target
        
        if target_path.exists():
            try:
                if target_path.is_file():
                    target_path.unlink()
                    deleted_files.append(str(target_path.relative_to(project_root)))
                    print(f"  Deleted file: {target}")
                elif target_path.is_dir():
                    shutil.rmtree(target_path)
                    deleted_dirs.append(str(target_path.relative_to(project_root)))
                    print(f"  Deleted directory: {target}")
            except Exception as e:
                print(f"  Error deleting {target}: {e}")
    
    # Recreate essential directory structure with .gitkeep files
    essential_dirs = [
        'results/models',
        'results/federated_learning', 
        'results/analysis',
        'results/figures',
        'results/publication_figures',
        'results/logs',
        'results/final_analysis',
        'results/private_federated_learning',
        'data/processed',
        'data/splits'
    ]
    
    print(f"\nRecreating enhanced directory structure...")
    
    for essential_dir in essential_dirs:
        dir_path = project_root / essential_dir
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep file to preserve directory in version control
        gitkeep_file = dir_path / '.gitkeep'
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"  Created: {essential_dir}/.gitkeep")
    
    # Summary report
    print("\n" + "=" * 50)
    print("ENHANCED CLEANUP SUMMARY")
    print("=" * 50)
    
    print(f"Files deleted: {len(deleted_files)}")
    print(f"Directories deleted: {len(deleted_dirs)}")
    print(f"Essential directories recreated: {len(essential_dirs)}")
    
    print("\n✓ Enhanced pipeline cleanup completed successfully")
    print("✓ Ready for fresh experimental runs with baseline integration")
    print("✓ All enhanced federated learning outputs cleared")
    
    return len(deleted_files), len(deleted_dirs)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1].lower() == "enhanced":
        print("Running enhanced federated learning cleanup...")
    
    clean_pipeline_results()