# Federated Learning for Smart Grid Anomaly Detection - Project Documentation

## 1. Project Overview

This project implements a comprehensive federated learning framework for anomaly detection in smart grid systems. It uses the Mississippi State University Power System Attack Dataset to build and evaluate distributed machine learning models that can detect abnormal events while preserving data privacy across multiple smart grid operators.

### Research Objectives

- Develop a federated learning framework for smart grid security applications
- Compare centralized and distributed learning approaches in critical infrastructure
- Implement privacy-preserving mechanisms using differential privacy
- Analyze privacy-utility trade-offs in anomaly detection tasks
- Generate publication-ready experimental results for academic research

### Key Features

- Complete data processing pipeline for smart grid anomaly detection
- Multiple state-of-the-art ML models (Random Forest, XGBoost, LightGBM, CatBoost)
- Advanced federated learning with baseline model integration
- Privacy-preserving mechanisms with configurable privacy budgets
- Comprehensive evaluation framework with publication-quality outputs

## 2. Core Scripts and Functionality

The project is structured around six core components, each handling a specific aspect of the federated learning research pipeline.

### 2.1 Data Pipeline (`src/data_pipeline.py`)

This script handles the complete data processing pipeline from raw ARFF files to ML-ready datasets.

**Key Functions:**
- `SmartGridDataProcessor` class: Manages the entire data preprocessing workflow
- `load_arff_files()`: Loads and combines multiple ARFF files into a single dataset
- `clean_feature_names()`: Standardizes feature names for ML compatibility
- `extract_scenario_info()`: Extracts scenario information from marker values
- `clean_data()`: Handles missing values, outliers, and data cleaning
- `preprocess_data()`: Applies feature scaling and preprocessing
- `create_data_splits()`: Creates stratified train/validation/test splits
- `run_complete_pipeline()`: Executes the complete data processing workflow

**Data Characteristics:**
- Processes 78K+ samples from the Mississippi State University dataset
- Handles 128 PMU (Phasor Measurement Unit) features
- Creates balanced splits preserving class distribution
- Implements standard ML preprocessing techniques

### 2.2 Baseline Models (`src/baseline_models.py`)

This script implements and evaluates multiple state-of-the-art machine learning algorithms for centralized learning performance benchmarking.

**Key Functions:**
- `BaselineEvaluator` class: Handles training and evaluation of baseline ML models
- `load_processed_data()`: Loads preprocessed data from the data pipeline
- `define_models()`: Creates optimized baseline model instances
- `train_and_evaluate_models()`: Trains models and evaluates performance
- `cross_validate_best_models()`: Performs cross-validation on top performers
- `evaluate_best_model_on_test()`: Evaluates best model on held-out test set
- `generate_performance_summary()`: Creates comprehensive performance reports
- `run_complete_evaluation()`: Executes the complete baseline evaluation pipeline

**Models Evaluated:**
- Random Forest: Tree-based ensemble method
- XGBoost: Gradient boosting framework
- LightGBM: Microsoft's gradient boosting implementation
- CatBoost: Yandex's gradient boosting implementation
- Logistic Regression: Linear baseline model

### 2.3 Enhanced Federated Learning (`src/enhanced_federated_learning.py`)

This script implements an advanced federated learning framework with automatic baseline model integration.

**Key Classes:**
- `AdvancedFederatedClient`: Enhanced client with improved local training
  - Handles local data storage and model training
  - Implements proper parameter extraction for aggregation
  - Supports model updates from global parameters

- `AdvancedFederatedServer`: Enhanced server with proper model aggregation
  - Implements FedAvg algorithm with weighted averaging
  - Supports differential privacy with configurable budgets
  - Handles early stopping and convergence monitoring

- `EnhancedFederatedExperiment`: Complete experimental framework
  - Manages federated learning experiments end-to-end
  - Automatically integrates best baseline model configuration
  - Supports various federated settings (IID, non-IID, privacy)
  - Conducts comparative analysis with baseline models

**Key Enhancements:**
- Automatic best baseline model integration
- Proper federated averaging for neural networks
- Dynamic learning rates and early stopping
- Robust differential privacy implementation
- Better client data heterogeneity handling
- Comprehensive evaluation metrics

### 2.4 Flower Federated Learning (`src/flower_federated_learning.py`)

This script provides an alternative implementation using the Flower framework, an industry-standard federated learning library.

**Key Components:**
- `SmartGridNN`: Neural network for smart grid classification using PyTorch
- `SmartGridClient`: Flower client implementation with dataset handling
- Client-side training and evaluation functions
- Server-side strategy configuration and evaluation
- Support for weighted aggregation and metrics collection

**Purpose:**
- Validates the custom implementation against an industry standard
- Provides an alternative approach to federated learning
- Demonstrates interoperability with PyTorch neural networks

### 2.5 Analysis and Evaluation (`src/analysis_and_evaluation.py`)

This script handles the processing of experimental results and generates the necessary data files for visualization in the Jupyter notebook.

**Key Functions:**
- `EnhancedComprehensiveAnalyzer` class: Manages all analysis tasks
- `load_all_results()`: Loads results from all enhanced experiments
- `extract_performance_data()`: Extracts and organizes performance metrics
- `create_performance_comparison_table()`: Generates comparative tables
- `generate_research_summary()`: Produces comprehensive research summary
- `run_complete_analysis()`: Executes the complete analysis pipeline

**Outputs:**
- Performance comparison tables (CSV)
- Processed data files for visualization
- Comprehensive research summary
- Metadata for Jupyter notebook visualization

### 2.6 Visualization Notebook (`notebook.ipynb`)

This Jupyter notebook provides interactive visualization and analysis of the experimental results.

**Key Features:**
- Dynamic loading of experimental results
- Interactive visualization of model performance
- Comparative analysis of federated vs. centralized learning
- Privacy-utility trade-off visualizations
- Publication-quality figure generation
- Detailed performance metrics exploration
- Custom visualization functions for research presentation

**Visualizations Include:**
- Model performance comparison charts
- Training convergence plots
- Confusion matrices for model evaluation
- Privacy impact analysis
- Client performance distribution
- Feature importance visualization
- Aggregated performance metrics

## 3. Utility and Configuration

### 3.1 Configuration Management (`src/utils/config.py`)

This script manages configuration settings through YAML files, providing default values and easy parameter modification.

**Key Features:**
- Centralized configuration management
- Default values for all experiments
- Easy parameter modification for different scenarios

### 3.2 Enhanced Configuration (`configs/enhanced_config.yaml`)

The enhanced configuration file provides detailed settings for all aspects of the experiments:

**Key Sections:**
- Data processing settings
- Model training parameters
- Federated learning configuration
- Privacy settings
- Neural network architecture
- Random forest settings
- Flower framework settings
- Evaluation metrics
- Output and visualization settings

## 4. Interactive Interface

### 4.1 CLI Interface (`optimized_cli.py`)

The optimized CLI provides an interactive interface for running experiments and managing the research workflow.

**Key Features:**
- Interactive menus for all experimental components
- Complete pipeline execution option
- Individual step execution
- Enhanced federated learning configuration menu
- Results summary and visualization
- Project overview and code explanation

### 4.2 Project Optimization (`optimize_project.py`)

This script streamlines the project structure by removing unnecessary complexity and organizing the codebase for clear academic presentation.

## 5. Federated Learning Implementation Details

### 5.1 Baseline Model Integration

One of the key innovations in this project is the automatic integration of the best baseline model with federated learning:

1. The baseline evaluation identifies the best-performing model and configuration
2. This optimal configuration is automatically extracted and applied to federated clients
3. Federated learning starts with proven parameters rather than default settings
4. Performance comparison shows the benefits of baseline integration

### 5.2 Federated Averaging Algorithm

The implementation uses a proper FedAvg algorithm with weighted model aggregation:

**For Neural Networks:**
```python
# Weighted averaging for neural network parameters
aggregated_coefs = []
aggregated_intercepts = []

# Average each layer's weights
for layer_idx in range(len(client_updates[0]['coefs'])):
    layer_coefs = np.zeros_like(client_updates[0]['coefs'][layer_idx])
    layer_intercepts = np.zeros_like(client_updates[0]['intercepts'][layer_idx])
    
    for update in client_updates:
        weight = update['data_size'] / total_samples
        layer_coefs += weight * update['coefs'][layer_idx]
        layer_intercepts += weight * update['intercepts'][layer_idx]
    
    # Add differential privacy noise if requested
    if apply_privacy and self.privacy_budget > self.privacy_used:
        noise_scale = 0.1 / (self.privacy_budget - self.privacy_used)
        layer_coefs += np.random.laplace(0, noise_scale, layer_coefs.shape)
        layer_intercepts += np.random.laplace(0, noise_scale, layer_intercepts.shape)
        self.privacy_used += 0.1
    
    aggregated_coefs.append(layer_coefs)
    aggregated_intercepts.append(layer_intercepts)
```

**For Tree-based Models:**
- Model selection based on weighted performance scores
- Feature importance aggregation
- Client data size weighting

### 5.3 Privacy Mechanisms

The project implements differential privacy with:

- Configurable privacy budgets (epsilon values)
- Laplace noise mechanism
- Privacy accounting
- Privacy-utility trade-off analysis

### 5.4 Data Distribution Scenarios

The framework supports multiple data distribution scenarios:

**IID (Independent and Identically Distributed):**
- Random distribution of data across clients
- Similar data distribution for all clients

**Non-IID:**
- Heterogeneous data distribution
- Dominant class assignment to specific clients
- Configurable class imbalance

## 6. Expected Results

Based on the research methodology, the project should produce:

- **Centralized Performance:** 80-85% accuracy with best baseline model
- **Federated Performance:** 5-10% accuracy reduction from centralized
- **Privacy Impact:** Variable performance across different privacy budgets
- **Baseline Integration:** Improved federated performance with optimal configuration

## 7. Project Structure

```
federated_smart_grid_detection/
├── src/
│   ├── data_pipeline.py           # Complete data processing pipeline
│   ├── baseline_models.py         # Centralized ML model comparison
│   ├── enhanced_federated_learning.py   # Enhanced distributed learning
│   ├── flower_federated_learning.py     # Flower framework implementation
│   ├── analysis_and_evaluation.py       # Results analysis and processing
│   └── utils/
│       ├── config.py              # Configuration management
│       └── __init__.py
├── data/
│   ├── raw/                       # Place ARFF files here
│   ├── processed/                 # Generated processed data
│   └── splits/                    # Data splits for federated learning
├── results/
│   ├── models/                    # Saved trained models
│   ├── federated_learning/        # Federated learning results
│   ├── analysis/                  # Analysis outputs
│   └── figures/                   # Generated figures
├── configs/
│   └── enhanced_config.yaml       # Experiment configuration
├── notebook.ipynb                 # Visualization and analysis notebook
├── optimized_cli.py               # Interactive CLI interface
├── optimize_project.py            # Project optimization script
└── README.md                      # Project documentation
```

## 8. Running Experiments

### 8.1 Prerequisites

```
Python >= 3.8
Required packages: pandas, numpy, scikit-learn, matplotlib
Optional packages: torch, flwr (for Flower implementation)
```

### 8.2 Data Setup

1. Download the Mississippi State University Power System Attack Dataset (ARFF files)
2. Place all `*.arff` files in the `data/raw/` directory
3. The dataset should contain 15 files: `data1 Sampled Scenarios.csv.arff` through `data15 Sampled Scenarios.csv.arff`

### 8.3 Running Experiments

**Option 1: Interactive CLI (Recommended)**
```bash
python optimized_cli.py
```

**Option 2: Individual Scripts**
```bash
# 1. Data Pipeline
python src/data_pipeline.py

# 2. Baseline Models  
python src/baseline_models.py

# 3. Enhanced Federated Learning
python src/enhanced_federated_learning.py

# 4. Analysis & Evaluation
python src/analysis_and_evaluation.py

# 5. Visualization (after running previous steps)
jupyter notebook notebook.ipynb
```
