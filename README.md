# DPFed-GridGuard: An Enhanced Federated Learning Framework with Multi-Dimensional Differential Privacy for Intrusion Detection in Decentralized Smart Grids

## 🎯 Overview

**DPFed-GridGuard** is a comprehensive privacy-enhanced federated learning framework designed for intrusion detection in decentralized smart grid systems. This framework addresses the critical challenge of detecting cyber threats and operational anomalies while preserving the privacy of sensitive infrastructure data across multiple grid operators through multi-dimensional differential privacy techniques.

### 📊 Key Results
- **102.2%** utility retention on Power System Attack Dataset (F1-score: 0.609)
- **105.1%** utility retention on Pecan Street Residential Energy Dataset (F1-score: 0.845)
- **101.4%** utility retention on SGCC Electricity Theft Dataset (F1-score: 0.731)
- **35-40%** reduction in communication rounds (7 rounds vs 11-12 for baselines)
- **Convergence acceleration** while maintaining formal differential privacy guarantees (ε=1.0)
- **Superior performance** under non-IID conditions with up to 9.2% improvement over standard federated learning

## 🔬 Research Paper

This repository contains the implementation for our paper:

> **DPFed-GridGuard: An Enhanced Federated Learning Framework with Multi-Dimensional Differential Privacy for Intrusion Detection in Decentralized Smart Grids**
> 
> **Authors:** Anass BETOUIL¹, Samia EL HADDOUTI¹,²,³, Habiba CHAOUI¹
> 
> ¹Laboratory of Advanced Systems Engineering, National School of Applied Sciences, Ibn Tofail University, Kenitra, Morocco  
> ²National Higher School For Computer Science and Systems Analysis, Mohamed V University, Rabat, Morocco  
> ³National Center for Scientific and Technical Research, Rabat, Morocco

## 🚀 Features

### Core Technical Innovations

1. **Multi-Dimensional Differential Privacy**: Integration of four complementary privacy mechanisms providing formal guarantees while maintaining operational effectiveness (≥90% accuracy requirement)

2. **Context-Aware Privacy Protection**: Feature-sensitive noise calibration that adapts protection based on privacy risk vs. utility importance, enabling >100% utility retention

3. **Heterogeneity-Aware Design**: Client clustering and adaptive privacy allocation specifically designed for smart grid data heterogeneity (geographic, temporal, infrastructure variations)

4. **Intelligent Resource Management**: Dynamic privacy budget allocation and selective encryption reducing computational overhead while maintaining security

### Privacy Mechanisms
- **CA-LDP** (Context-Aware Local Differential Privacy): Adaptive noise calibration based on feature sensitivity analysis using mutual information and SHAP values
- **CADP** (Cluster-Adaptive Differential Privacy): K-means clustering of clients with adaptive privacy budget allocation based on data characteristics  
- **S-HE** (Selective Homomorphic Encryption): Encrypts only the most sensitive model dimensions (typically 30%) reducing overhead by 70%
- **UANS** (Utility-Aware Noise Scheduler): Exponentially decaying privacy budget allocation with convergence-aware adjustment

### Supported Datasets
1. **Power System Attack Dataset**: 90,670 total samples (63,469 train, 9,067 validation, 18,134 test) with 50 power system measurements across 3 security categories (Normal operation, Natural events, Attack events)
2. **Pecan Street Residential Energy Dataset**: 965 processed samples (675 train, 97 validation, 193 test) with 8 engineered features for binary anomaly detection based on consumption patterns
3. **SGCC Electricity Theft Dataset**: 30,863 customer records (21,603 train, 3,087 validation, 6,173 test) with 50 engineered features for binary classification (normal vs. theft detection)

### Models Implemented
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Neural Networks (MLP)

## 📁 Repository Structure

```
federated_smart_grid_detection/
├── src/                              # Source code
│   ├── data_pipeline.py             # Privacy-enhanced data processing pipeline
│   ├── baseline_models.py           # Centralized model benchmarks
│   ├── enhanced_federated_learning.py # DPFed-GridGuard implementation
│   ├── flower_federated_learning.py # Flower framework integration
│   ├── analysis_and_evaluation.py   # Results analysis and evaluation
│   └── utils/                       # Utility functions and configuration
├── scripts/                         # Execution scripts
│   ├── run_experiments.py          # Complete experiment runner
│   ├── analyze_experiments.py      # Experiment analysis
│   ├── generate_paper_figures.py   # Figure generation for paper
│   └── cli.py                      # Interactive CLI interface
├── configs/                        # Configuration files
│   └── enhanced_config.yaml       # Experiment settings
├── data/                          # Data directory (datasets not included)
│   ├── raw/                       # Place dataset files here
│   ├── processed/                 # Processed datasets
│   └── splits/                    # Data split configurations
├── examples/                      # Usage examples
│   ├── quick_start.py            # Minimal example
│   └── custom_experiment.py      # Custom configuration example
├── notebooks/                     # Jupyter notebooks for analysis
│   └── analysis.ipynb            # Analysis and visualization notebook
├── access (1).tex                # IEEE Access paper submission
└── requirements.txt              # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for neural network acceleration)
- 16GB RAM minimum

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/federated_smart_grid_detection.git
cd federated_smart_grid_detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare datasets**
   - **Power System Attack Dataset**: Download ARFF files and place in `data/raw/`
   - **Pecan Street Dataset**: Obtain from [Pecan Street Dataport](https://dataport.pecanstreet.org/) (requires registration)
   - **SGCC Dataset**: Contact authors for access to processed version
   - The data processing pipeline will handle format conversion and privacy-enhanced preprocessing

## 🚀 Quick Start

### Option 1: Minimal Example
```bash
python examples/quick_start.py
```

This runs a minimal example with the Power System Attack dataset using default privacy settings (ε=1.0) and demonstrates the complete DPFed-GridGuard pipeline.

### Option 2: Interactive CLI (Recommended)
```bash
python scripts/cli.py
```

The CLI provides an interactive menu for:
- Configuring datasets and privacy settings
- Running complete pipeline or individual steps
- Viewing results and generating figures

### Option 3: Custom Configuration
```bash
python examples/custom_experiment.py --mode custom
```

For privacy-utility trade-off analysis:
```bash
python examples/custom_experiment.py --mode privacy
```

### Option 4: Complete Experimental Suite
```bash
python scripts/run_experiments.py
```

This runs the complete experimental suite with various privacy budgets and generates all paper figures.

## 📊 Reproducing Paper Results

To reproduce the exact results from our paper:

1. **Prepare all three datasets** as described in the installation section
2. **Run the complete experiment suite**:
   ```bash
   python scripts/run_experiments.py
   ```
3. **Generate comprehensive analysis**:
   ```bash
   python scripts/analyze_experiments.py
   ```
4. **Generate paper figures**:
   ```bash
   python scripts/generate_paper_figures.py
   ```

**Key Results to Expect:**
- Power System Dataset: F1-score 0.609, 102.2% utility retention at ε=1.0
- Pecan Street Dataset: F1-score 0.845, 105.1% utility retention at ε=1.0  
- SGCC Dataset: F1-score 0.731, 101.4% utility retention at ε=1.0
- Convergence in 7 rounds vs 11-12 for standard federated learning baselines
- Privacy-utility trade-off: 96.2-107.0% retention across ε=0.5 to ε=5.0

Expected runtime: 2-4 hours on a modern workstation with GPU acceleration.

## 🔧 Configuration

Edit `configs/enhanced_config.yaml` to modify:
- **Privacy Settings**: epsilon values (0.1, 0.5, 1.0, 2.0, 5.0, 10.0), noise mechanisms, privacy accounting
- **Federated Learning**: 5-10 clients, maximum 20 rounds with early stopping, IID/non-IID distributions  
- **Model Selection**: Neural networks (MLP), Random Forest, XGBoost, LightGBM, CatBoost
- **Privacy Mechanisms**: Enable/disable CA-LDP, CADP, S-HE, UANS components
- **Baseline Integration**: Automatic best model detection and parameter transfer

## 📈 Results Visualization

After running experiments, visualize results using:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{,
  title={DPFed-GridGuard: An Enhanced Federated Learning Framework with Multi-Dimensional Differential Privacy for Intrusion Detection in Decentralized Smart Grids},
  author={BETOUIL Anass, EL HADDOUTI Samia, CHAOUI Habiba},
  journal={},
  year={2025},
  doi={},
  note={}
}
```

## 🙏 Acknowledgments

- The developers of the Power System Attack Dataset used in our experiments
- [Pecan Street Inc.](https://dataport.pecanstreet.org/) for providing access to residential energy consumption data
- State Grid Corporation of China for the electricity theft detection dataset
- Laboratory of Advanced Systems Engineering, Ibn Tofail University, for research support

## 📧 Contact

**Authors**
- **Anass BETOUIL**: anass.betouil@uit.ac.ma
- **Samia EL HADDOUTI**: samia_elhaddouti@um5.ac.ma
- **CHAOUI Habiba**: habiba_chaoui@uit.ac.ma



**Affiliations:**
- Laboratory of Advanced Systems Engineering, National School of Applied Sciences, Ibn Tofail University, Kenitra, Morocco
