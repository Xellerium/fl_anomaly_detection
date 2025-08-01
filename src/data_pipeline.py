# data_pipeline.py
"""
Enhanced Data Pipeline for Multi-Dataset Smart Grid Federated Learning Research
This module handles complete data processing for three different smart grid datasets
with privacy-enhanced preprocessing techniques.

Datasets:
1. Mississippi State University (MSU) - Power system attack dataset
2. Pecan Street - Real-world energy consumption data
3. SGCC - State Grid Corporation of China electricity theft dataset

Privacy-Enhanced Preprocessing Techniques:
1. Adaptive Feature Binning - Reduces sensitivity while preserving patterns
2. Private Oversampling - Handles class imbalance without exposing rare samples
3. Privacy-Oriented Feature Selection - Reduces attack surface
4. Bounded Local Normalization - Protects statistical properties
5. Context-Based Noise Calibration - Optimizes privacy-utility trade-off
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import pickle
import re
import warnings
from typing import Dict, Tuple, Optional, List
from datetime import datetime

warnings.filterwarnings('ignore')

# Check for CUDA availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"[SUCCESS] CUDA is available for GPU-accelerated preprocessing")
except ImportError:
    CUDA_AVAILABLE = False

class PrivacyEnhancedDataProcessor:
    """Handles complete data processing pipeline with privacy enhancements for smart grid datasets"""
    
    def __init__(self, data_path="data/raw", random_state=42):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        self.bin_discretizers = {}
        
        # Privacy parameters
        self.privacy_config = {
            'adaptive_binning': True,
            'bin_strategy': 'quantile',  # 'quantile', 'uniform', 'kmeans'
            'n_bins': 10,
            'feature_selection_k': 50,
            'noise_epsilon': 1.0,  # Differential privacy parameter
            'bounded_norm_clip': 3.0,  # Clip norm for bounded normalization
            'private_sampling_ratio': 0.3,  # Ratio for private oversampling
        }
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'msu': {
                'file_pattern': '*.arff',
                'label_column': 'marker',
                'scenario_mapping': {
                    'natural_events': [1, 2, 3, 4, 5, 6, 13, 14],
                    'normal_operation': [41],
                    # All other scenarios are attack events
                }
            },
            'pecan': {
                'file_pattern': 'pecan_*.csv',
                'label_column': None,  # Will create synthetic anomalies
                'anomaly_threshold': 0.95,  # Percentile for anomaly detection
                'time_column': 'localminute',
                'value_column': 'use',
                'id_column': 'dataid'
            },
            'sgcc': {
                'file_pattern': 'datasetsmall.csv',
                'label_column': 'FLAG',
                'id_column': 'CONS_NO',
                'date_columns': None  # Will be detected dynamically
            }
        }
    
    def apply_adaptive_binning(self, X: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Apply adaptive feature binning to reduce data sensitivity"""
        print(f"Applying adaptive feature binning for {dataset_name}...")
        
        if dataset_name not in self.bin_discretizers:
            self.bin_discretizers[dataset_name] = {}
        
        X_binned = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if X[col].nunique() > self.privacy_config['n_bins']:
                # Create discretizer for this column
                discretizer = KBinsDiscretizer(
                    n_bins=self.privacy_config['n_bins'],
                    encode='ordinal',
                    strategy=self.privacy_config['bin_strategy']
                )
                
                # Fit and transform
                col_binned = discretizer.fit_transform(X[[col]])
                X_binned[col] = col_binned.flatten()
                
                # Store discretizer for later use
                self.bin_discretizers[dataset_name][col] = discretizer
        
        print(f"  Binned {len(numeric_cols)} numeric features")
        return X_binned
    
    def apply_private_oversampling(self, X: pd.DataFrame, y: np.ndarray, 
                                 dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply privacy-preserving oversampling for imbalanced data"""
        print(f"Applying private oversampling for {dataset_name}...")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        min_class_size = counts.min()
        max_class_size = counts.max()
        
        if max_class_size / min_class_size > 2:  # Significant imbalance
            # Apply SMOTE with privacy considerations
            sampling_strategy = {}
            for cls, cnt in zip(unique, counts):
                if cnt < max_class_size * self.privacy_config['private_sampling_ratio']:
                    # Only oversample minority classes up to a privacy-safe ratio
                    sampling_strategy[cls] = int(max_class_size * self.privacy_config['private_sampling_ratio'])
            
            if sampling_strategy:
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=min(5, min_class_size - 1)
                )
                
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"  Oversampled from {len(X)} to {len(X_resampled)} samples")
                return pd.DataFrame(X_resampled, columns=X.columns), y_resampled
        
        return X, y
    
    def apply_privacy_oriented_feature_selection(self, X: pd.DataFrame, y: np.ndarray,
                                               dataset_name: str) -> pd.DataFrame:
        """Select features while minimizing privacy risks"""
        print(f"Applying privacy-oriented feature selection for {dataset_name}...")
        
        # Remove highly correlated features (potential privacy leak)
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > 0.95
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > 0.95)]
        
        X_filtered = X.drop(columns=to_drop)
        print(f"  Removed {len(to_drop)} highly correlated features")
        
        # Select top k features based on statistical importance
        k = min(self.privacy_config['feature_selection_k'], X_filtered.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = selector.fit_transform(X_filtered, y)
        selected_features = X_filtered.columns[selector.get_support()]
        
        self.feature_selectors[dataset_name] = {
            'selector': selector,
            'features': selected_features,
            'dropped_corr': to_drop
        }
        
        print(f"  Selected {len(selected_features)} features from {X.shape[1]}")
        return pd.DataFrame(X_selected, columns=selected_features)
    
    def apply_bounded_normalization(self, X: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Apply normalization with bounded values to protect statistical properties"""
        print(f"Applying bounded normalization for {dataset_name}...")
        
        if dataset_name not in self.scalers:
            self.scalers[dataset_name] = StandardScaler()
        
        # First, apply standard scaling
        X_scaled = pd.DataFrame(
            self.scalers[dataset_name].fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Then clip values to protect against extreme outliers
        clip_value = self.privacy_config['bounded_norm_clip']
        X_bounded = X_scaled.clip(lower=-clip_value, upper=clip_value)
        
        # Count clipped values
        n_clipped = ((X_scaled < -clip_value) | (X_scaled > clip_value)).sum().sum()
        if n_clipped > 0:
            print(f"  Clipped {n_clipped} extreme values")
        
        return X_bounded
    
    def add_calibrated_noise(self, X: pd.DataFrame, feature_sensitivity: Dict[str, float],
                           dataset_name: str) -> pd.DataFrame:
        """Add calibrated noise based on feature sensitivity"""
        print(f"Adding context-based calibrated noise for {dataset_name}...")
        
        X_noisy = X.copy()
        epsilon = self.privacy_config['noise_epsilon']
        
        for col in X.columns:
            # Determine sensitivity level (default to medium)
            sensitivity = feature_sensitivity.get(col, 0.5)
            
            # Calculate noise scale based on sensitivity and epsilon
            # Higher sensitivity = more noise
            noise_scale = sensitivity / epsilon
            
            # Add Laplace noise
            noise = np.random.laplace(loc=0, scale=noise_scale, size=len(X))
            X_noisy[col] += noise
        
        return X_noisy
    
    def load_msu_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Load and preprocess MSU power system attack dataset"""
        print("\nLoading MSU Power System Attack Dataset...")
        
        combined_data = []
        file_count = 0
        
        for arff_file in self.data_path.glob("data*.arff"):
            try:
                data, meta = arff.loadarff(arff_file)
                df = pd.DataFrame(data)
                
                # Convert bytes to strings
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str)
                
                combined_data.append(df)
                file_count += 1
                
            except Exception as e:
                print(f"  Error loading {arff_file.name}: {e}")
        
        if not combined_data:
            raise ValueError("No MSU ARFF files found")
        
        # Combine all dataframes
        df_combined = pd.concat(combined_data, ignore_index=True)
        print(f"  Loaded {file_count} files with {len(df_combined)} samples")
        
        # Extract features and labels
        feature_cols = [c for c in df_combined.columns if c != 'marker']
        X = df_combined[feature_cols]
    
        # Clean feature names
        cleaned_names = []
        for name in feature_cols:
            cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
            cleaned = re.sub(r'_+', '_', cleaned).strip('_')
            cleaned_names.append(cleaned)
        X.columns = cleaned_names
        
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Extract and encode labels
        scenarios = df_combined['marker'].apply(self._extract_msu_scenario)
        y_cat = scenarios.apply(self._categorize_msu_scenario)
        
        if 'msu' not in self.label_encoders:
            self.label_encoders['msu'] = LabelEncoder()
        
        y = self.label_encoders['msu'].fit_transform(y_cat)
        
        return X, y, y_cat.tolist()
    
    def _extract_msu_scenario(self, marker_value):
        """Extract scenario number from MSU marker value"""
        try:
            marker_str = str(marker_value).replace("b'", "").replace("'", "")
            marker_int = int(float(marker_str))
            
            if marker_int >= 1000000:
                scenario_num = int(str(marker_int)[:-6])
            elif marker_int >= 1000:
                scenario_num = int(str(marker_int)[:-3])
            else:
                scenario_num = marker_int
                
            return scenario_num
        except:
            return 0
    
    def _categorize_msu_scenario(self, scenario_num):
        """Categorize MSU scenario into attack classes"""
        mapping = self.dataset_configs['msu']['scenario_mapping']
        
        if scenario_num in mapping['natural_events']:
            return 'natural_event'
        elif scenario_num in mapping['normal_operation']:
            return 'normal_operation'
        else:
            return 'attack_event'
    
    def _process_pecan_chunk(self, df: pd.DataFrame) -> Tuple[List[Dict], List[int]]:
        """Process a chunk of Pecan Street data"""
        features = []
        labels = []
        
        # Convert time column
        df['localminute'] = pd.to_datetime(df['localminute'])
        
        # Group by household
        for dataid, group in df.groupby('dataid'):
            if len(group) < 50:  # Need minimum data points
                continue
            
            # Sort by time
            group = group.sort_values('localminute')
            
            # Sample every 10th record to reduce size
            group = group.iloc[::10]
            
            # Create features for last few time points
            for i in range(min(5, len(group) - 1)):
                idx = -(i + 1)
                
                # Basic features
                feature_dict = {
                    'current_use': group['use'].iloc[idx],
                    'prev_use': group['use'].iloc[idx-1] if idx > -len(group) else 0,
                    'hour': group['localminute'].iloc[idx].hour,
                    'day_of_week': group['localminute'].iloc[idx].dayofweek,
                    'is_weekend': int(group['localminute'].iloc[idx].dayofweek >= 5)
                }
                
                # Simple statistics
                recent_data = group['use'].iloc[max(idx-10, -len(group)):idx+1]
                feature_dict['recent_mean'] = recent_data.mean()
                feature_dict['recent_std'] = recent_data.std()
                feature_dict['recent_max'] = recent_data.max()
                feature_dict['recent_min'] = recent_data.min()
                
                features.append(feature_dict)
                
                # Simple anomaly detection
                threshold = group['use'].quantile(0.95)
                is_anomaly = group['use'].iloc[idx] > threshold
                labels.append(1 if is_anomaly else 0)
        
        return features, labels
    
    def load_pecan_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Load and preprocess Pecan Street energy consumption dataset"""
        print("\nLoading Pecan Street Energy Dataset...")
        
        # Process files one at a time to avoid memory issues
        all_features = []
        all_labels = []
        
        # Only process first few files for manageable dataset size
        csv_files = sorted(list(self.data_path.glob("pecan_*.csv")))[:3]  # Use first 3 files
        
        for csv_file in csv_files:
            try:
                print(f"  Processing {csv_file.name}...")
                df = pd.read_csv(csv_file, nrows=100000)  # Limit rows per file
                
                # Process this chunk
                features, labels = self._process_pecan_chunk(df)
                all_features.extend(features)
                all_labels.extend(labels)
                
                print(f"    Processed {len(features)} samples from {csv_file.name}")
                
            except Exception as e:
                print(f"  Error processing {csv_file.name}: {e}")
        
        if not all_features:
            raise ValueError("No Pecan Street data could be processed")
        
        # Convert to DataFrame
        X = pd.DataFrame(all_features)
        if 'dataid' in X.columns:
            X = X.drop(columns=['dataid'])  # Remove ID column
        y = np.array(all_labels)
        
        # Create label categories
        y_cat = ['anomaly' if l == 1 else 'normal' for l in all_labels]
        
        if 'pecan' not in self.label_encoders:
            self.label_encoders['pecan'] = LabelEncoder()
            # Fit with actual labels to ensure proper encoding
            self.label_encoders['pecan'].fit(y_cat)
        
        # Re-encode to ensure consistency
        y = self.label_encoders['pecan'].transform(y_cat)
        
        print(f"  Created {len(X)} samples with {X.shape[1]} features")
        print(f"  Anomaly rate: {y.mean():.2%}")
        
        return X, y, y_cat
    
    def load_sgcc_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Load and preprocess SGCC electricity theft dataset"""
        print("\nLoading SGCC Electricity Theft Dataset...")
        
        # Load the dataset
        df = pd.read_csv(self.data_path / 'datasetsmall.csv')
        print(f"  Loaded {len(df)} samples")
        
        # Identify columns
        date_cols = [col for col in df.columns if '/' in str(col)]
        
        # Extract features (consumption patterns)
        X = df[date_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)  # Missing consumption treated as zero
        
        # Create additional statistical features
        X['mean_consumption'] = X.mean(axis=1)
        X['std_consumption'] = X.std(axis=1)
        X['max_consumption'] = X.max(axis=1)
        X['min_consumption'] = X.min(axis=1)
        X['consumption_range'] = X['max_consumption'] - X['min_consumption']
        
        # Count zero consumption days (potential theft indicator)
        X['zero_days'] = (X[date_cols] == 0).sum(axis=1)
        X['zero_ratio'] = X['zero_days'] / len(date_cols)
        
        # Sudden drops in consumption
        for i in range(1, len(date_cols)):
            prev_col = date_cols[i-1]
            curr_col = date_cols[i]
            X[f'drop_{i}'] = (df[prev_col] - df[curr_col]).clip(lower=0)
        
        # Extract labels
        y = df['FLAG'].values
        y_cat = ['theft' if l == 1 else 'normal' for l in y]
        
        if 'sgcc' not in self.label_encoders:
            self.label_encoders['sgcc'] = LabelEncoder()
            self.label_encoders['sgcc'].fit(['normal', 'theft'])
        
        print(f"  Created {X.shape[1]} features")
        print(f"  Theft rate: {y.mean():.2%}")
        
        return X, y, y_cat
    
    def process_dataset(self, dataset_name: str, 
                       apply_privacy: bool = True) -> Dict[str, any]:
        """Process a specific dataset with optional privacy enhancements"""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} Dataset")
        print(f"Privacy Enhancements: {'ENABLED' if apply_privacy else 'DISABLED'}")
        print(f"{'='*60}")
        
        # Load dataset
        if dataset_name == 'msu':
            X, y, y_cat = self.load_msu_data()
        elif dataset_name == 'pecan':
            X, y, y_cat = self.load_pecan_data()
        elif dataset_name == 'sgcc':
            X, y, y_cat = self.load_sgcc_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Clean data
        X = self._clean_data(X)
        
        # Apply privacy-enhanced preprocessing if enabled
        if apply_privacy:
            # 1. Adaptive Feature Binning
            if self.privacy_config['adaptive_binning']:
                X = self.apply_adaptive_binning(X, dataset_name)
            
            # 2. Privacy-Oriented Feature Selection
            X = self.apply_privacy_oriented_feature_selection(X, y, dataset_name)
            
            # 3. Private Oversampling for Rare Events
            X, y = self.apply_private_oversampling(X, y, dataset_name)
            
            # 4. Bounded Local Normalization
            X = self.apply_bounded_normalization(X, dataset_name)
            
            # 5. Context-Based Noise Calibration
            # Define feature sensitivities (can be customized per dataset)
            feature_sensitivity = {col: 0.5 for col in X.columns}  # Default medium sensitivity
            X = self.add_calibrated_noise(X, feature_sensitivity, dataset_name)
        else:
            # Standard preprocessing without privacy enhancements
            if dataset_name not in self.scalers:
                self.scalers[dataset_name] = StandardScaler()
            
        X_scaled = pd.DataFrame(
                self.scalers[dataset_name].fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        X = X_scaled
        
        # Create train/val/test splits
        splits = self.create_data_splits(X, y)
        
        # Add metadata
        splits['dataset_name'] = dataset_name
        splits['privacy_applied'] = apply_privacy
        splits['n_features'] = X.shape[1]
        splits['n_samples'] = len(X)
        splits['class_distribution'] = dict(zip(*np.unique(y, return_counts=True)))
        
        return splits
    
    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling infinities and outliers"""
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col] = X[col].fillna(median_val)
        
        # Remove zero-variance features
        zero_var = X.std() == 0
        if zero_var.any():
            X = X.loc[:, ~zero_var]
        
        return X
    
    def create_data_splits(self, X: pd.DataFrame, y: np.ndarray,
                          test_size: float = 0.2, val_size: float = 0.1) -> Dict:
        """Create stratified train/validation/test splits"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
        
    def save_processed_data(self, dataset_name: str, splits: Dict,
                          save_path: str = "data/processed"):
        """Save processed data and preprocessing objects"""
        save_path = Path(save_path)
        dataset_path = save_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Save data splits
        with open(dataset_path / 'data_splits.pkl', 'wb') as f:
            pickle.dump(splits, f)
        
        # Save preprocessing objects
        preprocessing_objects = {
            'scaler': self.scalers.get(dataset_name),
            'label_encoder': self.label_encoders.get(dataset_name),
            'feature_selector': self.feature_selectors.get(dataset_name),
            'bin_discretizers': self.bin_discretizers.get(dataset_name),
            'privacy_config': self.privacy_config
        }
        
        with open(dataset_path / 'preprocessing_objects.pkl', 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        print(f"\nProcessed data saved to {dataset_path}")
    
    def process_all_datasets(self, apply_privacy: bool = True) -> Dict[str, Dict]:
        """Process all available datasets"""
        results = {}
        
        for dataset_name in ['msu', 'pecan', 'sgcc']:
            try:
                splits = self.process_dataset(dataset_name, apply_privacy)
                self.save_processed_data(dataset_name, splits)
                results[dataset_name] = splits
                
                print(f"\n{dataset_name.upper()} Dataset Summary:")
                print(f"  Features: {splits['n_features']}")
                print(f"  Samples: {splits['n_samples']}")
                print(f"  Classes: {splits['class_distribution']}")
                
            except Exception as e:
                print(f"\nError processing {dataset_name}: {e}")
                results[dataset_name] = None
        
        return results


def main():
    """Run the enhanced data processing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Data Pipeline for Smart Grid FL')
    parser.add_argument('--dataset', choices=['msu', 'pecan', 'sgcc', 'all'], 
                       default='all', help='Dataset to process')
    parser.add_argument('--no-privacy', action='store_true', 
                       help='Disable privacy enhancements')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Differential privacy epsilon parameter')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PrivacyEnhancedDataProcessor()
    
    # Update privacy config if specified
    if args.epsilon:
        processor.privacy_config['noise_epsilon'] = args.epsilon
    
    apply_privacy = not args.no_privacy
    
    print("="*60)
    print("PRIVACY-ENHANCED SMART GRID DATA PROCESSING PIPELINE")
    print("="*60)
    print(f"Privacy Mode: {'ENABLED' if apply_privacy else 'DISABLED'}")
    if apply_privacy:
        print(f"Differential Privacy ε: {processor.privacy_config['noise_epsilon']}")
    print()
    
    # Process datasets
    if args.dataset == 'all':
        results = processor.process_all_datasets(apply_privacy)
    else:
        results = {
            args.dataset: processor.process_dataset(args.dataset, apply_privacy)
        }
        processor.save_processed_data(args.dataset, results[args.dataset])
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    for dataset_name, result in results.items():
        if result:
            print(f"\n{dataset_name.upper()}:")
            print(f"  Training samples: {len(result['X_train'])}")
            print(f"  Validation samples: {len(result['X_val'])}")
            print(f"  Test samples: {len(result['X_test'])}")


if __name__ == "__main__":
    main()