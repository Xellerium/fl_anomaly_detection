# data_pipeline.py
"""
Streamlined Data Pipeline for Smart Grid Federated Learning Research
This module handles complete data processing from raw ARFF files to ML-ready datasets

Key Functions:
1. Load and combine multiple ARFF files 
2. Clean feature names for ML compatibility
3. Extract scenario information from markers
4. Create train/validation/test splits
5. Apply feature scaling and preprocessing

For PhD Supervision Discussion:
- Handles 78K+ samples from Mississippi State University dataset
- Processes 128 PMU (Phasor Measurement Unit) features  
- Creates balanced splits preserving class distribution
- Implements standard ML preprocessing pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

class SmartGridDataProcessor:
    """Handles complete data processing pipeline for smart grid anomaly detection"""
    
    def __init__(self, data_path="data/raw", random_state=42):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Scenario categorization based on research paper
        self.scenario_mapping = {
            'natural_events': [1, 2, 3, 4, 5, 6, 13, 14],  # Faults and maintenance
            'normal_operation': [41],                        # Normal load changes
            # All other scenarios are attack events (7-12, 15-40)
        }
    
    def load_arff_files(self):
        """Load and combine all ARFF files into single dataset"""
        print("Loading ARFF files from smart grid dataset...")
        
        combined_data = []
        file_count = 0
        
        for arff_file in self.data_path.glob("*.arff"):
            try:
                # Load ARFF file using scipy
                data, meta = arff.loadarff(arff_file)
                df = pd.DataFrame(data)
                
                # Convert bytes to strings for categorical columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str)
                
                combined_data.append(df)
                file_count += 1
                
            except Exception as e:
                print(f"Error loading {arff_file.name}: {e}")
        
        # Combine all dataframes
        self.raw_data = pd.concat(combined_data, ignore_index=True)
        print(f"Successfully loaded {file_count} files with {len(self.raw_data)} total samples")
        
        return self.raw_data
    
    def clean_feature_names(self, feature_names):
        """Clean feature names to remove special characters incompatible with ML libraries"""
        cleaned_names = []
        for name in feature_names:
            # Remove special characters and replace with underscores
            cleaned_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
            cleaned_name = re.sub(r'_+', '_', cleaned_name).strip('_')
            cleaned_names.append(cleaned_name)
        return cleaned_names
    
    def extract_scenario_info(self, marker_value):
        """Extract scenario number from marker value for classification"""
        try:
            # Marker format: scenario_number + fault_location + load_condition
            marker_str = str(int(float(marker_value)))
            if len(marker_str) >= 6:
                scenario_num = int(marker_str[:-6])
            elif len(marker_str) >= 3:
                scenario_num = int(marker_str[:-3]) if len(marker_str) > 3 else int(marker_str)
            else:
                scenario_num = int(marker_str)
            return scenario_num
        except:
            return 0
    
    def categorize_scenario(self, scenario_num):
        """Categorize scenario into three classes for supervised learning"""
        if scenario_num in self.scenario_mapping['natural_events']:
            return 'natural_event'
        elif scenario_num in self.scenario_mapping['normal_operation']:
            return 'normal_operation'
        else:
            return 'attack_event'  # All other scenarios are attacks
    
    def clean_data(self, X):
        """Clean data by handling infinite values and outliers"""
        print("Cleaning data...")
        
        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Check for NaN values
        nan_cols = X.isnull().sum()
        if nan_cols.sum() > 0:
            print(f"Found {nan_cols.sum()} NaN values")
            
            # For numeric columns, replace NaN with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isnull().sum() > 0:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                    print(f"  Filled {nan_cols[col]} NaN values in {col} with median {median_val:.3f}")
        
        # Handle extreme outliers (values beyond 3 standard deviations)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        for col in numeric_cols:
            mean_val = X[col].mean()
            std_val = X[col].std()
            
            # Define outlier bounds
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            # Count outliers
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_count += outliers
                # Cap outliers to bounds
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outlier_count > 0:
            print(f"Capped {outlier_count} extreme outliers")
        
        return X
    
    def preprocess_data(self):
        """Complete preprocessing pipeline from raw data to ML-ready format"""
        print("Starting data preprocessing pipeline…")
        
        # 1) Select feature columns (everything except the 'marker')
        feature_cols = [c for c in self.raw_data.columns if c != 'marker']
        X = self.raw_data[feature_cols].copy()
        
        # 2) Clean up feature names
        cleaned_feature_names = self.clean_feature_names(feature_cols)
        X.columns = cleaned_feature_names
        
        # 3) Force everything to numeric—anything unparseable becomes NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # 4) First pass at infinities & outliers + median‐fill NaNs
        X = self.clean_data(X)
        
        # 5) Drop any feature that has zero variance
        zero_var = X.std() == 0
        zero_var_cols = zero_var[zero_var].index.tolist()
        if zero_var_cols:
            print(f"Dropping {len(zero_var_cols)} zero-variance features: {zero_var_cols[:5]}…")
            X.drop(columns=zero_var_cols, inplace=True)
        
        # 6) One last sweep to guarantee no inf/nan
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isnull().any():
                median = X[col].median()
                X[col].fillna(median, inplace=True)
                print(f"  [final clean] filled NaNs in {col} with median={median:.3f}")
        
        # 7) Extract + encode targets
        scenarios = self.raw_data['marker'].apply(self.extract_scenario_info)
        y_cat = scenarios.apply(self.categorize_scenario)
        y_enc = self.label_encoder.fit_transform(y_cat)
        
        # 8) Scale
        print("Scaling features…")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        print("Preprocessing completed:")
        print(f"  Features: {X_scaled.shape[1]}")
        print(f"  Samples:  {len(X_scaled)}")
        print(f"  Classes:  {list(self.label_encoder.classes_)}")
        
        return X_scaled, y_enc, y_cat

    
    def create_data_splits(self, X, y, test_size=0.2, val_size=0.1):
        """Create stratified train/validation/test splits for ML experiments"""
        print("Creating data splits...")
        
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
        
        splits = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
        
        print(f"Data splits created:")
        print(f"  Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return splits
    
    def save_processed_data(self, splits, save_path="data/processed"):
        """Save processed data and preprocessing objects for reuse"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save data splits
        with open(save_path / 'data_splits.pkl', 'wb') as f:
            pickle.dump(splits, f)
        
        # Save preprocessing objects
        with open(save_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(save_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Processed data saved to {save_path}")
    
    def run_complete_pipeline(self):
        """Execute complete data processing pipeline"""
        print("="*60)
        print("SMART GRID DATA PROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load raw data
        self.load_arff_files()
        
        # Step 2: Preprocess data
        X, y_encoded, y_categorical = self.preprocess_data()
        
        # Step 3: Create splits
        splits = self.create_data_splits(X, y_encoded)
        
        # Step 4: Save processed data
        self.save_processed_data(splits)
        
        print("="*60)
        print("DATA PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return splits

# Main execution function
def main():
    """Run the complete data processing pipeline"""
    processor = SmartGridDataProcessor()
    splits = processor.run_complete_pipeline()
    
    # Display summary statistics
    print("\nDataset Summary:")
    print(f"Total samples: {len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']):,}")
    print(f"Features: {splits['X_train'].shape[1]}")
    print(f"Classes: {len(np.unique(splits['y_train']))}")

if __name__ == "__main__":
    main()