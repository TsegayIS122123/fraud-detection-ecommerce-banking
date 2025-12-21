"""
Data Transformation Module - Handles scaling, encoding, and data splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataTransformer:
    """Handles data transformation, encoding, and splitting."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.transformation_report = {}
    
    def identify_column_types(self, df):
        """Identify numerical and categorical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present
        if 'class' in numerical_cols:
            numerical_cols.remove('class')
        if 'Class' in numerical_cols:
            numerical_cols.remove('Class')
        
        return numerical_cols, categorical_cols
    
    def scale_numerical_features(self, df, numerical_cols, scaler_type='standard', target_col=None):
        """Scale numerical features using specified scaler."""
        print(f"\n{'='*60}")
        print(f"SCALING NUMERICAL FEATURES ({scaler_type.upper()} SCALER)")
        print(f"{'='*60}")
        
        df = df.copy()
        
        # Remove target column from scaling
        if target_col and target_col in numerical_cols:
            numerical_cols = [col for col in numerical_cols if col != target_col]
        
        if not numerical_cols:
            print("No numerical columns to scale.")
            return df
        
        print(f"\nScaling {len(numerical_cols)} numerical columns:")
        print("-" * 40)
        
        # Initialize scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            print(f"Unknown scaler type: {scaler_type}. Using StandardScaler.")
            scaler = StandardScaler()
        
        # Scale features
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[numerical_cols]),
            columns=[f"{col}_scaled" for col in numerical_cols],
            index=df.index
        )
        
        # Add scaled features to dataframe
        for col in numerical_cols:
            original_col = f"{col}_scaled"
            df[original_col] = df_scaled[original_col]
        
        # Store scaler for future use
        self.scalers[scaler_type] = scaler
        self.scalers['scaled_columns'] = numerical_cols
        
        # Display scaling summary
        for i, col in enumerate(numerical_cols[:10]):  # Show first 10
            print(f"  {col:25s} → {col}_scaled")
        
        if len(numerical_cols) > 10:
            print(f"  ... and {len(numerical_cols) - 10} more columns")
        
        print(f"\nScaling completed. Added {len(numerical_cols)} scaled columns.")
        
        # Store in report
        self.transformation_report['scaling'] = {
            'scaler_type': scaler_type,
            'scaled_columns': numerical_cols,
            'scaler_params': scaler.get_params()
        }
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols, encoding_type='onehot', 
                                    max_categories=10, target_col=None):
        """Encode categorical features using specified encoding method."""
        print(f"\n{'='*60}")
        print(f"ENCODING CATEGORICAL FEATURES ({encoding_type.upper()} ENCODING)")
        print(f"{'='*60}")
        
        df = df.copy()
        
        # Remove target column from encoding
        if target_col and target_col in categorical_cols:
            categorical_cols = [col for col in categorical_cols if col != target_col]
        
        if not categorical_cols:
            print("No categorical columns to encode.")
            return df
        
        print(f"\nEncoding {len(categorical_cols)} categorical columns:")
        print("-" * 40)
        
        encoded_dfs = []
        
        for col in categorical_cols:
            print(f"\nProcessing: {col}")
            print(f"  Unique values: {df[col].nunique()}")
            
            # Handle high cardinality columns
            if df[col].nunique() > max_categories:
                print(f"  ⚠️  High cardinality ({df[col].nunique()} unique values)")
                
                # Keep top N-1 categories, rest as 'Other'
                top_categories = df[col].value_counts().head(max_categories - 1).index
                df[col] = np.where(df[col].isin(top_categories), df[col], 'Other')
                print(f"  → Reduced to {df[col].nunique()} categories")
            
            if encoding_type == 'onehot':
                # One-Hot Encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]])
                
                # Create column names
                categories = encoder.categories_[0]
                encoded_cols = [f"{col}_{cat}" for cat in categories]
                
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
                encoded_dfs.append(encoded_df)
                
                print(f"  → Created {len(encoded_cols)} one-hot columns")
                
                # Store encoder
                if 'onehot_encoders' not in self.encoders:
                    self.encoders['onehot_encoders'] = {}
                self.encoders['onehot_encoders'][col] = encoder
                
            elif encoding_type == 'label':
                # Label Encoding
                encoder = LabelEncoder()
                df[f"{col}_encoded"] = encoder.fit_transform(df[col].fillna('Unknown'))
                
                print(f"  → Created label encoded column: {col}_encoded")
                
                # Store encoder
                if 'label_encoders' not in self.encoders:
                    self.encoders['label_encoders'] = {}
                self.encoders['label_encoders'][col] = encoder
                
            elif encoding_type == 'frequency':
                # Frequency Encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f"{col}_freq"] = df[col].map(freq_map)
                
                print(f"  → Created frequency encoded column: {col}_freq")
                
                # Store mapping
                if 'frequency_maps' not in self.encoders:
                    self.encoders['frequency_maps'] = {}
                self.encoders['frequency_maps'][col] = freq_map
                
            elif encoding_type == 'target':
                # Target Encoding (mean encoding)
                if target_col and target_col in df.columns:
                    target_mean = df.groupby(col)[target_col].mean().to_dict()
                    df[f"{col}_target"] = df[col].map(target_mean)
                    
                    print(f"  → Created target encoded column: {col}_target")
                    
                    # Store mapping
                    if 'target_maps' not in self.encoders:
                        self.encoders['target_maps'] = {}
                    self.encoders['target_maps'][col] = target_mean
                else:
                    print(f"  ⚠️  Cannot use target encoding - target column not found")
                    # Fallback to label encoding
                    encoder = LabelEncoder()
                    df[f"{col}_encoded"] = encoder.fit_transform(df[col].fillna('Unknown'))
                    print(f"  → Fallback: Created label encoded column: {col}_encoded")
        
        # Combine all encoded DataFrames for one-hot encoding
        if encoding_type == 'onehot' and encoded_dfs:
            encoded_df = pd.concat(encoded_dfs, axis=1)
            df = pd.concat([df, encoded_df], axis=1)
        
        # Store in report
        if 'encoding' not in self.transformation_report:
            self.transformation_report['encoding'] = {}
        
        self.transformation_report['encoding'][encoding_type] = {
            'encoded_columns': categorical_cols,
            'encoding_method': encoding_type,
            'max_categories': max_categories
        }
        
        print(f"\nEncoding completed.")
        
        return df
    
    def prepare_data_for_modeling(self, df, target_col, test_size=0.2, 
                                  scale_numerical=True, encode_categorical=True,
                                  scaler_type='standard', encoding_type='onehot'):
        """Prepare complete dataset for modeling with scaling and encoding."""
        print(f"\n{'='*60}")
        print("COMPLETE DATA PREPARATION FOR MODELING")
        print(f"{'='*60}")
        
        df = df.copy()
        
        # Identify column types
        numerical_cols, categorical_cols = self.identify_column_types(df)
        
        # Remove target column from feature lists
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        print(f"\nDataset Overview:")
        print("-" * 40)
        print(f"Total samples: {len(df):,}")
        print(f"Total features: {df.shape[1] - 1} (excluding target)")
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Target column: '{target_col}'")
        
        # Step 1: Scale numerical features
        if scale_numerical and numerical_cols:
            df = self.scale_numerical_features(df, numerical_cols, scaler_type, target_col)
        
        # Step 2: Encode categorical features
        if encode_categorical and categorical_cols:
            df = self.encode_categorical_features(df, categorical_cols, encoding_type, 
                                                 target_col=target_col)
        
        # Step 3: Prepare final feature set
        # Get all feature columns (excluding target and original categorical columns if one-hot encoded)
        feature_columns = []
        
        # Add scaled numerical columns
        if scale_numerical and numerical_cols:
            feature_columns.extend([f"{col}_scaled" for col in numerical_cols])
        else:
            feature_columns.extend(numerical_cols)
        
        # Add encoded categorical columns
        if encode_categorical and categorical_cols:
            if encoding_type == 'onehot':
                # For one-hot, we need to get the actual column names
                for col in categorical_cols:
                    encoded_cols = [c for c in df.columns if c.startswith(f"{col}_") and not c.endswith('_freq') and not c.endswith('_target')]
                    feature_columns.extend(encoded_cols)
            else:
                feature_columns.extend([f"{col}_encoded" for col in categorical_cols if f"{col}_encoded" in df.columns])
                feature_columns.extend([f"{col}_freq" for col in categorical_cols if f"{col}_freq" in df.columns])
                feature_columns.extend([f"{col}_target" for col in categorical_cols if f"{col}_target" in df.columns])
        
        # Remove any duplicates
        feature_columns = list(set(feature_columns))
        
        print(f"\n{'='*60}")
        print("FINAL FEATURE SET")
        print(f"{'='*60}")
        print(f"Total features prepared: {len(feature_columns)}")
        
        # Display first 20 features
        print(f"\nFirst 20 features:")
        for i, feat in enumerate(feature_columns[:20]):
            print(f"  {i+1:3d}. {feat}")
        
        if len(feature_columns) > 20:
            print(f"  ... and {len(feature_columns) - 20} more features")
        
        # Prepare X and y
        X = df[feature_columns]
        y = df[target_col]
        
        print(f"\nFeature matrix X shape: {X.shape}")
        print(f"Target vector y shape: {y.shape}")
        
        # Store in report
        self.transformation_report['final_features'] = {
            'feature_count': len(feature_columns),
            'feature_list': feature_columns,
            'X_shape': X.shape,
            'y_shape': y.shape
        }
        
        return X, y, feature_columns
    
    def train_test_split_data(self, X, y, test_size=0.2, stratify=True):
        """Split data into train and test sets with optional stratification."""
        print(f"\n{'='*60}")
        print("TRAIN-TEST SPLIT")
        print(f"{'='*60}")
        
        if stratify and y is not None:
            print(f"Using stratified split (preserving class distribution)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=y
            )
        else:
            print(f"Using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state
            )
        
        print(f"\nSplit Results:")
        print("-" * 40)
        print(f"Training set: {X_train.shape[0]:,} samples ({100*(1-test_size):.1f}%)")
        print(f"Testing set:  {X_test.shape[0]:,} samples ({100*test_size:.1f}%)")
        print(f"Features:     {X_train.shape[1]}")
        
        if y is not None:
            print(f"\nClass Distribution in Training Set:")
            train_class_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
            for class_val, percent in train_class_dist.items():
                print(f"  Class {class_val}: {percent:.2%}")
            
            print(f"\nClass Distribution in Testing Set:")
            test_class_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
            for class_val, percent in test_class_dist.items():
                print(f"  Class {class_val}: {percent:.2%}")
        
        # Store in report
        self.transformation_report['train_test_split'] = {
            'test_size': test_size,
            'stratify': stratify,
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'y_train_shape': y_train.shape if y is not None else None,
            'y_test_shape': y_test.shape if y is not None else None
        }
        
        return X_train, X_test, y_train, y_test
    
    def get_transformation_report(self):
        """Return complete transformation report."""
        return self.transformation_report
    
    def save_transformers(self, filepath):
        """Save fitted scalers and encoders for future use."""
        import joblib
        
        transformers = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'transformation_report': self.transformation_report
        }
        
        joblib.dump(transformers, filepath)
        print(f"Transformers saved to: {filepath}")
    
    def load_transformers(self, filepath):
        """Load previously saved transformers."""
        import joblib
        
        transformers = joblib.load(filepath)
        self.scalers = transformers.get('scalers', {})
        self.encoders = transformers.get('encoders', {})
        self.transformation_report = transformers.get('transformation_report', {})
        
        print(f"Transformers loaded from: {filepath}")
        return self