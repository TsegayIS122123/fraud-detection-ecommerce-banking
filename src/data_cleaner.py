"""
Data Cleaning Module - Handles missing values, duplicates, and data types.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Handles all data cleaning operations with proper justifications."""
    
    def __init__(self, df_name=""):
        self.df_name = df_name
        self.cleaning_report = {}
        
    def clean_fraud_data(self, df):
        """Clean e-commerce fraud dataset."""
        print(f"\n{'='*60}")
        print(f"CLEANING E-COMMERCE DATA: {self.df_name}")
        print(f"{'='*60}")
        
        original_shape = df.shape
        self.cleaning_report['original_shape'] = original_shape
        
        # 1. Check and handle missing values
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        
        self.cleaning_report['missing_before'] = missing_values[missing_values > 0].to_dict()
        
        if missing_values.sum() > 0:
            print("\n1. HANDLING MISSING VALUES:")
            print("-" * 40)
            
            for col in missing_values[missing_values > 0].index:
                print(f"   {col}: {missing_values[col]} missing ({missing_percent[col]:.2f}%)")
                
                # Strategy based on column type
                if col in ['age', 'purchase_value']:
                    # Numerical columns - impute with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"   → Imputed with median: {median_val:.2f}")
                    
                elif col in ['browser', 'source', 'sex']:
                    # Categorical columns - impute with mode
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    print(f"   → Imputed with mode: '{mode_val}'")
                    
                elif col in ['signup_time', 'purchase_time']:
                    # Timestamp columns - drop rows (critical for time-based features)
                    df = df.dropna(subset=[col])
                    print(f"   → Dropped rows with missing timestamps")
        
        # 2. Remove duplicates
        duplicates = df.duplicated().sum()
        self.cleaning_report['duplicates_found'] = duplicates
        
        if duplicates > 0:
            print(f"\n2. DUPLICATE ROWS: {duplicates} found")
            print("-" * 40)
            df = df.drop_duplicates()
            print(f"   → Removed all duplicates")
        
        # 3. Correct data types
        print(f"\n3. CORRECTING DATA TYPES:")
        print("-" * 40)
        
        # Convert timestamps
        if 'signup_time' in df.columns:
            df['signup_time'] = pd.to_datetime(df['signup_time'])
            print("   → signup_time converted to datetime")
            
        if 'purchase_time' in df.columns:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
            print("   → purchase_time converted to datetime")
        
        # Convert categorical columns to proper types
        categorical_cols = ['source', 'browser', 'sex', 'device_id']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Ensure numerical columns are correct
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].fillna(df['age'].median()).astype(int)
        
        if 'purchase_value' in df.columns:
            df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce')
            df['purchase_value'] = df['purchase_value'].fillna(df['purchase_value'].median())
        
        # Final report
        final_shape = df.shape
        self.cleaning_report['final_shape'] = final_shape
        self.cleaning_report['rows_removed'] = original_shape[0] - final_shape[0]
        
        print(f"\n{'='*60}")
        print("CLEANING SUMMARY:")
        print(f"{'='*60}")
        print(f"Original shape: {original_shape}")
        print(f"Final shape:    {final_shape}")
        print(f"Rows removed:   {self.cleaning_report['rows_removed']}")
        print(f"Columns:        {original_shape[1]} → {final_shape[1]}")
        
        return df
    
    def clean_creditcard_data(self, df):
        """Clean credit card fraud dataset."""
        print(f"\n{'='*60}")
        print(f"CLEANING CREDIT CARD DATA")
        print(f"{'='*60}")
        
        original_shape = df.shape
        self.cleaning_report['original_shape'] = original_shape
        
        # 1. Check for missing values
        missing_values = df.isnull().sum()
        self.cleaning_report['missing_before'] = missing_values[missing_values > 0].to_dict()
        
        if missing_values.sum() > 0:
            print("\n1. HANDLING MISSING VALUES:")
            print("-" * 40)
            
            for col in missing_values[missing_values > 0].index:
                print(f"   {col}: {missing_values[col]} missing")
                
                if col.startswith('V'):
                    # PCA features - impute with mean
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    print(f"   → Imputed with mean: {mean_val:.6f}")
                    
                elif col == 'Amount':
                    # Transaction amount - impute with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"   → Imputed with median: {median_val:.2f}")
        
        # 2. Remove duplicates
        duplicates = df.duplicated().sum()
        self.cleaning_report['duplicates_found'] = duplicates
        
        if duplicates > 0:
            print(f"\n2. DUPLICATE ROWS: {duplicates} found")
            print("-" * 40)
            df = df.drop_duplicates()
            print(f"   → Removed all duplicates")
        
        # 3. Correct data types
        print(f"\n3. CORRECTING DATA TYPES:")
        print("-" * 40)
        
        # Convert Time from seconds to datetime-like features
        if 'Time' in df.columns:
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            print("   → Time converted to numeric")
        
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            print("   → Amount converted to numeric")
        
        if 'Class' in df.columns:
            df['Class'] = df['Class'].astype(int)
            print("   → Class converted to integer")
        
        # Final report
        final_shape = df.shape
        self.cleaning_report['final_shape'] = final_shape
        
        print(f"\n{'='*60}")
        print("CLEANING SUMMARY:")
        print(f"{'='*60}")
        print(f"Original shape: {original_shape}")
        print(f"Final shape:    {final_shape}")
        print(f"Duplicates removed: {duplicates}")
        
        return df
    
    def get_cleaning_report(self):
        """Return the cleaning report."""
        return self.cleaning_report