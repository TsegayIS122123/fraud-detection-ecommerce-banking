"""
Feature Engineering Module - Creates time-based and behavioral features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Creates engineered features for fraud detection."""
    
    def __init__(self):
        self.feature_report = {}
        self.created_features = []
    
    def create_time_features(self, df):
        """Create time-based features from datetime columns."""
        print(f"\n{'='*60}")
        print("TIME-BASED FEATURE ENGINEERING")
        print(f"{'='*60}")
        
        df = df.copy()
        original_cols = set(df.columns)
        
        print("\n1. EXTRACTING BASIC TIME FEATURES:")
        print("-" * 40)
        
        # For e-commerce data
        if 'purchase_time' in df.columns:
            # Basic time features
            df['purchase_hour'] = df['purchase_time'].dt.hour
            df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek
            df['purchase_day'] = df['purchase_time'].dt.day
            df['purchase_month'] = df['purchase_time'].dt.month
            
            print(f"   → Created: purchase_hour, purchase_dayofweek, purchase_day, purchase_month")
            
            # Cyclical encoding for hour and day
            df['hour_sin'] = np.sin(2 * np.pi * df['purchase_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['purchase_hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['purchase_dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['purchase_dayofweek'] / 7)
            
            print(f"   → Created cyclical: hour_sin, hour_cos, day_sin, day_cos")
            
            # Time of day categories
            df['time_of_day'] = pd.cut(df['purchase_hour'], 
                                      bins=[-1, 6, 12, 18, 24],
                                      labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            
            print(f"   → Created categorical: time_of_day")
            
            # Weekend flag
            df['is_weekend'] = (df['purchase_dayofweek'] >= 5).astype(int)
            print(f"   → Created flag: is_weekend")
        
        # For credit card data
        elif 'Time' in df.columns:
            # Convert seconds to hours
            df['transaction_hour'] = (df['Time'] % (24 * 3600)) / 3600
            df['transaction_hour_int'] = df['transaction_hour'].astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
            
            print(f"   → Created: transaction_hour, hour_sin, hour_cos")
        
        # Time since signup (for e-commerce)
        if 'signup_time' in df.columns and 'purchase_time' in df.columns:
            print("\n2. CALCULATING TIME SINCE SIGNUP:")
            print("-" * 40)
            
            df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
            
            # Convert to different units
            df['time_since_signup_hours'] = df['time_since_signup'] / 3600
            df['time_since_signup_days'] = df['time_since_signup_hours'] / 24
            
            # Create risk flags
            df['new_user_flag'] = (df['time_since_signup_hours'] < 1).astype(int)  # Within 1 hour
            df['very_new_user_flag'] = (df['time_since_signup_hours'] < 0.5).astype(int)  # Within 30 mins
            
            print(f"   → Created: time_since_signup_hours, time_since_signup_days")
            print(f"   → Created flags: new_user_flag, very_new_user_flag")
            
            # Binned features
            df['signup_recency'] = pd.qcut(df['time_since_signup_hours'], 
                                          q=5, 
                                          labels=['Very Recent', 'Recent', 'Medium', 'Old', 'Very Old'])
        
        # Track created features
        new_features = [col for col in df.columns if col not in original_cols]
        self.created_features.extend(new_features)
        
        print(f"\nTotal new time features created: {len(new_features)}")
        print(f"Features: {', '.join(new_features)}")
        
        return df
    
    def create_behavioral_features(self, df):
        """Create user behavioral and transaction features."""
        print(f"\n{'='*60}")
        print("BEHAVIORAL FEATURE ENGINEERING")
        print(f"{'='*60}")
        
        df = df.copy()
        original_cols = set(df.columns)
        
        # For e-commerce data
        if 'user_id' in df.columns:
            print("\n1. USER-LEVEL FEATURES:")
            print("-" * 40)
            
            # User transaction statistics
            user_stats = df.groupby('user_id').agg({
                'purchase_time': ['count', 'min', 'max'],
                'purchase_value': ['mean', 'std', 'sum', 'min', 'max'],
                'device_id': 'nunique',
                'ip_address': 'nunique'
            }).reset_index()
            
            # Flatten column names
            user_stats.columns = [
                'user_id',
                'user_txn_count', 'user_first_txn', 'user_last_txn',
                'user_avg_value', 'user_std_value', 'user_total_value',
                'user_min_value', 'user_max_value',
                'user_device_count', 'user_ip_count'
            ]
            
            # Calculate additional features
            user_stats['user_value_range'] = user_stats['user_max_value'] - user_stats['user_min_value']
            user_stats['user_value_cv'] = user_stats['user_std_value'] / (user_stats['user_avg_value'] + 1e-6)
            
            print(f"   → Created {user_stats.shape[1]-1} user-level features")
            
            # Merge back to original data
            df = df.merge(user_stats, on='user_id', how='left')
            
            # Create transaction velocity features
            if 'user_first_txn' in df.columns and 'user_last_txn' in df.columns:
                df['user_txn_duration'] = (df['user_last_txn'] - df['user_first_txn']).dt.total_seconds() / 3600
                df['user_txn_frequency'] = df['user_txn_count'] / (df['user_txn_duration'] + 1)
                
                print(f"   → Created: user_txn_duration, user_txn_frequency")
            
            # Risk flags based on user behavior
            df['multi_device_flag'] = (df['user_device_count'] > 1).astype(int)
            df['multi_ip_flag'] = (df['user_ip_count'] > 1).astype(int)
            df['high_frequency_user'] = (df['user_txn_frequency'] > df['user_txn_frequency'].quantile(0.9)).astype(int)
            
            print(f"   → Created risk flags: multi_device_flag, multi_ip_flag, high_frequency_user")
        
        # Transaction-level features
        print("\n2. TRANSACTION-LEVEL FEATURES:")
        print("-" * 40)
        
        # Amount/value features
        if 'purchase_value' in df.columns:
            df['value_zscore'] = (df['purchase_value'] - df['purchase_value'].mean()) / df['purchase_value'].std()
            df['high_value_flag'] = (df['purchase_value'] > df['purchase_value'].quantile(0.95)).astype(int)
            df['low_value_flag'] = (df['purchase_value'] < df['purchase_value'].quantile(0.05)).astype(int)
            
            print(f"   → Created: value_zscore, high_value_flag, low_value_flag")
        
        elif 'Amount' in df.columns:
            df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
            df['high_amount_flag'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
            
            print(f"   → Created: amount_zscore, high_amount_flag")
        
        # Device-based features
        if 'device_id' in df.columns:
            device_stats = df.groupby('device_id').agg({
                'user_id': 'nunique',
                'purchase_time': 'count'
            }).reset_index()
            
            device_stats.columns = ['device_id', 'device_user_count', 'device_txn_count']
            
            df = df.merge(device_stats, on='device_id', how='left')
            
            df['shared_device_flag'] = (df['device_user_count'] > 1).astype(int)
            df['high_usage_device'] = (df['device_txn_count'] > df['device_txn_count'].quantile(0.95)).astype(int)
            
            print(f"   → Created: device_user_count, device_txn_count, shared_device_flag, high_usage_device")
        
        # Country risk features (if country exists)
        if 'country' in df.columns:
            country_stats = df.groupby('country').agg({
                'class': 'mean'} if 'class' in df.columns else {'Class': 'mean'
            }).reset_index()
            
            country_stats.columns = ['country', 'country_fraud_rate']
            df = df.merge(country_stats, on='country', how='left')
            
            # Create risk tiers
            df['country_risk_tier'] = pd.qcut(df['country_fraud_rate'], 
                                             q=4, 
                                             labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
            
            print(f"   → Created: country_fraud_rate, country_risk_tier")
        
        # Track created features
        new_features = [col for col in df.columns if col not in original_cols]
        self.created_features.extend(new_features)
        
        print(f"\nTotal new behavioral features created: {len(new_features)}")
        
        # Store feature information
        self.feature_report['total_features_created'] = len(self.created_features)
        self.feature_report['feature_list'] = self.created_features
        
        return df
    
    def create_transaction_velocity(self, df, window_hours=24):
        """Create transaction velocity features (transactions per time window)."""
        if 'user_id' not in df.columns or 'purchase_time' not in df.columns:
            print("Cannot create transaction velocity - missing user_id or purchase_time")
            return df
        
        print(f"\n3. TRANSACTION VELOCITY FEATURES ({window_hours}-hour window):")
        print("-" * 40)
        
        df = df.copy()
        df = df.sort_values(['user_id', 'purchase_time'])
        
        # Calculate rolling transaction count
        df['prev_purchase_time'] = df.groupby('user_id')['purchase_time'].shift(1)
        df['time_since_last_txn'] = (df['purchase_time'] - df['prev_purchase_time']).dt.total_seconds() / 3600
        
        # Flag for rapid successive transactions
        df['rapid_succession_flag'] = (df['time_since_last_txn'] < 1).astype(int)  # Within 1 hour
        
        # Calculate transactions in last X hours
        df['txn_in_last_24h'] = 0
        df['txn_in_last_6h'] = 0
        
        # This is simplified - in practice would use rolling window
        for idx, row in df.iterrows():
            user_mask = (df['user_id'] == row['user_id']) & \
                       (df['purchase_time'] < row['purchase_time']) & \
                       (df['purchase_time'] >= row['purchase_time'] - pd.Timedelta(hours=24))
            df.at[idx, 'txn_in_last_24h'] = user_mask.sum()
            
            user_mask_6h = (df['user_id'] == row['user_id']) & \
                          (df['purchase_time'] < row['purchase_time']) & \
                          (df['purchase_time'] >= row['purchase_time'] - pd.Timedelta(hours=6))
            df.at[idx, 'txn_in_last_6h'] = user_mask_6h.sum()
        
        # Create velocity flags
        df['high_24h_velocity'] = (df['txn_in_last_24h'] > 3).astype(int)
        df['high_6h_velocity'] = (df['txn_in_last_6h'] > 2).astype(int)
        
        print(f"   → Created: time_since_last_txn, rapid_succession_flag")
        print(f"   → Created: txn_in_last_24h, txn_in_last_6h")
        print(f"   → Created flags: high_24h_velocity, high_6h_velocity")
        
        # Clean up temporary column
        df = df.drop(columns=['prev_purchase_time'])
        
        return df
    
    def get_feature_report(self):
        """Return feature engineering report."""
        return self.feature_report