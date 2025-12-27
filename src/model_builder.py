"""
Model Builder Module - Handles model training for fraud detection.
Compatible with existing Task 1 preprocessing.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ModelBuilder:
    """Builds and trains fraud detection models with class imbalance handling."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.training_report = {}
        
    def prepare_data(self, X_train, y_train, X_test, y_test, dataset_type='ecommerce'):
        """Prepare data for modeling with appropriate class weights."""
        print(f"\n{'='*60}")
        print(f"PREPARING DATA FOR {dataset_type.upper()} MODELING")
        print(f"{'='*60}")
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        n_samples = len(y_train)
        n_classes = len(class_counts)
        
        print(f"Training samples: {n_samples:,}")
        print(f"Class distribution: {dict(zip(range(n_classes), class_counts))}")
        
        # Calculate class weights (inverse of frequency)
        weights = n_samples / (n_classes * class_counts)
        class_weights = {i: w for i, w in enumerate(weights)}
        
        print(f"Class weights for balancing: {class_weights}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'class_weights': class_weights,
            'dataset_type': dataset_type
        }
    
    def build_logistic_regression(self, data_dict, max_iter=1000):
        """Build and train Logistic Regression baseline model."""
        print(f"\n{'='*60}")
        print("BUILDING BASELINE: LOGISTIC REGRESSION")
        print(f"{'='*60}")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        class_weights = data_dict['class_weights']
        
        # Create model with class weights
        model = LogisticRegression(
            class_weight=class_weights,
            max_iter=max_iter,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        print("Training Logistic Regression...")
        model.fit(X_train, y_train)
        
        # Store model
        model_id = 'logistic_regression'
        self.models[model_id] = {
            'model': model,
            'type': 'logistic_regression',
            'data': data_dict
        }
        
        print(f"✅ Logistic Regression trained successfully!")
        print(f"   Coefficients: {len(model.coef_[0])} features")
        print(f"   Intercept: {model.intercept_[0]:.4f}")
        
        return model
    
    def build_random_forest(self, data_dict, n_estimators=100, max_depth=None):
        """Build and train Random Forest ensemble model."""
        print(f"\n{'='*60}")
        print("BUILDING ENSEMBLE: RANDOM FOREST")
        print(f"{'='*60}")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        class_weights = data_dict['class_weights']
        
        # Create model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weights,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        print(f"Training Random Forest with {n_estimators} trees...")
        model.fit(X_train, y_train)
        
        # Store model
        model_id = 'random_forest'
        self.models[model_id] = {
            'model': model,
            'type': 'random_forest',
            'data': data_dict,
            'params': {'n_estimators': n_estimators, 'max_depth': max_depth}
        }
        
        print(f"✅ Random Forest trained successfully!")
        print(f"   Trees: {n_estimators}")
        print(f"   Max depth: {'None' if max_depth is None else max_depth}")
        
        return model
    
    def build_xgboost(self, data_dict, n_estimators=100, max_depth=3):
        """Build and train XGBoost ensemble model."""
        print(f"\n{'='*60}")
        print("BUILDING ENSEMBLE: XGBOOST")
        print(f"{'='*60}")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        # Calculate scale_pos_weight for imbalance
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
        
        # Create model
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        print(f"Training XGBoost with {n_estimators} estimators...")
        model.fit(X_train, y_train)
        
        # Store model
        model_id = 'xgboost'
        self.models[model_id] = {
            'model': model,
            'type': 'xgboost',
            'data': data_dict,
            'params': {'n_estimators': n_estimators, 'max_depth': max_depth}
        }
        
        print(f"✅ XGBoost trained successfully!")
        print(f"   Estimators: {n_estimators}")
        print(f"   Max depth: {max_depth}")
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")
        
        return model
    
    def build_lightgbm(self, data_dict, n_estimators=100, max_depth=-1):
        """Build and train LightGBM ensemble model."""
        print(f"\n{'='*60}")
        print("BUILDING ENSEMBLE: LIGHTGBM")
        print(f"{'='*60}")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        # Calculate class weights for LightGBM
        class_counts = np.bincount(y_train)
        if len(class_counts) > 1:
            is_unbalance = True  # Let LightGBM handle imbalance
        else:
            is_unbalance = False
        
        # Create model
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            is_unbalance=is_unbalance,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1  # Suppress output
        )
        
        print(f"Training LightGBM with {n_estimators} estimators...")
        model.fit(X_train, y_train)
        
        # Store model
        model_id = 'lightgbm'
        self.models[model_id] = {
            'model': model,
            'type': 'lightgbm',
            'data': data_dict,
            'params': {'n_estimators': n_estimators, 'max_depth': max_depth}
        }
        
        print(f"✅ LightGBM trained successfully!")
        print(f"   Estimators: {n_estimators}")
        print(f"   Max depth: {'None' if max_depth == -1 else max_depth}")
        
        return model
    
    def get_model(self, model_id):
        """Retrieve trained model by ID."""
        return self.models.get(model_id)
    
    def get_all_models(self):
        """Return all trained models."""
        return self.models
    
    def save_model(self, model_id, filepath):
        """Save trained model to disk."""
        import joblib
        
        if model_id in self.models:
            joblib.dump(self.models[model_id]['model'], filepath)
            print(f"✅ Model saved to: {filepath}")
        else:
            print(f"❌ Model '{model_id}' not found!")