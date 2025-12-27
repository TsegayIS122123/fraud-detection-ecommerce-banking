"""
Cross Validator Module - Implements stratified k-fold cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                           roc_auc_score, average_precision_score)

class CrossValidator:
    """Performs stratified k-fold cross-validation for reliable performance estimation."""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
        
    def cross_validate(self, model, X, y, model_name='model'):
        """Perform stratified k-fold cross-validation."""
        print(f"\n{'='*60}")
        print(f"STRATIFIED {self.n_splits}-FOLD CROSS VALIDATION")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        # Initialize metrics storage
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'pr_auc': []
        }
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                             random_state=self.random_state)
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            print(f"\n📁 Fold {fold}/{self.n_splits}:")
            print("-" * 30)
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            fold_metrics = {
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba),
                'pr_auc': average_precision_score(y_val, y_pred_proba)
            }
            
            # Store metrics
            for key in metrics.keys():
                metrics[key].append(fold_metrics[key])
            
            # Print fold results
            print(f"   Precision: {fold_metrics['precision']:.4f}")
            print(f"   Recall:    {fold_metrics['recall']:.4f}")
            print(f"   F1-Score:  {fold_metrics['f1']:.4f}")
            print(f"   ROC AUC:   {fold_metrics['roc_auc']:.4f}")
            print(f"   PR AUC:    {fold_metrics['pr_auc']:.4f}")
            
            fold += 1
        
        # Calculate mean and std across folds
        results = {}
        for metric, values in metrics.items():
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        
        # Store results
        self.cv_results[model_name] = results
        
        # Print summary
        self._print_cv_summary(model_name, results)
        
        return results
    
    def _print_cv_summary(self, model_name, results):
        """Print cross-validation summary."""
        print(f"\n📊 CROSS-VALIDATION SUMMARY - {model_name}:")
        print("-" * 60)
        
        metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        for metric in metrics:
            mean_val = results[f'{metric}_mean']
            std_val = results[f'{metric}_std']
            print(f"{metric:10s}: {mean_val:.4f} ± {std_val:.4f}")
    
    def compare_cv_results(self):
        """Compare cross-validation results across models."""
        if not self.cv_results:
            print("No cross-validation results to compare!")
            return None
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.cv_results).T
        
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS COMPARISON")
        print("="*60)
        print("\nMean ± Standard Deviation across folds:")
        print("-" * 60)
        
        display(comparison_df)
        
        return comparison_df
    
    def get_best_model_by_metric(self, metric='pr_auc'):
        """Return best model based on specified metric."""
        if not self.cv_results:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, results in self.cv_results.items():
            score = results.get(f'{metric}_mean', -1)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        print(f"\n🏆 Best model by {metric.upper()}: {best_model} ({best_score:.4f})")
        return best_model, best_score