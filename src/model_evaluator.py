"""
Model Evaluator Module - Evaluates models with appropriate metrics for imbalanced data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve)

class ModelEvaluator:
    """Evaluates fraud detection models with comprehensive metrics."""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """Evaluate model with multiple metrics appropriate for fraud detection."""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print("\n📊 PERFORMANCE METRICS:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:15s}: {value:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm, model_name='Model'):
        """Plot confusion matrix with annotations."""
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Print interpretation
        tn, fp, fn, tp = cm.ravel()
        print(f"\n🔍 CONFUSION MATRIX INTERPRETATION:")
        print(f"   True Negatives (Correct Legit):  {tn:,}")
        print(f"   False Positives (Wrong Fraud):   {fp:,}")
        print(f"   False Negatives (Missed Fraud):  {fn:,}")
        print(f"   True Positives (Correct Fraud):  {tp:,}")
        
        if fp > 0:
            print(f"   False Positive Rate: {fp/(fp+tn):.2%}")
        if fn > 0:
            print(f"   False Negative Rate: {fn/(fn+tp):.2%}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name='Model'):
        """Plot precision-recall curve for imbalanced data."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        
        # Add baseline (fraud rate)
        fraud_rate = np.mean(y_true)
        plt.axhline(y=fraud_rate, color='r', linestyle='--', 
                   label=f'Baseline (Fraud Rate = {fraud_rate:.2%})')
        
        plt.xlabel('Recall (True Positive Rate)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 PR AUC: {pr_auc:.4f}")
        print(f"   Baseline (Random): {fraud_rate:.4f}")
        print(f"   Improvement: {pr_auc - fraud_rate:.4f}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name='Model'):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\n📈 ROC AUC: {roc_auc:.4f}")
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud'])
        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print("-" * 60)
        print(report)
        return report
    
    def get_evaluation_summary(self):
        """Return summary of all evaluations."""
        summary = {}
        for model_name, results in self.evaluation_results.items():
            summary[model_name] = results['metrics']
        return pd.DataFrame(summary).T