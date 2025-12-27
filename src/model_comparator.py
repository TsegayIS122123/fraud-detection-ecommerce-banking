"""
Model Comparator Module - Compares models and selects the best one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ModelComparator:
    """Compares multiple models and selects the best based on various criteria."""
    
    def __init__(self):
        self.comparison_results = {}
        self.best_model = None
        
    def compare_models(self, model_results, metric_weights=None):
        """Compare multiple models and rank them."""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON AND SELECTION")
        print(f"{'='*60}")
        
        if metric_weights is None:
            # Default weights for fraud detection
            metric_weights = {
                'pr_auc': 0.35,     # Most important for imbalanced data
                'recall': 0.25,     # Important for catching fraud
                'precision': 0.20,   # Important for reducing false positives
                'f1_score': 0.15,    # Balance between precision and recall
                'roc_auc': 0.05      # Less important for imbalanced data
            }
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in model_results.items():
            row = {'model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df.set_index('model', inplace=True)
        
        # Calculate weighted score
        weighted_scores = []
        for idx, row in df.iterrows():
            score = 0
            for metric, weight in metric_weights.items():
                if metric in row:
                    score += row[metric] * weight
            weighted_scores.append(score)
        
        df['weighted_score'] = weighted_scores
        df = df.sort_values('weighted_score', ascending=False)
        
        # Store results
        self.comparison_results = df
        
        # Print comparison
        print("\n📊 MODEL COMPARISON TABLE:")
        print("-" * 70)
        print(df.round(4))
        
        # Select best model
        self.best_model = df.index[0]
        best_score = df.iloc[0]['weighted_score']
        
        print(f"\n🏆 SELECTED BEST MODEL: {self.best_model}")
        print(f"   Weighted Score: {best_score:.4f}")
        
        return df
    
    def plot_model_comparison(self):
        """Create visualization comparing model performance."""
        if self.comparison_results.empty:
            print("No comparison results to plot!")
            return
        
        df = self.comparison_results
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot of weighted scores
        bars = axes[0].bar(df.index, df['weighted_score'], color='steelblue')
        axes[0].set_title('Model Comparison - Weighted Scores', fontsize=14)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel('Weighted Score', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Radar chart for key metrics
        metrics_to_plot = ['pr_auc', 'recall', 'precision', 'f1_score']
        n_metrics = len(metrics_to_plot)
        
        # Normalize metrics for radar chart
        radar_data = {}
        for metric in metrics_to_plot:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    radar_data[metric] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    radar_data[metric] = df[metric] * 0 + 0.5
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        for i, model_name in enumerate(df.index):
            values = []
            for metric in metrics_to_plot:
                if metric in radar_data:
                    values.append(radar_data[metric].loc[model_name])
            values += values[:1]  # Close the circle
            
            axes[1].plot(angles, values, 'o-', linewidth=2, label=model_name)
            axes[1].fill(angles, values, alpha=0.25)
        
        axes[1].set_xticks(angles[:-1])
        axes[1].set_xticklabels(metrics_to_plot, fontsize=11)
        axes[1].set_title('Model Performance Radar Chart', fontsize=14)
        axes[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        axes[1].grid(True)
        
        plt.suptitle('Fraud Detection Model Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def get_best_model_details(self, model_builder):
        """Get details of the best model."""
        if self.best_model and self.best_model in model_builder.models:
            best_model_info = model_builder.models[self.best_model]
            print(f"\n🔍 DETAILS OF BEST MODEL ({self.best_model}):")
            print("-" * 50)
            
            model_type = best_model_info['type']
            print(f"Model Type: {model_type}")
            
            if 'params' in best_model_info:
                print(f"Parameters: {best_model_info['params']}")
            
            print(f"Data Used: {best_model_info['data']['dataset_type']}")
            
            # Get feature importance if available
            model = best_model_info['model']
            if hasattr(model, 'feature_importances_'):
                print(f"Has Feature Importance: Yes")
                n_features = len(model.feature_importances_)
                print(f"Number of Features: {n_features}")
            elif hasattr(model, 'coef_'):
                print(f"Has Coefficients: Yes")
                n_features = len(model.coef_[0])
                print(f"Number of Features: {n_features}")
            
            return best_model_info
        else:
            print(f"Best model '{self.best_model}' not found in model builder!")
            return None
    
    def justify_selection(self):
        """Provide justification for model selection."""
        if self.comparison_results.empty:
            print("No comparison results available!")
            return
        
        df = self.comparison_results
        
        print(f"\n📝 JUSTIFICATION FOR SELECTING '{self.best_model}':")
        print("="*60)
        
        # Compare with runner-up
        if len(df) > 1:
            runner_up = df.index[1]
            diff = df.loc[self.best_model, 'weighted_score'] - df.loc[runner_up, 'weighted_score']
            
            print(f"1. Weighted Score Advantage: {diff:.4f} over {runner_up}")
            
            # Compare key metrics
            key_metrics = ['pr_auc', 'recall', 'precision', 'f1_score']
            for metric in key_metrics:
                if metric in df.columns:
                    best_val = df.loc[self.best_model, metric]
                    runner_val = df.loc[runner_up, metric]
                    advantage = best_val - runner_val
                    
                    if advantage > 0:
                        print(f"   • {metric}: +{advantage:.4f} better")
                    elif advantage < 0:
                        print(f"   • {metric}: {advantage:.4f} worse")
                    else:
                        print(f"   • {metric}: equal")
        
        # Business considerations
        print("\n2. Business Considerations:")
        best_model_type = self.best_model
        
        if 'logistic' in best_model_type.lower():
            print("   • High interpretability (coefficients can be explained)")
            print("   • Fast training and prediction")
            print("   • Suitable for regulatory compliance needs")
        elif 'forest' in best_model_type.lower():
            print("   • Good balance of performance and interpretability")
            print("   • Feature importance available")
            print("   • Robust to overfitting")
        elif 'xgboost' in best_model_type.lower() or 'lightgbm' in best_model_type.lower():
            print("   • High predictive performance")
            print("   • Handles imbalanced data well")
            print("   • Feature importance available")
        
        print("\n3. Practical Considerations:")
        print("   • Model complexity appropriate for production deployment")
        print("   • Training time reasonable for retraining needs")
        print("   • Memory requirements acceptable for real-time scoring")