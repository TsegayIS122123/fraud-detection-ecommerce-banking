# src/imbalance_handler_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ImbalanceHandler:
    """Handles class imbalance with fallback options."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.resampling_report = {}
        self.best_method = None
        
    def analyze_imbalance(self, y):
        """Analyze the severity of class imbalance."""
        print(f"\n{'='*60}")
        print("CLASS IMBALANCE ANALYSIS")
        print(f"{'='*60}")
        
        class_counts = Counter(y)
        total_samples = len(y)
        
        print(f"\nClass Distribution:")
        print("-" * 40)
        
        for class_val, count in sorted(class_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"Class {class_val}: {count:,} samples ({percentage:.4f}%)")
        
        # Calculate imbalance metrics
        majority_class = max(class_counts.values())
        minority_class = min(class_counts.values())
        imbalance_ratio = majority_class / minority_class
        
        print(f"\nImbalance Metrics:")
        print("-" * 40)
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"Majority Class Samples: {majority_class:,}")
        print(f"Minority Class Samples: {minority_class:,}")
        
        # Determine imbalance severity
        if imbalance_ratio > 100:
            severity = "Extreme"
            recommendation = "Use combination methods or specialized algorithms"
        elif imbalance_ratio > 10:
            severity = "Severe"
            recommendation = "Use resampling methods"
        elif imbalance_ratio > 3:
            severity = "Moderate"
            recommendation = "Use resampling or weighted algorithms"
        else:
            severity = "Mild"
            recommendation = "May not need resampling"
        
        print(f"\nImbalance Severity: {severity}")
        print(f"Recommendation: {recommendation}")
        
        # Visualize class distribution
        self._plot_class_distribution(y, class_counts)
        
        # Store analysis in report
        self.resampling_report['original_distribution'] = dict(class_counts)
        self.resampling_report['imbalance_ratio'] = imbalance_ratio
        self.resampling_report['severity'] = severity
        self.resampling_report['recommendation'] = recommendation
        
        return imbalance_ratio, severity
    
    def _plot_class_distribution(self, y, class_counts):
        """Visualize class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        
        bars = axes[0].bar([str(c) for c in classes], counts, 
                          color=['lightblue', 'lightcoral'])
        axes[0].set_title('Class Distribution', fontsize=14)
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Number of Samples', fontsize=12)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                        f'{count:,}', ha='center', va='bottom', fontsize=11)
        
        # Pie chart
        labels = [f'Class {c} ({count:,})' for c, count in class_counts.items()]
        axes[1].pie(counts, labels=labels, autopct='%1.3f%%', startangle=90,
                   colors=['lightblue', 'lightcoral'], explode=[0.1, 0])
        axes[1].set_title('Class Proportion', fontsize=14)
        
        plt.suptitle('Class Imbalance Analysis', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()
    
    def apply_random_oversampling(self, X_train, y_train):
        """Apply Random Over-sampling."""
        print(f"\nApplying Random Over-sampling...")
        
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_np = X_train.values
        else:
            X_np = X_train
            
        if isinstance(y_train, pd.Series):
            y_np = y_train.values
        else:
            y_np = y_train
        
        # Get class distribution
        unique_classes, class_counts = np.unique(y_np, return_counts=True)
        
        # Find majority and minority classes
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        # Separate classes
        minority_indices = np.where(y_np == minority_class)[0]
        
        # Calculate how many samples to generate
        samples_needed = class_counts.max() - class_counts.min()
        
        # Randomly select from minority class with replacement
        np.random.seed(self.random_state)
        selected_indices = np.random.choice(minority_indices, size=samples_needed, replace=True)
        
        # Create synthetic samples
        X_synthetic = X_np[selected_indices]
        y_synthetic = np.full(samples_needed, minority_class)
        
        # Combine with original
        X_resampled = np.vstack([X_np, X_synthetic])
        y_resampled = np.concatenate([y_np, y_synthetic])
        
        # Shuffle the data
        shuffle_indices = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_indices]
        y_resampled = y_resampled[shuffle_indices]
        
        # Convert back to original format
        if isinstance(X_train, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        if isinstance(y_train, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        self._report_resampling_results('RandomOverSampler', y_train, y_resampled)
        
        return X_resampled, y_resampled
    
    def apply_random_undersampling(self, X_train, y_train, minority_ratio=0.1):
        """Apply Random Under-sampling.
        
        Args:
            minority_ratio: Desired minority class ratio relative to majority (default: 0.1)
        """
        print(f"\nApplying Random Under-sampling (target minority ratio: {minority_ratio})...")
        
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_np = X_train.values
        else:
            X_np = X_train
            
        if isinstance(y_train, pd.Series):
            y_np = y_train.values
        else:
            y_np = y_train
        
        # Get class distribution
        unique_classes, class_counts = np.unique(y_np, return_counts=True)
        
        # Find majority and minority classes
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        
        # Calculate target majority count based on minority_ratio
        target_majority_count = int(minority_count / minority_ratio)
        
        # Ensure we don't exceed available samples
        target_majority_count = min(target_majority_count, majority_count)
        
        # Sample from majority class
        majority_indices = np.where(y_np == majority_class)[0]
        np.random.seed(self.random_state)
        sampled_majority_indices = np.random.choice(
            majority_indices, 
            size=target_majority_count, 
            replace=False
        )
        
        # Get all minority samples
        minority_indices = np.where(y_np == minority_class)[0]
        
        # Combine indices
        sampled_indices = np.concatenate([sampled_majority_indices, minority_indices])
        
        # Create resampled dataset
        X_resampled = X_np[sampled_indices]
        y_resampled = y_np[sampled_indices]
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[shuffle_indices]
        y_resampled = y_resampled[shuffle_indices]
        
        # Convert back to original format
        if isinstance(X_train, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        if isinstance(y_train, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        self._report_resampling_results('RandomUnderSampler', y_train, y_resampled)
        
        return X_resampled, y_resampled
    
    def _report_resampling_results(self, method_name, y_before, y_after):
        """Report resampling results."""
        before_counts = Counter(y_before)
        after_counts = Counter(y_after)
        
        print(f"\n{method_name} Results:")
        print("-" * 40)
        
        for class_val in sorted(set(list(before_counts.keys()) + list(after_counts.keys()))):
            before = before_counts.get(class_val, 0)
            after = after_counts.get(class_val, 0)
            change = after - before
            change_pct = (change / before * 100) if before > 0 else 100
            
            print(f"Class {class_val}: {before:,} → {after:,} " +
                  f"(Δ{change:+,} | {change_pct:+.1f}%)")
    
    def get_resampling_report(self):
        """Return complete resampling report."""
        return self.resampling_report