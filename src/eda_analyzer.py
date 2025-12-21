"""
Exploratory Data Analysis Module - Performs univariate and bivariate analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """Performs comprehensive EDA with visualization."""
    
    def __init__(self, target_col='class'):
        self.target_col = target_col
        self.eda_report = {}
        
    def set_plotting_style(self):
        """Set consistent plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def analyze_class_distribution(self, df):
        """Analyze and visualize class imbalance."""
        print(f"\n{'='*60}")
        print("CLASS DISTRIBUTION ANALYSIS")
        print(f"{'='*60}")
        
        if self.target_col not in df.columns:
            print(f"Target column '{self.target_col}' not found in data!")
            return
        
        class_counts = df[self.target_col].value_counts()
        class_percent = df[self.target_col].value_counts(normalize=True) * 100
        
        self.eda_report['class_counts'] = class_counts.to_dict()
        self.eda_report['class_percent'] = class_percent.to_dict()
        self.eda_report['imbalance_ratio'] = class_counts.max() / class_counts.min()
        
        print(f"Class 0 (Legitimate): {class_counts.get(0, 0):,} ({class_percent.get(0, 0):.2f}%)")
        print(f"Class 1 (Fraud):      {class_counts.get(1, 0):,} ({class_percent.get(1, 0):.2f}%)")
        print(f"Imbalance Ratio: {self.eda_report['imbalance_ratio']:.1f}:1")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        axes[0].pie(class_counts, labels=['Legitimate', 'Fraud'], 
                   autopct='%1.1f%%', colors=['lightblue', 'lightcoral'],
                   startangle=90, explode=(0.1, 0))
        axes[0].set_title('Class Distribution', fontsize=14)
        
        # Bar chart with counts
        bars = axes[1].bar(['Legitimate', 'Fraud'], class_counts.values,
                          color=['lightblue', 'lightcoral'])
        axes[1].set_title('Transaction Count by Class', fontsize=14)
        axes[1].set_ylabel('Count', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 100,
                        f'{count:,}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        return self.eda_report
    
    def univariate_analysis(self, df, numerical_cols=None, categorical_cols=None):
        """Perform univariate analysis for key variables."""
        print(f"\n{'='*60}")
        print("UNIVARIATE ANALYSIS")
        print(f"{'='*60}")
        
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analyze numerical features
        print("\n1. NUMERICAL FEATURES:")
        print("-" * 40)
        
        num_stats = pd.DataFrame(index=numerical_cols, columns=['min', '25%', 'mean', 'median', '75%', 'max', 'std', 'skew'])
        
        for col in numerical_cols:
            if col != self.target_col:
                num_stats.loc[col] = [
                    df[col].min(), df[col].quantile(0.25),
                    df[col].mean(), df[col].median(),
                    df[col].quantile(0.75), df[col].max(),
                    df[col].std(), df[col].skew()
                ]
        
        print(num_stats.round(3))
        
        # Visualize numerical features
        if len(numerical_cols) > 0:
            num_to_plot = min(6, len([c for c in numerical_cols if c != self.target_col]))
            plot_cols = [c for c in numerical_cols if c != self.target_col][:num_to_plot]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(plot_cols):
                axes[idx].hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}', fontsize=12)
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Frequency', fontsize=10)
                
                # Add vertical line for mean
                mean_val = df[col].mean()
                axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                 label=f'Mean: {mean_val:.2f}')
                axes[idx].legend(fontsize=9)
            
            # Hide unused subplots
            for idx in range(len(plot_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Numerical Feature Distributions', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()
        
        # Analyze categorical features
        print("\n2. CATEGORICAL FEATURES:")
        print("-" * 40)
        
        for col in categorical_cols:
            if col != self.target_col:
                value_counts = df[col].value_counts()
                print(f"\n{col}:")
                for val, count in value_counts.head(5).items():
                    percent = (count / len(df)) * 100
                    print(f"  {val}: {count:,} ({percent:.1f}%)")
        
        # Visualize top categorical features
        if len(categorical_cols) > 0:
            cat_to_plot = min(3, len([c for c in categorical_cols if c != self.target_col]))
            plot_cols = [c for c in categorical_cols if c != self.target_col][:cat_to_plot]
            
            fig, axes = plt.subplots(1, cat_to_plot, figsize=(15, 5))
            if cat_to_plot == 1:
                axes = [axes]
            
            for idx, col in enumerate(plot_cols):
                top_categories = df[col].value_counts().head(10)
                axes[idx].bar(top_categories.index.astype(str), top_categories.values)
                axes[idx].set_title(f'Top 10 {col} values', fontsize=12)
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Count', fontsize=10)
                axes[idx].tick_params(axis='x', rotation=45)
            
            plt.suptitle('Categorical Feature Distributions', fontsize=16, y=1.05)
            plt.tight_layout()
            plt.show()
        
        self.eda_report['numerical_stats'] = num_stats.to_dict()
        return self.eda_report
    
    def bivariate_analysis(self, df, feature_cols=None):
        """Analyze relationships between features and target."""
        print(f"\n{'='*60}")
        print("BIVARIATE ANALYSIS (Features vs Target)")
        print(f"{'='*60}")
        
        if self.target_col not in df.columns:
            print(f"Target column '{self.target_col}' not found!")
            return
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != self.target_col]
        
        # For numerical features vs target
        numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_features) > 0:
            print("\n1. NUMERICAL FEATURES CORRELATION WITH TARGET:")
            print("-" * 50)
            
            correlations = {}
            for col in numerical_features:
                if col != self.target_col:
                    corr = df[col].corr(df[self.target_col])
                    correlations[col] = corr
            
            # Sort by absolute correlation
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("Top 10 features most correlated with target:")
            for col, corr in sorted_corr[:10]:
                direction = "positive" if corr > 0 else "negative"
                print(f"  {col:20s}: {corr:8.4f} ({direction})")
            
            # Visualize top correlations
            top_n = min(8, len(sorted_corr))
            top_features = [col for col, _ in sorted_corr[:top_n]]
            top_corrs = [corr for _, corr in sorted_corr[:top_n]]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['lightcoral' if c < 0 else 'lightblue' for c in top_corrs]
            bars = ax.barh(top_features, top_corrs, color=colors)
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.set_title('Top Features Correlated with Fraud', fontsize=14)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add correlation values on bars
            for bar, corr in zip(bars, top_corrs):
                width = bar.get_width()
                label_x = width + (0.01 if width >= 0 else -0.05)
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       f'{corr:.3f}', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.show()
            
            self.eda_report['correlations'] = correlations
        
        # For categorical features vs target
        categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_features) > 0:
            print("\n2. CATEGORICAL FEATURES vs TARGET:")
            print("-" * 50)
            
            for col in categorical_features[:4]:  # Limit to first 4 for readability
                print(f"\n{col}:")
                cross_tab = pd.crosstab(df[col], df[self.target_col], 
                                       normalize='index') * 100
                cross_tab.columns = ['% Legitimate', '% Fraud']
                print(cross_tab.round(2).head())
        
        return self.eda_report
    
    def generate_eda_report(self, df, save_path=None):
        """Generate comprehensive EDA report."""
        self.set_plotting_style()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # 1. Class distribution
        self.analyze_class_distribution(df)
        
        # 2. Univariate analysis
        self.univariate_analysis(df)
        
        # 3. Bivariate analysis
        self.bivariate_analysis(df)
        
        # 4. Generate summary insights
        print(f"\n{'='*60}")
        print("EDA SUMMARY INSIGHTS")
        print(f"{'='*60}")
        
        fraud_rate = self.eda_report.get('class_percent', {}).get(1, 0)
        print(f"1. Class Imbalance: {fraud_rate:.2f}% of transactions are fraudulent")
        
        if 'correlations' in self.eda_report:
            top_corr_feature = max(self.eda_report['correlations'].items(), 
                                  key=lambda x: abs(x[1]))
            print(f"2. Most predictive feature: '{top_corr_feature[0]}' " +
                  f"(correlation: {top_corr_feature[1]:.4f})")
        
        if 'imbalance_ratio' in self.eda_report:
            ratio = self.eda_report['imbalance_ratio']
            print(f"3. Imbalance ratio: {ratio:.1f}:1 (legitimate:fraud)")
            if ratio > 10:
                print("   → Severe imbalance detected! Will require special handling.")
        
        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(self.eda_report, f, indent=4)
            print(f"\nEDA report saved to: {save_path}")
        
        return self.eda_report