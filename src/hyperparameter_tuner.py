"""
Simplified Hyperparameter Tuner - Faster tuning for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score

class FastHyperparameterTuner:
    """Performs faster hyperparameter tuning using RandomizedSearchCV."""
    
    def __init__(self, cv=3, random_state=42, n_iter=20, scoring='average_precision'):
        self.cv = cv
        self.random_state = random_state
        self.n_iter = n_iter  # Number of parameter settings sampled
        self.scoring = scoring
        self.tuning_results = {}
        
    def tune_random_forest_fast(self, X_train, y_train, n_jobs=-1):
        """Fast Random Forest tuning with limited parameters."""
        print(f"\n{'='*60}")
        print("FAST HYPERPARAMETER TUNING: RANDOM FOREST")
        print(f"{'='*60}")
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Simplified parameter grid
        param_dist = {
            'n_estimators': [50, 100],  # Reduced from [50, 100, 200]
            'max_depth': [None, 10, 20],  # Reduced from [None, 10, 20, 30]
            'min_samples_split': [2, 5],  # Reduced from [2, 5, 10]
            'min_samples_leaf': [1, 2],  # Reduced from [1, 2, 4]
            'class_weight': ['balanced']  # Only one option
        }
        
        # Create model
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=n_jobs)
        
        # Perform RandomizedSearchCV (much faster)
        print(f"Performing Randomized Search ({self.n_iter} iterations)...")
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=self.n_iter,  # Only try 20 random combinations
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        # Store results
        self.tuning_results['random_forest'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_
        }
        
        # Print results
        print(f"\n✅ Tuning Complete!")
        print(f"Best Parameters: {random_search.best_params_}")
        print(f"Best Score ({self.scoring}): {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def tune_random_forest_basic(self, X_train, y_train):
        """Even faster: Basic tuning with just 2 key parameters."""
        print(f"\n{'='*60}")
        print("BASIC HYPERPARAMETER TUNING: RANDOM FOREST")
        print(f"{'='*60}")
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Test a few key combinations manually
        param_combinations = [
            {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced'},
            {'n_estimators': 200, 'max_depth': 20, 'class_weight': 'balanced'},
            {'n_estimators': 50, 'max_depth': 10, 'class_weight': 'balanced'},
        ]
        
        best_score = -1
        best_params = {}
        best_model = None
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nTrying combination {i}/{len(param_combinations)}: {params}")
            
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                class_weight=params['class_weight'],
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Simple cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                model, X_train, y_train,
                cv=3, scoring=self.scoring, n_jobs=-1
            )
            
            mean_score = np.mean(scores)
            print(f"  Score: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = model
        
        # Train best model on full training data
        print(f"\n🏆 Best combination: {best_params}")
        print(f"Best score: {best_score:.4f}")
        
        best_model.fit(X_train, y_train)
        
        # Store results
        self.tuning_results['random_forest_basic'] = {
            'best_params': best_params,
            'best_score': best_score,
            'best_estimator': best_model
        }
        
        return best_model
    
    def tune_xgboost_fast(self, X_train, y_train):
        """Fast XGBoost tuning."""
        print(f"\n{'='*60}")
        print("FAST HYPERPARAMETER TUNING: XGBOOST")
        print(f"{'='*60}")
        
        from xgboost import XGBClassifier
        
        # Calculate scale_pos_weight for imbalance
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
        
        # Simplified parameter grid
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'scale_pos_weight': [scale_pos_weight]
        }
        
        # Create model
        xgb = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Perform RandomizedSearchCV
        print(f"Performing Randomized Search ({self.n_iter} iterations)...")
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=min(self.n_iter, 8),  # Max 8 combinations
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        # Store results
        self.tuning_results['xgboost'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_
        }
        
        print(f"\n✅ Tuning Complete!")
        print(f"Best Parameters: {random_search.best_params_}")
        print(f"Best Score ({self.scoring}): {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_