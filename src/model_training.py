"""
Model Training Module for Customer Churn Prediction

This module handles:
- Logistic Regression model training
- Random Forest model training
- Cross-validation
- Hyperparameter tuning with GridSearchCV
- Model persistence (save/load)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from pathlib import Path
import joblib
import time


class ChurnModelTrainer:
    """Train and tune machine learning models for churn prediction."""
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        
    def train_logistic_regression(self, X_train, y_train, tune=False):
        """
        Train Logistic Regression model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
        
        Returns:
        --------
        LogisticRegression
            Trained model
        """
        print("\n" + "="*70)
        print(" "*20 + "TRAINING LOGISTIC REGRESSION")
        print("="*70)
        
        if tune:
            print("\n🔧 Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
            
            # Initialize base model
            lr = LogisticRegression(random_state=self.random_state)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                lr,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            
            # Get best model
            model = grid_search.best_estimator_
            self.best_params['logistic_regression'] = grid_search.best_params_
            
            print(f"\n✓ Tuning completed in {elapsed_time:.2f} seconds")
            print(f"\n🏆 Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"   - {param}: {value}")
            print(f"\n📊 Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
            
        else:
            print("\n🚀 Training with default parameters...")
            
            # Train with default parameters
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            
            print(f"\n✓ Training completed in {elapsed_time:.2f} seconds")
        
        # Perform cross-validation
        print("\n📈 Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        self.cv_scores['logistic_regression'] = cv_scores
        
        print(f"   CV ROC-AUC Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store model
        self.models['logistic_regression'] = model
        
        print("\n" + "="*70)
        print("✓ Logistic Regression training completed")
        print("="*70)
        
        return model
    
    def train_random_forest(self, X_train, y_train, tune=False):
        """
        Train Random Forest model.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
        
        Returns:
        --------
        RandomForestClassifier
            Trained model
        """
        print("\n" + "="*70)
        print(" "*20 + "TRAINING RANDOM FOREST")
        print("="*70)
        
        if tune:
            print("\n🔧 Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced', None]
            }
            
            # Initialize base model
            rf = RandomForestClassifier(random_state=self.random_state)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                rf,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            
            # Get best model
            model = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_
            
            print(f"\n✓ Tuning completed in {elapsed_time:.2f} seconds")
            print(f"\n🏆 Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"   - {param}: {value}")
            print(f"\n📊 Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
            
        else:
            print("\n🚀 Training with optimized default parameters...")
            
            # Train with good default parameters
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            start_time = time.time()
            model.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            
            print(f"\n✓ Training completed in {elapsed_time:.2f} seconds")
        
        # Perform cross-validation
        print("\n📈 Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        self.cv_scores['random_forest'] = cv_scores
        
        print(f"   CV ROC-AUC Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store model
        self.models['random_forest'] = model
        
        print("\n" + "="*70)
        print("✓ Random Forest training completed")
        print("="*70)
        
        return model
    
    def train_all_models(self, X_train, y_train, tune_lr=False, tune_rf=False):
        """
        Train all models.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training target
        tune_lr : bool
            Whether to tune Logistic Regression
        tune_rf : bool
            Whether to tune Random Forest
        
        Returns:
        --------
        dict
            Dictionary of trained models
        """
        print("\n" + "="*70)
        print(" "*25 + "TRAINING ALL MODELS")
        print("="*70)
        
        # Train Logistic Regression
        self.train_logistic_regression(X_train, y_train, tune=tune_lr)
        
        # Train Random Forest
        self.train_random_forest(X_train, y_train, tune=tune_rf)
        
        print("\n" + "="*70)
        print(" "*20 + "ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70)
        
        # Summary
        print("\n📊 Training Summary:")
        print(f"\n{'Model':<25} {'Mean CV ROC-AUC':<20} {'Std Dev':<15}")
        print("-" * 60)
        
        for model_name, scores in self.cv_scores.items():
            print(f"{model_name:<25} {scores.mean():<20.4f} {scores.std():<15.4f}")
        
        return self.models
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance from trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model ('logistic_regression' or 'random_forest')
        feature_names : list
            List of feature names
        
        Returns:
        --------
        pd.DataFrame
            Feature importance sorted by importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train the model first.")
        
        model = self.models[model_name]
        
        if model_name == 'logistic_regression':
            # For logistic regression, use absolute coefficients
            importance = np.abs(model.coef_[0])
        elif model_name == 'random_forest':
            # For random forest, use feature importances
            importance = model.feature_importances_
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, model_name, filepath):
        """
        Save trained model to file.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        filepath : str or Path
            Output file path
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.models[model_name],
            'cv_scores': self.cv_scores.get(model_name),
            'best_params': self.best_params.get(model_name)
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model '{model_name}' saved to: {filepath}")
    
    def load_model(self, model_name, filepath):
        """
        Load trained model from file.
        
        Parameters:
        -----------
        model_name : str
            Name to assign to the loaded model
        filepath : str or Path
            Input file path
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models[model_name] = model_data['model']
        if model_data.get('cv_scores') is not None:
            self.cv_scores[model_name] = model_data['cv_scores']
        if model_data.get('best_params') is not None:
            self.best_params[model_name] = model_data['best_params']
        
        print(f"✓ Model '{model_name}' loaded from: {filepath}")
    
    def predict(self, model_name, X):
        """
        Make predictions using trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : pd.DataFrame or np.array
            Features for prediction
        
        Returns:
        --------
        np.array
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name, X):
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : pd.DataFrame or np.array
            Features for prediction
        
        Returns:
        --------
        np.array
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.models[model_name].predict_proba(X)


# Example usage
if __name__ == "__main__":
    from data_loader import TelecomDataGenerator
    from preprocessing import ChurnDataPreprocessor
    from feature_engineering import ChurnFeatureEngineer
    
    # Generate and prepare data
    print("Generating sample data...")
    generator = TelecomDataGenerator(n_samples=2000)
    df = generator.generate_dataset()
    
    # Feature engineering
    feature_engineer = ChurnFeatureEngineer()
    df = feature_engineer.create_all_features(df)
    
    # Preprocessing
    preprocessor = ChurnDataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train models
    trainer = ChurnModelTrainer()
    models = trainer.train_all_models(X_train, y_train, tune_lr=False, tune_rf=False)
    
    print("\n✓ Model training module test completed successfully!")
