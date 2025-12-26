"""
Model Interpretability Module using SHAP

This module handles:
- SHAP value calculation for different model types
- Summary plots (global feature importance)
- Force plots (individual prediction explanations)
- Dependence plots (feature interactions)
- Waterfall plots (detailed contribution breakdown)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ChurnModelInterpreter:
    """Interpret churn prediction models using SHAP values."""
    
    def __init__(self, output_dir='visualizations/plots/shap'):
        """
        Initialize the interpreter.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save SHAP visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.explainers = {}
        self.shap_values = {}
        
        # Initialize SHAP
        shap.initjs()
    
    def create_explainer(self, model, X_train, model_type='tree'):
        """
        Create SHAP explainer for the model.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_train : pd.DataFrame or np.array
            Training data for background
        model_type : str
            'tree' for tree-based models, 'linear' for linear models
        
        Returns:
        --------
        shap.Explainer
            SHAP explainer object
        """
        print(f"\n🔍 Creating SHAP explainer ({model_type})...")
        
        if model_type == 'tree':
            # For tree-based models (Random Forest, XGBoost, etc.)
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            # For linear models (Logistic Regression, Linear SVM, etc.)
            explainer = shap.LinearExplainer(model, X_train)
        else:
            # General explainer (slower but works for any model)
            explainer = shap.KernelExplainer(model.predict_proba, 
                                            shap.sample(X_train, 100))
        
        print("✓ SHAP explainer created successfully")
        return explainer
    
    def calculate_shap_values(self, explainer, X, model_name='model'):
        """
        Calculate SHAP values for the dataset.
        
        Parameters:
        -----------
        explainer : shap.Explainer
            SHAP explainer object
        X : pd.DataFrame or np.array
            Data to explain
        model_name : str
            Name of the model
        
        Returns:
        --------
        shap.Explanation or np.array
            SHAP values
        """
        print(f"\n📊 Calculating SHAP values for {X.shape[0]} samples...")
        
        shap_values = explainer.shap_values(X)
        
        # For binary classification, some explainers return values for both classes
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (churn)
        
        self.shap_values[model_name] = shap_values
        
        print(f"✓ SHAP values calculated (shape: {shap_values.shape})")
        return shap_values
    
    def plot_summary(self, shap_values, X, model_name='Model', max_display=20, save=True):
        """
        Create SHAP summary plot (global feature importance).
        
        Parameters:
        -----------
        shap_values : np.array
            SHAP values
        X : pd.DataFrame
            Feature data
        model_name : str
            Name of the model
        max_display : int
            Maximum number of features to display
        save : bool
            Whether to save the plot
        """
        print(f"\n📊 Creating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'shap_summary_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP summary plot saved to: {filename}")
            plt.close()
        else:
            plt.show()
    
    def plot_summary_bar(self, shap_values, X, model_name='Model', max_display=20, save=True):
        """
        Create SHAP summary bar plot (mean absolute SHAP values).
        
        Parameters:
        -----------
        shap_values : np.array
            SHAP values
        X : pd.DataFrame
            Feature data
        model_name : str
            Name of the model
        max_display : int
            Maximum number of features to display
        save : bool
            Whether to save the plot
        """
        print(f"\n📊 Creating SHAP bar plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type='bar', max_display=max_display, show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'shap_bar_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP bar plot saved to: {filename}")
            plt.close()
        else:
            plt.show()
    
    def plot_waterfall(self, explainer, X, index=0, model_name='Model', save=True):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Parameters:
        -----------
        explainer : shap.Explainer
            SHAP explainer
        X : pd.DataFrame
            Feature data
        index : int
            Index of the sample to explain
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
        """
        print(f"\n📊 Creating SHAP waterfall plot for sample {index}...")
        
        # Calculate SHAP values for single sample
        shap_values = explainer.shap_values(X.iloc[index:index+1])
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) 
                        else explainer.expected_value[1],
            data=X.iloc[index].values,
            feature_names=X.columns.tolist()
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - {model_name} (Sample {index})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'shap_waterfall_{model_name.lower().replace(" ", "_")}_sample_{index}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP waterfall plot saved to: {filename}")
            plt.close()
        else:
            plt.show()
    
    def plot_force(self, explainer, X, index=0, model_name='Model', save=True):
        """
        Create SHAP force plot for a single prediction.
        
        Parameters:
        -----------
        explainer : shap.Explainer
            SHAP explainer
        X : pd.DataFrame
            Feature data
        index : int
            Index of the sample to explain
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
        """
        print(f"\n📊 Creating SHAP force plot for sample {index}...")
        
        # Calculate SHAP values for single sample
        shap_values = explainer.shap_values(X.iloc[index:index+1])
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            expected_value = explainer.expected_value
        
        # Create force plot
        force_plot = shap.force_plot(
            expected_value,
            shap_values[0],
            X.iloc[index],
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot - {model_name} (Sample {index})', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'shap_force_{model_name.lower().replace(" ", "_")}_sample_{index}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP force plot saved to: {filename}")
            plt.close()
        else:
            plt.show()
    
    def plot_dependence(self, shap_values, X, feature, interaction_feature='auto',
                       model_name='Model', save=True):
        """
        Create SHAP dependence plot showing feature interactions.
        
        Parameters:
        -----------
        shap_values : np.array
            SHAP values
        X : pd.DataFrame
            Feature data
        feature : str
            Feature to plot
        interaction_feature : str or 'auto'
            Feature to color by (auto-detected if 'auto')
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
        """
        print(f"\n📊 Creating SHAP dependence plot for '{feature}'...")
        
        plt.figure(figsize=(10, 6))
        
        if interaction_feature == 'auto':
            shap.dependence_plot(feature, shap_values, X, show=False)
        else:
            shap.dependence_plot(feature, shap_values, X, 
                               interaction_index=interaction_feature, show=False)
        
        plt.title(f'SHAP Dependence Plot - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save:
            feature_safe = feature.replace('/', '_').replace(' ', '_')
            filename = self.output_dir / f'shap_dependence_{model_name.lower().replace(" ", "_")}_{feature_safe}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP dependence plot saved to: {filename}")
            plt.close()
        else:
            plt.show()
    
    def get_top_features(self, shap_values, feature_names, top_n=10):
        """
        Get top N most important features based on mean absolute SHAP values.
        
        Parameters:
        -----------
        shap_values : np.array
            SHAP values
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame
            Top features with their importance
        """
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def explain_prediction(self, explainer, X, index, model_name='Model'):
        """
        Provide detailed explanation for a single prediction.
        
        Parameters:
        -----------
        explainer : shap.Explainer
            SHAP explainer
        X : pd.DataFrame
            Feature data
        index : int
            Index of the sample to explain
        model_name : str
            Name of the model
        
        Returns:
        --------
        pd.DataFrame
            Feature contributions for the prediction
        """
        print(f"\n" + "="*70)
        print(f" "*15 + f"EXPLAINING PREDICTION FOR SAMPLE {index}")
        print("="*70)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X.iloc[index:index+1])
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': X.columns,
            'value': X.iloc[index].values,
            'shap_value': shap_values[0]
        })
        
        explanation_df['abs_shap'] = np.abs(explanation_df['shap_value'])
        explanation_df = explanation_df.sort_values('abs_shap', ascending=False)
        
        print("\n📊 Top Feature Contributions:")
        print("-" * 70)
        print(f"{'Feature':<30} {'Value':<15} {'SHAP Value':<15} {'Impact':<10}")
        print("-" * 70)
        
        for _, row in explanation_df.head(10).iterrows():
            impact = "↑ Churn" if row['shap_value'] > 0 else "↓ Churn"
            print(f"{row['feature']:<30} {row['value']:<15.4f} {row['shap_value']:<15.4f} {impact:<10}")
        
        print("="*70)
        
        return explanation_df
    
    def generate_interpretation_report(self, model, X_train, X_test, model_name='Model',
                                      model_type='tree', sample_indices=[0, 1, 2]):
        """
        Generate complete interpretation report with all SHAP visualizations.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_train : pd.DataFrame
            Training data
        X_test : pd.DataFrame
            Test data
        model_name : str
            Name of the model
        model_type : str
            'tree' or 'linear'
        sample_indices : list
            Indices of samples to explain individually
        """
        print("\n" + "="*70)
        print(f" "*10 + f"GENERATING SHAP INTERPRETATION REPORT FOR {model_name.upper()}")
        print("="*70)
        
        # Create explainer
        explainer = self.create_explainer(model, X_train, model_type)
        
        # Calculate SHAP values
        shap_values = self.calculate_shap_values(explainer, X_test, model_name)
        
        # Generate global plots
        print("\n📊 Generating global interpretation plots...")
        self.plot_summary(shap_values, X_test, model_name)
        self.plot_summary_bar(shap_values, X_test, model_name)
        
        # Get top features
        top_features = self.get_top_features(shap_values, X_test.columns, top_n=5)
        print(f"\n🏆 Top 5 Most Important Features:")
        for idx, row in top_features.iterrows():
            print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Generate dependence plots for top features
        print("\n📊 Generating dependence plots for top features...")
        for feature in top_features['feature'].head(3):
            try:
                self.plot_dependence(shap_values, X_test, feature, model_name=model_name)
            except Exception as e:
                print(f"⚠️  Could not create dependence plot for {feature}: {e}")
        
        # Generate individual prediction explanations
        print("\n📊 Generating individual prediction explanations...")
        for idx in sample_indices[:3]:  # Limit to 3 samples
            if idx < len(X_test):
                try:
                    self.plot_waterfall(explainer, X_test, idx, model_name)
                    self.explain_prediction(explainer, X_test, idx, model_name)
                except Exception as e:
                    print(f"⚠️  Could not explain sample {idx}: {e}")
        
        print("\n✓ SHAP interpretation report generated successfully!")
        print("="*70)
        
        return explainer, shap_values


# Example usage
if __name__ == "__main__":
    from data_loader import TelecomDataGenerator
    from preprocessing import ChurnDataPreprocessor
    from feature_engineering import ChurnFeatureEngineer
    from model_training import ChurnModelTrainer
    
    # Generate and prepare data
    generator = TelecomDataGenerator(n_samples=1000)
    df = generator.generate_dataset()
    
    # Feature engineering
    feature_engineer = ChurnFeatureEngineer()
    df = feature_engineer.create_all_features(df)
    
    # Preprocessing
    preprocessor = ChurnDataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.3)
    
    # Train models
    trainer = ChurnModelTrainer()
    rf_model = trainer.train_random_forest(X_train, y_train)
    
    # Interpret model
    interpreter = ChurnModelInterpreter()
    explainer, shap_values = interpreter.generate_interpretation_report(
        rf_model, X_train, X_test, 'Random Forest', 'tree'
    )
    
    print("\n✓ Model interpretability module test completed successfully!")
