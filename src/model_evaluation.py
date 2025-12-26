"""
Model Evaluation Module for Customer Churn Prediction

This module handles:
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curve plotting
- Precision-Recall curves
- Model comparison utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from pathlib import Path


class ChurnModelEvaluator:
    """Evaluate and compare churn prediction models."""
    
    def __init__(self, output_dir='visualizations/plots'):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save visualization plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_results = {}
        
        # Set visualization style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name='Model'):
        """
        Evaluate model performance with comprehensive metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities for positive class
        model_name : str
            Name of the model
        
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        print("\n" + "="*70)
        print(f" "*20 + f"EVALUATING {model_name.upper()}")
        print("="*70)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        # Print metrics
        print("\n📊 Performance Metrics:")
        print("-" * 70)
        print(f"{'Metric':<20} {'Score':<15} {'Interpretation':<35}")
        print("-" * 70)
        print(f"{'Accuracy':<20} {metrics['accuracy']:<15.4f} {'Overall correctness':<35}")
        print(f"{'Precision':<20} {metrics['precision']:<15.4f} {'Accuracy of positive predictions':<35}")
        print(f"{'Recall':<20} {metrics['recall']:<15.4f} {'Coverage of actual positives':<35}")
        print(f"{'F1-Score':<20} {metrics['f1_score']:<15.4f} {'Balance of precision & recall':<35}")
        
        if y_pred_proba is not None:
            print(f"{'ROC-AUC':<20} {metrics['roc_auc']:<15.4f} {'Overall discriminative ability':<35}")
            print(f"{'Avg Precision':<20} {metrics['avg_precision']:<15.4f} {'Precision-recall trade-off':<35}")
        
        print("-" * 70)
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
        
        print("="*70)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model', save=True):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            Confusion matrix figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   ax=ax)
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to: {filename}")
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name='Model', save=True):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities for positive class
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            ROC curve figure
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to: {filename}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name='Model', save=True):
        """
        Plot Precision-Recall curve.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities for positive class
        model_name : str
            Name of the model
        save : bool
            Whether to save the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            Precision-Recall curve figure
        """
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot PR curve
        ax.plot(recall, precision, color='darkgreen', lw=2,
               label=f'PR curve (AP = {avg_precision:.4f})')
        
        # Baseline (proportion of positive class)
        baseline = y_true.sum() / len(y_true)
        ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
               label=f'Baseline (Churn Rate = {baseline:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'pr_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Precision-Recall curve saved to: {filename}")
        
        return fig
    
    def plot_feature_importance(self, feature_importance_df, model_name='Model', 
                                top_n=20, save=True):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        feature_importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' columns
        model_name : str
            Name of the model
        top_n : int
            Number of top features to display
        save : bool
            Whether to save the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            Feature importance figure
        """
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        colors = sns.color_palette('viridis', len(top_features))
        ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to: {filename}")
        
        return fig
    
    def compare_models(self, save=True):
        """
        Compare multiple models side by side.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            Model comparison figure
        """
        if len(self.evaluation_results) < 2:
            print("⚠️  Need at least 2 models to compare")
            return None
        
        # Prepare data for comparison
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        comparison_data = []
        
        for model_name, metrics in self.evaluation_results.items():
            for metric_name in metrics_to_compare:
                if metric_name in metrics:
                    comparison_data.append({
                        'Model': model_name,
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Score': metrics[metric_name]
                    })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot grouped bar chart
        metrics_list = df_comparison['Metric'].unique()
        x = np.arange(len(metrics_list))
        width = 0.35
        
        models = df_comparison['Model'].unique()
        for i, model in enumerate(models):
            model_data = df_comparison[df_comparison['Model'] == model]
            scores = [model_data[model_data['Metric'] == m]['Score'].values[0] 
                     if len(model_data[model_data['Metric'] == m]) > 0 else 0
                     for m in metrics_list]
            ax.bar(x + i * width, scores, width, label=model)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics_list)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        if save:
            filename = self.output_dir / 'model_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison plot saved to: {filename}")
        
        return fig
    
    def generate_evaluation_report(self, y_true, y_pred, y_pred_proba, 
                                   feature_importance_df, model_name='Model'):
        """
        Generate complete evaluation report with all visualizations.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like
            Predicted probabilities
        feature_importance_df : pd.DataFrame
            Feature importance data
        model_name : str
            Name of the model
        """
        print("\n" + "="*70)
        print(f" "*15 + f"GENERATING EVALUATION REPORT FOR {model_name.upper()}")
        print("="*70)
        
        # Evaluate model
        metrics = self.evaluate_model(y_true, y_pred, y_pred_proba, model_name)
        
        # Generate all plots
        print("\n📊 Generating visualizations...")
        self.plot_confusion_matrix(y_true, y_pred, model_name)
        self.plot_roc_curve(y_true, y_pred_proba, model_name)
        self.plot_precision_recall_curve(y_true, y_pred_proba, model_name)
        self.plot_feature_importance(feature_importance_df, model_name)
        
        print("\n✓ Evaluation report generated successfully!")
        print("="*70)
        
        return metrics


# Example usage
if __name__ == "__main__":
    from data_loader import TelecomDataGenerator
    from preprocessing import ChurnDataPreprocessor
    from feature_engineering import ChurnFeatureEngineer
    from model_training import ChurnModelTrainer
    
    # Generate and prepare data
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
    trainer.train_logistic_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    
    # Evaluate models
    evaluator = ChurnModelEvaluator()
    
    # Evaluate Logistic Regression
    y_pred_lr = trainer.predict('logistic_regression', X_test)
    y_pred_proba_lr = trainer.predict_proba('logistic_regression', X_test)[:, 1]
    feature_importance_lr = trainer.get_feature_importance('logistic_regression', X_train.columns)
    evaluator.generate_evaluation_report(y_test, y_pred_lr, y_pred_proba_lr, 
                                        feature_importance_lr, 'Logistic Regression')
    
    # Evaluate Random Forest
    y_pred_rf = trainer.predict('random_forest', X_test)
    y_pred_proba_rf = trainer.predict_proba('random_forest', X_test)[:, 1]
    feature_importance_rf = trainer.get_feature_importance('random_forest', X_train.columns)
    evaluator.generate_evaluation_report(y_test, y_pred_rf, y_pred_proba_rf,
                                        feature_importance_rf, 'Random Forest')
    
    # Compare models
    evaluator.compare_models()
    
    print("\n✓ Model evaluation module test completed successfully!")
