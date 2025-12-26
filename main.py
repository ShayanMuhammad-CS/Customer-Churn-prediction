"""
Main Execution Script for Customer Churn Prediction Project

This script provides a command-line interface to run different parts of the project:
- Generate dataset
- Train models
- Evaluate models
- Launch dashboard
- Run complete pipeline
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import TelecomDataGenerator, DataLoader, save_dataset
from preprocessing import ChurnDataPreprocessor
from feature_engineering import ChurnFeatureEngineer
from model_training import ChurnModelTrainer
from model_evaluation import ChurnModelEvaluator
from interpretability import ChurnModelInterpreter


def generate_data(n_samples=7043):
    """Generate synthetic telecom dataset."""
    print("\n" + "="*70)
    print(" "*20 + "GENERATING DATASET")
    print("="*70)
    
    generator = TelecomDataGenerator(n_samples=n_samples)
    df = generator.generate_dataset()
    
    # Save dataset
    output_path = Path('data/raw/telecom_churn.csv')
    save_dataset(df, output_path)
    
    print("\n✓ Dataset generation completed!")
    return df


def train_models(tune_lr=False, tune_rf=False):
    """Train machine learning models."""
    print("\n" + "="*70)
    print(" "*20 + "TRAINING MODELS")
    print("="*70)
    
    # Load data
    data_path = Path('data/raw/telecom_churn.csv')
    if not data_path.exists():
        print("⚠️  Dataset not found. Generating...")
        df = generate_data()
    else:
        loader = DataLoader(data_path)
        df = loader.load_data()
    
    # Feature engineering
    print("\n📊 Performing feature engineering...")
    feature_engineer = ChurnFeatureEngineer()
    df_engineered = feature_engineer.create_all_features(df)
    
    # Save engineered data
    engineered_path = Path('data/processed/telecom_churn_engineered.csv')
    save_dataset(df_engineered, engineered_path)
    
    # Preprocessing
    print("\n🔧 Preprocessing data...")
    preprocessor = ChurnDataPreprocessor()
    X, y = preprocessor.fit_transform(df_engineered)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Save preprocessor
    preprocessor_path = Path('models/preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    
    # Train models
    print("\n🤖 Training models...")
    trainer = ChurnModelTrainer()
    trainer.train_all_models(X_train, y_train, tune_lr=tune_lr, tune_rf=tune_rf)
    
    # Save models
    models_dir = Path('models/saved_models')
    trainer.save_model('logistic_regression', models_dir / 'logistic_regression.pkl')
    trainer.save_model('random_forest', models_dir / 'random_forest.pkl')
    
    print("\n✓ Model training completed!")
    
    return trainer, preprocessor, feature_engineer, X_train, X_test, y_train, y_test


def evaluate_models():
    """Evaluate trained models."""
    print("\n" + "="*70)
    print(" "*20 + "EVALUATING MODELS")
    print("="*70)
    
    # Load data
    data_path = Path('data/processed/telecom_churn_engineered.csv')
    if not data_path.exists():
        print("⚠️  Engineered dataset not found. Training models first...")
        trainer, preprocessor, feature_engineer, X_train, X_test, y_train, y_test = train_models()
    else:
        # Load preprocessor
        preprocessor = ChurnDataPreprocessor()
        preprocessor.load(Path('models/preprocessor.pkl'))
        
        # Load data
        loader = DataLoader(data_path)
        df = loader.load_data()
        X, y = preprocessor.prepare_features_target(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Load models
        trainer = ChurnModelTrainer()
        trainer.load_model('logistic_regression', Path('models/saved_models/logistic_regression.pkl'))
        trainer.load_model('random_forest', Path('models/saved_models/random_forest.pkl'))
    
    # Evaluate models
    evaluator = ChurnModelEvaluator()
    
    # Evaluate Logistic Regression
    print("\n📊 Evaluating Logistic Regression...")
    y_pred_lr = trainer.predict('logistic_regression', X_test)
    y_pred_proba_lr = trainer.predict_proba('logistic_regression', X_test)[:, 1]
    feature_importance_lr = trainer.get_feature_importance('logistic_regression', X_train.columns)
    evaluator.generate_evaluation_report(y_test, y_pred_lr, y_pred_proba_lr,
                                        feature_importance_lr, 'Logistic Regression')
    
    # Evaluate Random Forest
    print("\n📊 Evaluating Random Forest...")
    y_pred_rf = trainer.predict('random_forest', X_test)
    y_pred_proba_rf = trainer.predict_proba('random_forest', X_test)[:, 1]
    feature_importance_rf = trainer.get_feature_importance('random_forest', X_train.columns)
    evaluator.generate_evaluation_report(y_test, y_pred_rf, y_pred_proba_rf,
                                        feature_importance_rf, 'Random Forest')
    
    # Compare models
    print("\n📊 Comparing models...")
    evaluator.compare_models()
    
    print("\n✓ Model evaluation completed!")
    print(f"\n📁 Visualizations saved to: visualizations/plots/")


def run_shap_analysis():
    """Run SHAP interpretability analysis."""
    print("\n" + "="*70)
    print(" "*20 + "SHAP INTERPRETABILITY ANALYSIS")
    print("="*70)
    
    # Load data and models
    data_path = Path('data/processed/telecom_churn_engineered.csv')
    
    if not data_path.exists():
        print("⚠️  Models not found. Training first...")
        trainer, preprocessor, feature_engineer, X_train, X_test, y_train, y_test = train_models()
    else:
        # Load preprocessor
        preprocessor = ChurnDataPreprocessor()
        preprocessor.load(Path('models/preprocessor.pkl'))
        
        # Load data
        loader = DataLoader(data_path)
        df = loader.load_data()
        X, y = preprocessor.prepare_features_target(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Load models
        trainer = ChurnModelTrainer()
        trainer.load_model('logistic_regression', Path('models/saved_models/logistic_regression.pkl'))
        trainer.load_model('random_forest', Path('models/saved_models/random_forest.pkl'))
    
    # SHAP analysis
    interpreter = ChurnModelInterpreter()
    
    # Random Forest SHAP
    print("\n🔍 Analyzing Random Forest with SHAP...")
    rf_model = trainer.models['random_forest']
    interpreter.generate_interpretation_report(
        rf_model, X_train, X_test.head(100), 'Random Forest', 'tree'
    )
    
    # Logistic Regression SHAP
    print("\n🔍 Analyzing Logistic Regression with SHAP...")
    lr_model = trainer.models['logistic_regression']
    interpreter.generate_interpretation_report(
        lr_model, X_train, X_test.head(100), 'Logistic Regression', 'linear'
    )
    
    print("\n✓ SHAP analysis completed!")
    print(f"\n📁 SHAP visualizations saved to: visualizations/plots/shap/")


def launch_dashboard():
    """Launch Streamlit dashboard."""
    print("\n" + "="*70)
    print(" "*20 + "LAUNCHING DASHBOARD")
    print("="*70)
    
    import subprocess
    
    dashboard_path = Path('dashboard/streamlit_app.py')
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return
    
    print("\n🚀 Starting Streamlit dashboard...")
    print("📊 Dashboard will open in your browser")
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    subprocess.run(['streamlit', 'run', str(dashboard_path)])


def run_all():
    """Run complete pipeline."""
    print("\n" + "="*70)
    print(" "*15 + "RUNNING COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Generate data
    df = generate_data()
    
    # Step 2: Train models
    trainer, preprocessor, feature_engineer, X_train, X_test, y_train, y_test = train_models()
    
    # Step 3: Evaluate models
    evaluate_models()
    
    # Step 4: SHAP analysis
    run_shap_analysis()
    
    print("\n" + "="*70)
    print(" "*15 + "PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Next steps:")
    print("  1. Review visualizations in 'visualizations/plots/'")
    print("  2. Explore notebooks in 'notebooks/'")
    print("  3. Launch dashboard: python main.py --mode dashboard")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Customer Churn Prediction - Main Execution Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode generate_data
  python main.py --mode train
  python main.py --mode train --tune-rf
  python main.py --mode evaluate
  python main.py --mode shap
  python main.py --mode dashboard
  python main.py --mode all
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate_data', 'train', 'evaluate', 'shap', 'dashboard', 'all'],
        default='all',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=7043,
        help='Number of samples to generate (for generate_data mode)'
    )
    
    parser.add_argument(
        '--tune-lr',
        action='store_true',
        help='Perform hyperparameter tuning for Logistic Regression'
    )
    
    parser.add_argument(
        '--tune-rf',
        action='store_true',
        help='Perform hyperparameter tuning for Random Forest'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print(" "*10 + "CUSTOMER CHURN PREDICTION - TELECOM COMPANY")
    print("="*70)
    
    # Execute based on mode
    try:
        if args.mode == 'generate_data':
            generate_data(args.n_samples)
        
        elif args.mode == 'train':
            train_models(tune_lr=args.tune_lr, tune_rf=args.tune_rf)
        
        elif args.mode == 'evaluate':
            evaluate_models()
        
        elif args.mode == 'shap':
            run_shap_analysis()
        
        elif args.mode == 'dashboard':
            launch_dashboard()
        
        elif args.mode == 'all':
            run_all()
        
        print("\n✓ Execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
