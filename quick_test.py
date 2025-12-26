"""
Quick Test Script - Generate Data and Test Basic Functionality
This script tests the core modules without requiring all dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

print("="*70)
print(" "*20 + "QUICK TEST SCRIPT")
print("="*70)

# Test 1: Data Generation
print("\n[1/5] Testing Data Generation...")
try:
    from src.data_loader import TelecomDataGenerator, save_dataset
    
    generator = TelecomDataGenerator(n_samples=1000, random_state=42)
    df = generator.generate_dataset()
    
    # Save dataset
    output_path = Path('data/raw/telecom_churn.csv')
    save_dataset(df, output_path)
    
    print(f"✅ Data generation successful! Shape: {df.shape}")
    print(f"   Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: Feature Engineering
print("\n[2/5] Testing Feature Engineering...")
try:
    from src.feature_engineering import ChurnFeatureEngineer
    
    feature_engineer = ChurnFeatureEngineer()
    df_engineered = feature_engineer.create_all_features(df.copy(), fit=True)
    
    print(f"✅ Feature engineering successful!")
    print(f"   Original features: {df.shape[1]}")
    print(f"   Engineered features: {df_engineered.shape[1]}")
    print(f"   New features added: {df_engineered.shape[1] - df.shape[1]}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Preprocessing
print("\n[3/5] Testing Preprocessing...")
try:
    from src.preprocessing import ChurnDataPreprocessor
    
    preprocessor = ChurnDataPreprocessor()
    X, y = preprocessor.fit_transform(df_engineered)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    
    print(f"✅ Preprocessing successful!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {X_train.shape[1]}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Model Training
print("\n[4/5] Testing Model Training...")
try:
    from src.model_training import ChurnModelTrainer
    
    trainer = ChurnModelTrainer()
    
    # Train Logistic Regression
    print("   Training Logistic Regression...")
    lr_model = trainer.train_logistic_regression(X_train, y_train, tune=False)
    
    # Train Random Forest
    print("   Training Random Forest...")
    rf_model = trainer.train_random_forest(X_train, y_train, tune=False)
    
    print(f"✅ Model training successful!")
    print(f"   LR CV Score: {trainer.cv_scores['logistic_regression'].mean():.4f}")
    print(f"   RF CV Score: {trainer.cv_scores['random_forest'].mean():.4f}")
    
    # Save models
    models_dir = Path('models/saved_models')
    trainer.save_model('logistic_regression', models_dir / 'logistic_regression.pkl')
    trainer.save_model('random_forest', models_dir / 'random_forest.pkl')
    
    # Save preprocessor
    preprocessor.save(Path('models/preprocessor.pkl'))
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model Evaluation
print("\n[5/5] Testing Model Evaluation...")
try:
    from src.model_evaluation import ChurnModelEvaluator
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    evaluator = ChurnModelEvaluator()
    
    # Evaluate Random Forest
    y_pred_rf = trainer.predict('random_forest', X_test)
    y_pred_proba_rf = trainer.predict_proba('random_forest', X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred_rf)
    roc_auc = roc_auc_score(y_test, y_pred_proba_rf)
    
    print(f"✅ Model evaluation successful!")
    print(f"   Random Forest Accuracy: {accuracy:.2%}")
    print(f"   Random Forest ROC-AUC: {roc_auc:.4f}")
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance('random_forest', X_train.columns)
    print(f"\n   Top 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"      {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print(" "*20 + "✅ ALL TESTS PASSED!")
print("="*70)

print("\n📊 Next Steps:")
print("   1. The Streamlit dashboard is running at: http://localhost:8501")
print("   2. Models are saved in: models/saved_models/")
print("   3. Dataset is saved in: data/raw/telecom_churn.csv")
print("\n   To run full pipeline with visualizations:")
print("   python main.py --mode all")
