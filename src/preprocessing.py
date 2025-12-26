"""
Preprocessing Module for Customer Churn Prediction

This module handles:
- Missing value imputation
- Categorical encoding (One-Hot, Label Encoding)
- Numeric feature scaling
- Train-test splitting with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import joblib


class ChurnDataPreprocessor:
    """Comprehensive preprocessing pipeline for churn prediction data."""
    
    def __init__(self):
        """Initialize the preprocessor with necessary encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Churn'
        self.id_column = 'customerID'
        
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        strategy : str
            'auto', 'drop', 'mean', 'median', 'mode'
        
        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        df = df.copy()
        
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)
        
        # Check for missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if len(missing_cols) == 0:
            print("✓ No missing values found")
            return df
        
        print(f"\nMissing values found in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            pct = (count / len(df) * 100)
            print(f"  - {col}: {count} ({pct:.2f}%)")
        
        # Handle TotalCharges specifically (common issue in telecom data)
        if 'TotalCharges' in missing_cols:
            # For customers with 0 tenure, TotalCharges should be 0
            df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
            
            # For others, impute with tenure * MonthlyCharges
            mask = df['TotalCharges'].isnull()
            df.loc[mask, 'TotalCharges'] = df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']
            
            print(f"\n✓ Imputed TotalCharges using tenure * MonthlyCharges")
        
        # Handle other numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if strategy == 'mean' or strategy == 'auto':
                    df[col].fillna(df[col].mean(), inplace=True)
                    print(f"✓ Filled {col} with mean")
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                    print(f"✓ Filled {col} with median")
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"✓ Filled {col} with mode")
        
        print("\n✓ Missing value handling completed")
        print("="*60)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        fit : bool
            Whether to fit encoders (True for training, False for test/new data)
        
        Returns:
        --------
        pd.DataFrame
            Data with encoded categorical features
        """
        df = df.copy()
        
        print("\n" + "="*60)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*60)
        
        # Drop categorical columns created by feature engineering (they're already encoded as binary features)
        categorical_fe_cols = ['TenureGroup', 'LTV_Segment']
        cols_to_drop = [col for col in categorical_fe_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"\n✓ Dropped categorical feature engineering columns: {cols_to_drop}")
        
        # Separate features to encode
        # Binary categorical features (will be label encoded)
        binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'PaperlessBilling', 'Churn']
        
        # Multi-class categorical features (will be one-hot encoded)
        multi_class_features = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
        
        # Label encode binary features
        for col in binary_features:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"✓ Label encoded: {col} -> {list(le.classes_)}")
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # One-hot encode multi-class features
        if fit:
            df = pd.get_dummies(df, columns=[col for col in multi_class_features if col in df.columns], 
                               drop_first=True, dtype=int)
            print(f"\n✓ One-hot encoded {len(multi_class_features)} multi-class features")
        else:
            # For test data, ensure same columns as training
            df = pd.get_dummies(df, columns=[col for col in multi_class_features if col in df.columns], 
                               drop_first=True, dtype=int)
            
            # Add missing columns with 0s
            if self.feature_columns is not None:
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0
                
                # Remove extra columns
                extra_cols = set(df.columns) - set(self.feature_columns) - {self.id_column, self.target_column}
                df = df.drop(columns=list(extra_cols))
        
        print("="*60)
        
        return df
    
    def scale_numeric_features(self, df, fit=True, exclude_cols=None):
        """
        Scale numeric features using StandardScaler.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        fit : bool
            Whether to fit scaler (True for training, False for test/new data)
        exclude_cols : list
            Columns to exclude from scaling
        
        Returns:
        --------
        pd.DataFrame
            Data with scaled numeric features
        """
        df = df.copy()
        
        if exclude_cols is None:
            exclude_cols = [self.id_column, self.target_column]
        
        # Get numeric columns to scale
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        print("\n" + "="*60)
        print("SCALING NUMERIC FEATURES")
        print("="*60)
        print(f"\nScaling {len(cols_to_scale)} numeric features:")
        print(f"  {cols_to_scale}")
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            print("\n✓ Fitted and transformed features")
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            print("\n✓ Transformed features using existing scaler")
        
        print("="*60)
        
        return df
    
    def prepare_features_target(self, df):
        """
        Separate features and target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed data
        
        Returns:
        --------
        tuple
            (X, y) where X is features and y is target
        """
        # Drop ID column if present
        if self.id_column in df.columns:
            df = df.drop(columns=[self.id_column])
        
        # Separate features and target
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Store feature columns for later use
            self.feature_columns = X.columns.tolist()
            
            return X, y
        else:
            # For prediction on new data without target
            self.feature_columns = df.columns.tolist()
            return df, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed
        stratify : bool
            Whether to stratify split by target variable
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("SPLITTING DATA")
        print("="*60)
        
        stratify_by = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_by
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        if stratify:
            print(f"\n✓ Stratified split by target variable")
            print(f"  Train churn rate: {y_train.mean():.2%}")
            print(f"  Test churn rate: {y_test.mean():.2%}")
        
        print("="*60)
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, df, scale=True):
        """
        Fit and transform the entire preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw input data
        scale : bool
            Whether to scale numeric features
        
        Returns:
        --------
        tuple
            (X, y) preprocessed features and target
        """
        print("\n" + "="*70)
        print(" "*20 + "PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Step 3: Scale numeric features
        if scale:
            df = self.scale_numeric_features(df, fit=True)
        
        # Step 4: Prepare features and target
        X, y = self.prepare_features_target(df)
        
        print("\n" + "="*70)
        print(" "*20 + "PREPROCESSING COMPLETED")
        print("="*70)
        print(f"\n✓ Final feature matrix shape: {X.shape}")
        print(f"✓ Target variable shape: {y.shape}")
        print(f"✓ Number of features: {len(self.feature_columns)}")
        
        return X, y
    
    def transform(self, df, scale=True):
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            New data to transform
        scale : bool
            Whether to scale numeric features
        
        Returns:
        --------
        pd.DataFrame or tuple
            Transformed features (and target if present)
        """
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Step 3: Scale numeric features
        if scale:
            df = self.scale_numeric_features(df, fit=False)
        
        # Step 4: Prepare features and target
        X, y = self.prepare_features_target(df)
        
        if y is not None:
            return X, y
        else:
            return X
    
    def save(self, filepath):
        """Save preprocessor to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        
        print(f"✓ Preprocessor saved to: {filepath}")
    
    def load(self, filepath):
        """Load preprocessor from file."""
        data = joblib.load(filepath)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        
        print(f"✓ Preprocessor loaded from: {filepath}")


# Example usage
if __name__ == "__main__":
    from data_loader import TelecomDataGenerator
    
    # Generate sample data
    generator = TelecomDataGenerator(n_samples=1000)
    df = generator.generate_dataset()
    
    # Initialize preprocessor
    preprocessor = ChurnDataPreprocessor()
    
    # Fit and transform
    X, y = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print("\n✓ Preprocessing module test completed successfully!")
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
