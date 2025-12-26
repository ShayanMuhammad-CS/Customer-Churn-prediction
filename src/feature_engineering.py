"""
Feature Engineering Module for Customer Churn Prediction

This module handles:
- Tenure grouping (categorize customers by contract length)
- LTV (Lifetime Value) segmentation
- Interaction features creation
- Service aggregation features
"""

import pandas as pd
import numpy as np
from pathlib import Path


class ChurnFeatureEngineer:
    """Advanced feature engineering for churn prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.ltv_thresholds = None
        
    def create_tenure_groups(self, df):
        """
        Create tenure groups based on customer tenure.
        
        Tenure groups:
        - 0-12 months: New customers (high churn risk)
        - 12-24 months: Growing customers
        - 24-48 months: Established customers
        - 48+ months: Loyal customers (low churn risk)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data with 'tenure' column
        
        Returns:
        --------
        pd.DataFrame
            Data with tenure group features
        """
        df = df.copy()
        
        print("\n" + "="*60)
        print("CREATING TENURE GROUPS")
        print("="*60)
        
        # Create tenure groups
        df['TenureGroup'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, np.inf],
            labels=['0-12 months', '12-24 months', '24-48 months', '48+ months'],
            include_lowest=True
        )
        
        # Create binary features for each tenure group
        df['Tenure_New'] = (df['tenure'] <= 12).astype(int)
        df['Tenure_Growing'] = ((df['tenure'] > 12) & (df['tenure'] <= 24)).astype(int)
        df['Tenure_Established'] = ((df['tenure'] > 24) & (df['tenure'] <= 48)).astype(int)
        df['Tenure_Loyal'] = (df['tenure'] > 48).astype(int)
        
        # Tenure in years (continuous)
        df['TenureYears'] = df['tenure'] / 12
        
        print("\nTenure Group Distribution:")
        print(df['TenureGroup'].value_counts().sort_index())
        
        print("\n✓ Created tenure group features:")
        print("  - TenureGroup (categorical)")
        print("  - Tenure_New, Tenure_Growing, Tenure_Established, Tenure_Loyal (binary)")
        print("  - TenureYears (continuous)")
        print("="*60)
        
        return df
    
    def create_ltv_segmentation(self, df, fit=True):
        """
        Create Customer Lifetime Value (LTV) segmentation.
        
        LTV = TotalCharges (or estimated: tenure * MonthlyCharges)
        Segments: Low, Medium, High, Premium
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data with charge information
        fit : bool
            Whether to fit thresholds (True for training, False for test)
        
        Returns:
        --------
        pd.DataFrame
            Data with LTV features
        """
        df = df.copy()
        
        print("\n" + "="*60)
        print("CREATING LTV SEGMENTATION")
        print("="*60)
        
        # Calculate LTV
        if 'TotalCharges' in df.columns:
            df['LTV'] = df['TotalCharges']
        else:
            df['LTV'] = df['tenure'] * df['MonthlyCharges']
        
        # Handle any missing or zero LTV
        df['LTV'] = df['LTV'].fillna(0)
        
        if fit:
            # Define thresholds based on quartiles
            self.ltv_thresholds = {
                'low': df['LTV'].quantile(0.25),
                'medium': df['LTV'].quantile(0.50),
                'high': df['LTV'].quantile(0.75)
            }
            print(f"\nLTV Thresholds:")
            print(f"  Low: ${self.ltv_thresholds['low']:.2f}")
            print(f"  Medium: ${self.ltv_thresholds['medium']:.2f}")
            print(f"  High: ${self.ltv_thresholds['high']:.2f}")
        
        # Create LTV segments
        df['LTV_Segment'] = pd.cut(
            df['LTV'],
            bins=[0, self.ltv_thresholds['low'], self.ltv_thresholds['medium'], 
                  self.ltv_thresholds['high'], np.inf],
            labels=['Low', 'Medium', 'High', 'Premium'],
            include_lowest=True
        )
        
        # Create binary features for each segment
        df['LTV_Low'] = (df['LTV_Segment'] == 'Low').astype(int)
        df['LTV_Medium'] = (df['LTV_Segment'] == 'Medium').astype(int)
        df['LTV_High'] = (df['LTV_Segment'] == 'High').astype(int)
        df['LTV_Premium'] = (df['LTV_Segment'] == 'Premium').astype(int)
        
        # LTV per month (average monthly value)
        df['LTV_PerMonth'] = df['LTV'] / (df['tenure'] + 1)  # +1 to avoid division by zero
        
        print("\nLTV Segment Distribution:")
        print(df['LTV_Segment'].value_counts().sort_index())
        
        print("\n✓ Created LTV features:")
        print("  - LTV (continuous)")
        print("  - LTV_Segment (categorical)")
        print("  - LTV_Low, LTV_Medium, LTV_High, LTV_Premium (binary)")
        print("  - LTV_PerMonth (continuous)")
        print("="*60)
        
        return df
    
    def create_service_features(self, df):
        """
        Create aggregated service features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data with service columns
        
        Returns:
        --------
        pd.DataFrame
            Data with service aggregation features
        """
        df = df.copy()
        
        print("\n" + "="*60)
        print("CREATING SERVICE FEATURES")
        print("="*60)
        
        # Count total services
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies']
        
        # Total number of services (count 'Yes' values)
        df['TotalServices'] = 0
        for col in service_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df['TotalServices'] += (df[col] == 'Yes').astype(int)
                else:
                    df['TotalServices'] += df[col]
        
        # Has internet service
        if 'InternetService' in df.columns:
            if df['InternetService'].dtype == 'object':
                df['HasInternet'] = (df['InternetService'] != 'No').astype(int)
            else:
                df['HasInternet'] = df['InternetService']
        
        # Has phone service
        if 'PhoneService' in df.columns:
            if df['PhoneService'].dtype == 'object':
                df['HasPhone'] = (df['PhoneService'] == 'Yes').astype(int)
            else:
                df['HasPhone'] = df['PhoneService']
        
        # Count of protection services
        protection_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['ProtectionServices'] = 0
        for col in protection_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df['ProtectionServices'] += (df[col] == 'Yes').astype(int)
                else:
                    df['ProtectionServices'] += df[col]
        
        # Count of streaming services
        streaming_cols = ['StreamingTV', 'StreamingMovies']
        df['StreamingServices'] = 0
        for col in streaming_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df['StreamingServices'] += (df[col] == 'Yes').astype(int)
                else:
                    df['StreamingServices'] += df[col]
        
        # Service diversity ratio (services used / total available)
        df['ServiceDiversity'] = df['TotalServices'] / len(service_cols)
        
        print("\n✓ Created service features:")
        print("  - TotalServices: Total number of services")
        print("  - HasInternet: Has internet service (binary)")
        print("  - HasPhone: Has phone service (binary)")
        print("  - ProtectionServices: Count of protection services")
        print("  - StreamingServices: Count of streaming services")
        print("  - ServiceDiversity: Ratio of services used")
        
        print(f"\nService Statistics:")
        print(f"  Average services per customer: {df['TotalServices'].mean():.2f}")
        print(f"  Max services: {df['TotalServices'].max()}")
        print("="*60)
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between important variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        
        Returns:
        --------
        pd.DataFrame
            Data with interaction features
        """
        df = df.copy()
        
        print("\n" + "="*60)
        print("CREATING INTERACTION FEATURES")
        print("="*60)
        
        # Monthly charges per service
        df['ChargesPerService'] = df['MonthlyCharges'] / (df['TotalServices'] + 1)
        
        # Average monthly charges (total / tenure)
        df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Charges increase ratio
        df['ChargesIncreaseRatio'] = df['MonthlyCharges'] / (df['AvgMonthlyCharges'] + 1)
        
        # Senior citizen with dependents
        if 'SeniorCitizen' in df.columns and 'Dependents' in df.columns:
            if df['SeniorCitizen'].dtype == 'object':
                senior = (df['SeniorCitizen'] == 'Yes').astype(int)
            else:
                senior = df['SeniorCitizen']
            
            if df['Dependents'].dtype == 'object':
                dependents = (df['Dependents'] == 'Yes').astype(int)
            else:
                dependents = df['Dependents']
            
            df['SeniorWithDependents'] = senior * dependents
        
        # Contract type with payment method (high-risk combination)
        if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
            if df['Contract'].dtype == 'object' and df['PaymentMethod'].dtype == 'object':
                df['MonthToMonth_ElectronicCheck'] = (
                    (df['Contract'] == 'Month-to-month') & 
                    (df['PaymentMethod'] == 'Electronic check')
                ).astype(int)
        
        # High charges with short tenure (churn risk)
        median_charges = df['MonthlyCharges'].median()
        df['HighCharges_ShortTenure'] = (
            (df['MonthlyCharges'] > median_charges) & 
            (df['tenure'] <= 12)
        ).astype(int)
        
        # No protection services with internet
        if 'ProtectionServices' in df.columns and 'HasInternet' in df.columns:
            df['Internet_NoProtection'] = (
                (df['HasInternet'] == 1) & 
                (df['ProtectionServices'] == 0)
            ).astype(int)
        
        print("\n✓ Created interaction features:")
        print("  - ChargesPerService: Monthly charges divided by services")
        print("  - AvgMonthlyCharges: Historical average monthly charges")
        print("  - ChargesIncreaseRatio: Current vs historical charges")
        print("  - SeniorWithDependents: Senior citizen with dependents")
        print("  - MonthToMonth_ElectronicCheck: High-risk payment combination")
        print("  - HighCharges_ShortTenure: Expensive plan with short tenure")
        print("  - Internet_NoProtection: Internet without protection services")
        print("="*60)
        
        return df
    
    def create_all_features(self, df, fit=True):
        """
        Create all engineered features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        fit : bool
            Whether to fit parameters (True for training, False for test)
        
        Returns:
        --------
        pd.DataFrame
            Data with all engineered features
        """
        print("\n" + "="*70)
        print(" "*20 + "FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Create tenure groups
        df = self.create_tenure_groups(df)
        
        # Create LTV segmentation
        df = self.create_ltv_segmentation(df, fit=fit)
        
        # Create service features
        df = self.create_service_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        print("\n" + "="*70)
        print(" "*20 + "FEATURE ENGINEERING COMPLETED")
        print("="*70)
        
        # Count new features
        original_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
        
        new_features = [col for col in df.columns if col not in original_cols]
        
        print(f"\n✓ Created {len(new_features)} new features")
        print(f"✓ Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_summary(self, df):
        """
        Get summary of engineered features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with engineered features
        
        Returns:
        --------
        dict
            Feature summary statistics
        """
        summary = {
            'tenure_groups': df['TenureGroup'].value_counts().to_dict() if 'TenureGroup' in df.columns else None,
            'ltv_segments': df['LTV_Segment'].value_counts().to_dict() if 'LTV_Segment' in df.columns else None,
            'avg_total_services': df['TotalServices'].mean() if 'TotalServices' in df.columns else None,
            'avg_ltv': df['LTV'].mean() if 'LTV' in df.columns else None,
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    from data_loader import TelecomDataGenerator
    
    # Generate sample data
    generator = TelecomDataGenerator(n_samples=1000)
    df = generator.generate_dataset()
    
    print(f"\nOriginal dataset shape: {df.shape}")
    
    # Initialize feature engineer
    feature_engineer = ChurnFeatureEngineer()
    
    # Create all features
    df_engineered = feature_engineer.create_all_features(df, fit=True)
    
    print(f"\nEngineered dataset shape: {df_engineered.shape}")
    print(f"\nNew features added: {df_engineered.shape[1] - df.shape[1]}")
    
    # Get feature summary
    summary = feature_engineer.get_feature_summary(df_engineered)
    print("\n✓ Feature engineering module test completed successfully!")
