"""
Data Loader Module for Customer Churn Prediction

This module handles:
- Generating synthetic telecom customer data
- Loading datasets from various sources
- Initial data validation and quality checks
"""

import pandas as pd
import numpy as np
from pathlib import Path


class TelecomDataGenerator:
    """Generate realistic synthetic telecom customer data for churn prediction."""
    
    def __init__(self, n_samples=7043, random_state=42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        n_samples : int
            Number of customer records to generate
        random_state : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_dataset(self):
        """
        Generate a complete synthetic telecom dataset.
        
        Returns:
        --------
        pd.DataFrame
            Synthetic customer data with churn labels
        """
        print(f"Generating {self.n_samples} customer records...")
        
        # Customer Demographics
        gender = np.random.choice(['Male', 'Female'], self.n_samples)
        senior_citizen = np.random.choice([0, 1], self.n_samples, p=[0.84, 0.16])
        partner = np.random.choice(['Yes', 'No'], self.n_samples, p=[0.52, 0.48])
        dependents = np.random.choice(['Yes', 'No'], self.n_samples, p=[0.30, 0.70])
        
        # Tenure (months with company)
        tenure = np.random.exponential(scale=25, size=self.n_samples).astype(int)
        tenure = np.clip(tenure, 0, 72)  # Cap at 72 months
        
        # Phone Services
        phone_service = np.random.choice(['Yes', 'No'], self.n_samples, p=[0.90, 0.10])
        multiple_lines = np.where(
            phone_service == 'Yes',
            np.random.choice(['Yes', 'No', 'No phone service'], self.n_samples, p=[0.42, 0.48, 0.10]),
            'No phone service'
        )
        
        # Internet Services
        internet_service = np.random.choice(
            ['DSL', 'Fiber optic', 'No'], 
            self.n_samples, 
            p=[0.34, 0.44, 0.22]
        )
        
        # Internet-dependent services
        def internet_dependent_service(base_prob=0.5):
            return np.where(
                internet_service != 'No',
                np.random.choice(['Yes', 'No'], self.n_samples, p=[base_prob, 1-base_prob]),
                'No internet service'
            )
        
        online_security = internet_dependent_service(0.29)
        online_backup = internet_dependent_service(0.34)
        device_protection = internet_dependent_service(0.34)
        tech_support = internet_dependent_service(0.29)
        streaming_tv = internet_dependent_service(0.38)
        streaming_movies = internet_dependent_service(0.39)
        
        # Contract and Billing
        contract = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            self.n_samples, 
            p=[0.55, 0.21, 0.24]
        )
        paperless_billing = np.random.choice(['Yes', 'No'], self.n_samples, p=[0.59, 0.41])
        payment_method = np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            self.n_samples,
            p=[0.34, 0.23, 0.22, 0.21]
        )
        
        # Monthly Charges (influenced by services)
        base_charge = 20
        service_charges = (
            (phone_service == 'Yes') * 10 +
            (multiple_lines == 'Yes') * 10 +
            (internet_service == 'DSL') * 25 +
            (internet_service == 'Fiber optic') * 50 +
            (online_security == 'Yes') * 5 +
            (online_backup == 'Yes') * 5 +
            (device_protection == 'Yes') * 5 +
            (tech_support == 'Yes') * 5 +
            (streaming_tv == 'Yes') * 10 +
            (streaming_movies == 'Yes') * 10
        )
        monthly_charges = base_charge + service_charges + np.random.normal(0, 5, self.n_samples)
        monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
        
        # Total Charges (tenure * monthly charges with some variation)
        total_charges = tenure * monthly_charges + np.random.normal(0, 100, self.n_samples)
        total_charges = np.maximum(total_charges, 0)
        
        # Churn (influenced by multiple factors)
        churn_probability = self._calculate_churn_probability(
            tenure, contract, internet_service, monthly_charges, 
            senior_citizen, payment_method, tech_support, online_security
        )
        churn = (np.random.random(self.n_samples) < churn_probability).astype(int)
        churn = np.where(churn == 1, 'Yes', 'No')
        
        # Create DataFrame
        df = pd.DataFrame({
            'customerID': [f'CUST{i:04d}' for i in range(self.n_samples)],
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': np.round(monthly_charges, 2),
            'TotalCharges': np.round(total_charges, 2),
            'Churn': churn
        })
        
        # Introduce some missing values in TotalCharges (realistic scenario)
        missing_indices = np.random.choice(
            self.n_samples, 
            size=int(0.002 * self.n_samples), 
            replace=False
        )
        df.loc[missing_indices, 'TotalCharges'] = np.nan
        
        print(f"✓ Generated dataset with shape: {df.shape}")
        print(f"✓ Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
        
        return df
    
    def _calculate_churn_probability(self, tenure, contract, internet_service, 
                                    monthly_charges, senior_citizen, payment_method,
                                    tech_support, online_security):
        """Calculate churn probability based on customer characteristics."""
        # Base probability
        prob = np.full(self.n_samples, 0.15)
        
        # Tenure effect (longer tenure = less churn)
        prob += np.where(tenure < 6, 0.35, 0)
        prob += np.where((tenure >= 6) & (tenure < 12), 0.20, 0)
        prob += np.where(tenure > 48, -0.10, 0)
        
        # Contract effect
        prob += np.where(contract == 'Month-to-month', 0.25, 0)
        prob += np.where(contract == 'Two year', -0.15, 0)
        
        # Internet service effect
        prob += np.where(internet_service == 'Fiber optic', 0.15, 0)
        
        # Monthly charges effect
        prob += (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.08
        
        # Senior citizen effect
        prob += senior_citizen * 0.08
        
        # Payment method effect
        prob += np.where(payment_method == 'Electronic check', 0.12, 0)
        
        # Support services effect (reduce churn)
        prob += np.where(tech_support == 'Yes', -0.10, 0)
        prob += np.where(online_security == 'Yes', -0.08, 0)
        
        # Clip probabilities to valid range
        prob = np.clip(prob, 0.05, 0.85)
        
        return prob


class DataLoader:
    """Load and validate customer churn data."""
    
    def __init__(self, data_path=None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_path : str or Path, optional
            Path to the data file
        """
        self.data_path = Path(data_path) if data_path else None
    
    def load_data(self, file_path=None):
        """
        Load customer data from CSV file.
        
        Parameters:
        -----------
        file_path : str or Path, optional
            Path to CSV file. If None, uses self.data_path
        
        Returns:
        --------
        pd.DataFrame
            Loaded customer data
        """
        path = Path(file_path) if file_path else self.data_path
        
        if path is None:
            raise ValueError("No data path provided")
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def validate_data(self, df):
        """
        Perform basic data validation checks.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Customer data to validate
        
        Returns:
        --------
        dict
            Validation report
        """
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        report = {}
        
        # Check shape
        report['n_rows'] = len(df)
        report['n_columns'] = len(df.columns)
        print(f"\n📊 Dataset Shape: {df.shape}")
        
        # Check missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        report['missing_values'] = missing[missing > 0].to_dict()
        
        if missing.sum() > 0:
            print(f"\n⚠️  Missing Values Found:")
            for col, count in missing[missing > 0].items():
                print(f"   - {col}: {count} ({missing_pct[col]}%)")
        else:
            print(f"\n✓ No missing values found")
        
        # Check data types
        print(f"\n📋 Data Types:")
        print(df.dtypes.value_counts())
        
        # Check target variable
        if 'Churn' in df.columns:
            churn_dist = df['Churn'].value_counts()
            churn_pct = (churn_dist / len(df) * 100).round(2)
            report['churn_distribution'] = churn_dist.to_dict()
            
            print(f"\n🎯 Target Variable (Churn) Distribution:")
            for value, count in churn_dist.items():
                print(f"   - {value}: {count} ({churn_pct[value]}%)")
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        report['duplicates'] = duplicates
        
        if duplicates > 0:
            print(f"\n⚠️  Duplicate rows found: {duplicates}")
        else:
            print(f"\n✓ No duplicate rows found")
        
        print("\n" + "="*60)
        
        return report
    
    def get_data_summary(self, df):
        """
        Get comprehensive data summary.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Customer data
        
        Returns:
        --------
        dict
            Summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        return summary


def save_dataset(df, output_path, create_dirs=True):
    """
    Save dataset to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to save
    output_path : str or Path
        Output file path
    create_dirs : bool
        Whether to create parent directories if they don't exist
    """
    output_path = Path(output_path)
    
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Dataset saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic dataset
    generator = TelecomDataGenerator(n_samples=7043)
    df = generator.generate_dataset()
    
    # Save to file
    output_path = Path(__file__).parent.parent / 'data' / 'raw' / 'telecom_churn.csv'
    save_dataset(df, output_path)
    
    # Load and validate
    loader = DataLoader(output_path)
    df_loaded = loader.load_data()
    validation_report = loader.validate_data(df_loaded)
    
    print("\n✓ Data loader module test completed successfully!")
