"""
Streamlit Dashboard for Customer Churn Prediction

Interactive web application for:
- Data exploration and visualization
- Model performance monitoring
- Individual and batch churn predictions
- Feature importance and SHAP analysis
- Customer insights and segmentation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import TelecomDataGenerator, DataLoader
from preprocessing import ChurnDataPreprocessor
from feature_engineering import ChurnFeatureEngineer
from model_training import ChurnModelTrainer
from model_evaluation import ChurnModelEvaluator
from interpretability import ChurnModelInterpreter

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load or generate dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'telecom_churn.csv'
    
    if data_path.exists():
        loader = DataLoader(data_path)
        df = loader.load_data()
    else:
        generator = TelecomDataGenerator(n_samples=7043)
        df = generator.generate_dataset()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    
    return df


@st.cache_resource
def train_models(df):
    """Train and cache models."""
    # Feature engineering
    feature_engineer = ChurnFeatureEngineer()
    df_engineered = feature_engineer.create_all_features(df.copy())
    
    # Preprocessing
    preprocessor = ChurnDataPreprocessor()
    X, y = preprocessor.fit_transform(df_engineered)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train models
    trainer = ChurnModelTrainer()
    trainer.train_logistic_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    
    return trainer, preprocessor, feature_engineer, X_train, X_test, y_train, y_test


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📈 Data Explorer", "🤖 Model Performance", 
         "🔮 Churn Prediction", "🔍 Feature Importance", "👥 Customer Insights"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Train models
    if 'models_trained' not in st.session_state:
        with st.spinner("Training models... This may take a minute."):
            trainer, preprocessor, feature_engineer, X_train, X_test, y_train, y_test = train_models(df)
            st.session_state.models_trained = True
            st.session_state.trainer = trainer
            st.session_state.preprocessor = preprocessor
            st.session_state.feature_engineer = feature_engineer
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
    else:
        trainer = st.session_state.trainer
        preprocessor = st.session_state.preprocessor
        feature_engineer = st.session_state.feature_engineer
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
    
    # Page routing
    if page == "🏠 Home":
        show_home(df, trainer, X_test, y_test)
    elif page == "📈 Data Explorer":
        show_data_explorer(df)
    elif page == "🤖 Model Performance":
        show_model_performance(trainer, X_test, y_test)
    elif page == "🔮 Churn Prediction":
        show_prediction(trainer, preprocessor, feature_engineer)
    elif page == "🔍 Feature Importance":
        show_feature_importance(trainer, X_train, X_test)
    elif page == "👥 Customer Insights":
        show_customer_insights(df)


def show_home(df, trainer, X_test, y_test):
    """Home page with overview."""
    st.header("Welcome to the Customer Churn Prediction System")
    
    st.markdown("""
    This dashboard provides comprehensive tools for analyzing and predicting customer churn
    in the telecom industry. Use the sidebar to navigate between different sections.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_charges = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
    
    st.markdown("---")
    
    # Model performance summary
    st.subheader("📊 Model Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Logistic Regression")
        y_pred_lr = trainer.predict('logistic_regression', X_test)
        y_pred_proba_lr = trainer.predict_proba('logistic_regression', X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc_lr = accuracy_score(y_test, y_pred_lr)
        auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
        
        st.metric("Accuracy", f"{acc_lr:.2%}")
        st.metric("ROC-AUC", f"{auc_lr:.4f}")
    
    with col2:
        st.markdown("### Random Forest")
        y_pred_rf = trainer.predict('random_forest', X_test)
        y_pred_proba_rf = trainer.predict_proba('random_forest', X_test)[:, 1]
        
        acc_rf = accuracy_score(y_test, y_pred_rf)
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
        
        st.metric("Accuracy", f"{acc_rf:.2%}")
        st.metric("ROC-AUC", f"{auc_rf:.4f}")
    
    # Quick insights
    st.markdown("---")
    st.subheader("🔍 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by contract type
        fig = px.histogram(df, x='Contract', color='Churn', barmode='group',
                          title='Churn by Contract Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenure distribution
        fig = px.box(df, x='Churn', y='tenure', color='Churn',
                    title='Tenure Distribution by Churn Status')
        st.plotly_chart(fig, use_container_width=True)


def show_data_explorer(df):
    """Data exploration page."""
    st.header("📈 Data Explorer")
    
    # Filters
    st.sidebar.subheader("Filters")
    
    # Contract filter
    contracts = ['All'] + df['Contract'].unique().tolist()
    selected_contract = st.sidebar.selectbox("Contract Type", contracts)
    
    # Churn filter
    churn_filter = st.sidebar.selectbox("Churn Status", ['All', 'Yes', 'No'])
    
    # Apply filters
    df_filtered = df.copy()
    if selected_contract != 'All':
        df_filtered = df_filtered[df_filtered['Contract'] == selected_contract]
    if churn_filter != 'All':
        df_filtered = df_filtered[df_filtered['Churn'] == churn_filter]
    
    st.info(f"Showing {len(df_filtered)} customers")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df_filtered.head(100), use_container_width=True)
    
    # Statistics
    st.subheader("Statistical Summary")
    st.dataframe(df_filtered.describe(), use_container_width=True)
    
    # Visualizations
    st.subheader("Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly charges distribution
        fig = px.histogram(df_filtered, x='MonthlyCharges', color='Churn',
                          title='Monthly Charges Distribution', nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Internet service distribution
        fig = px.pie(df_filtered, names='InternetService', 
                    title='Internet Service Distribution')
        st.plotly_chart(fig, use_container_width=True)


def show_model_performance(trainer, X_test, y_test):
    """Model performance page."""
    st.header("🤖 Model Performance")
    
    # Model selection
    model_name = st.selectbox("Select Model", ['Logistic Regression', 'Random Forest'])
    model_key = 'logistic_regression' if model_name == 'Logistic Regression' else 'random_forest'
    
    # Get predictions
    y_pred = trainer.predict(model_key, X_test)
    y_pred_proba = trainer.predict_proba(model_key, X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    with col2:
        st.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    with col3:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    with col4:
        st.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    with col5:
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                    x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                    title=f'Confusion Matrix - {model_name}')
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.subheader("ROC Curve")
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                            line=dict(dash='dash')))
    fig.update_layout(title=f'ROC Curve - {model_name}',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate')
    st.plotly_chart(fig, use_container_width=True)


def show_prediction(trainer, preprocessor, feature_engineer):
    """Churn prediction page."""
    st.header("🔮 Churn Prediction")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Predict Churn for a Single Customer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
            partner = st.selectbox("Partner", ['No', 'Yes'])
            dependents = st.selectbox("Dependents", ['No', 'Yes'])
        
        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
            internet_service = st.selectbox("Internet Service", ['No', 'DSL', 'Fiber optic'])
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        
        with col3:
            paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
            payment_method = st.selectbox("Payment Method", 
                                         ['Electronic check', 'Mailed check', 
                                          'Bank transfer (automatic)', 'Credit card (automatic)'])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
        
        if st.button("Predict Churn", type="primary"):
            # Create customer data
            customer_data = pd.DataFrame({
                'customerID': ['CUST0000'],
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': ['No phone service' if phone_service == 'No' else 'No'],
                'InternetService': [internet_service],
                'OnlineSecurity': ['No internet service' if internet_service == 'No' else 'No'],
                'OnlineBackup': ['No internet service' if internet_service == 'No' else 'No'],
                'DeviceProtection': ['No internet service' if internet_service == 'No' else 'No'],
                'TechSupport': ['No internet service' if internet_service == 'No' else 'No'],
                'StreamingTV': ['No internet service' if internet_service == 'No' else 'No'],
                'StreamingMovies': ['No internet service' if internet_service == 'No' else 'No'],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })
            
            # Feature engineering
            customer_data = feature_engineer.create_all_features(customer_data, fit=False)
            
            # Preprocess
            X_customer = preprocessor.transform(customer_data, scale=True)
            
            # Predict
            model_name = st.selectbox("Select Model", ['Random Forest', 'Logistic Regression'])
            model_key = 'random_forest' if model_name == 'Random Forest' else 'logistic_regression'
            
            prediction = trainer.predict(model_key, X_customer)[0]
            probability = trainer.predict_proba(model_key, X_customer)[0]
            
            # Display result
            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠️ **High Churn Risk!** Churn Probability: {probability[1]:.1%}")
                st.markdown("**Recommended Actions:**")
                st.markdown("- Offer retention incentives")
                st.markdown("- Provide personalized customer support")
                st.markdown("- Consider contract upgrade options")
            else:
                st.success(f"✅ **Low Churn Risk** Churn Probability: {probability[1]:.1%}")
                st.markdown("**Customer Status: Stable**")
    
    with tab2:
        st.subheader("Batch Prediction")
        st.markdown("Upload a CSV file with customer data for batch predictions.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_batch)} customers")
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    # Feature engineering and preprocessing
                    df_batch = feature_engineer.create_all_features(df_batch, fit=False)
                    X_batch = preprocessor.transform(df_batch, scale=True)
                    
                    # Predict
                    predictions = trainer.predict('random_forest', X_batch)
                    probabilities = trainer.predict_proba('random_forest', X_batch)[:, 1]
                    
                    # Add results
                    df_batch['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                    df_batch['Churn_Probability'] = probabilities
                    
                    st.success("Predictions completed!")
                    st.dataframe(df_batch[['customerID', 'Churn_Prediction', 'Churn_Probability']].head(50))
                    
                    # Download results
                    csv = df_batch.to_csv(index=False)
                    st.download_button("Download Results", csv, "churn_predictions.csv", "text/csv")


def show_feature_importance(trainer, X_train, X_test):
    """Feature importance page."""
    st.header("🔍 Feature Importance Analysis")
    
    model_name = st.selectbox("Select Model", ['Random Forest', 'Logistic Regression'])
    model_key = 'random_forest' if model_name == 'Random Forest' else 'logistic_regression'
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance(model_key, X_train.columns)
    
    # Plot
    fig = px.bar(feature_importance.head(20), x='importance', y='feature', 
                orientation='h', title=f'Top 20 Feature Importance - {model_name}')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Show table
    st.subheader("Feature Importance Table")
    st.dataframe(feature_importance, use_container_width=True)


def show_customer_insights(df):
    """Customer insights page."""
    st.header("👥 Customer Insights")
    
    # Churn by segment
    st.subheader("Churn Analysis by Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # By contract
        churn_by_contract = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        churn_by_contract.columns = ['Contract', 'Churn_Rate']
        
        fig = px.bar(churn_by_contract, x='Contract', y='Churn_Rate',
                    title='Churn Rate by Contract Type', 
                    labels={'Churn_Rate': 'Churn Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # By internet service
        churn_by_internet = df.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        churn_by_internet.columns = ['InternetService', 'Churn_Rate']
        
        fig = px.bar(churn_by_internet, x='InternetService', y='Churn_Rate',
                    title='Churn Rate by Internet Service',
                    labels={'Churn_Rate': 'Churn Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Tenure vs Charges
    st.subheader("Tenure vs Monthly Charges")
    fig = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                    title='Customer Tenure vs Monthly Charges',
                    labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'})
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
