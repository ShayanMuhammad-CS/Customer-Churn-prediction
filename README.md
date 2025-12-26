# 📊 Customer Churn Prediction for Telecom Company

A comprehensive end-to-end machine learning project to predict customer churn in the telecom industry using advanced feature engineering, multiple ML models, and interactive visualizations.

## 🎯 Project Objectives

- **Predict Customer Churn**: Identify customers likely to leave the service
- **Feature Engineering**: Create meaningful features including tenure groups and LTV segmentation
- **Model Comparison**: Train and compare Logistic Regression and Random Forest models
- **Interpretability**: Use SHAP values to explain model predictions
- **Interactive Dashboard**: Provide a user-friendly interface for predictions and insights

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd CUSTOMER_CHURn
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run the Complete Pipeline
```bash
python main.py --mode all
```

#### Option 2: Step-by-Step Execution
```bash
# Generate synthetic dataset
python main.py --mode generate_data

# Train models
python main.py --mode train

# Evaluate models
python main.py --mode evaluate

# Launch dashboard
python main.py --mode dashboard
```

#### Option 3: Use Jupyter Notebooks
```bash
jupyter notebook
```
Then open notebooks in order:
1. `01_data_exploration.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_model_training.ipynb`

#### Option 4: Launch Dashboard Directly
```bash
streamlit run dashboard/streamlit_app.py
```

## 📁 Project Structure

```
CUSTOMER_CHURn/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_loader.py          # Data loading and generation
│   ├── preprocessing.py        # Data preprocessing
│   ├── feature_engineering.py  # Feature engineering
│   ├── model_training.py       # Model training pipeline
│   ├── model_evaluation.py     # Evaluation metrics
│   └── interpretability.py     # SHAP analysis
├── models/
│   └── saved_models/           # Trained model artifacts
├── visualizations/
│   └── plots/                  # Generated plots
├── dashboard/
│   └── streamlit_app.py        # Interactive dashboard
├── requirements.txt
├── README.md
└── main.py                     # Main execution script
```

## 🔍 Key Features

### 1. Advanced Feature Engineering
- **Tenure Groups**: Categorize customers by contract length (0-12, 12-24, 24-48, 48+ months)
- **LTV Segmentation**: Calculate Customer Lifetime Value and create segments (Low, Medium, High, Premium)
- **Interaction Features**: Charges per service, contract-payment combinations
- **Service Aggregation**: Total services used, service diversity metrics

### 2. Machine Learning Models
- **Logistic Regression**: Fast, interpretable baseline model
- **Random Forest**: Powerful ensemble model for complex patterns
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters

### 3. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, Precision-Recall curves
- **Comparison**: Side-by-side model performance analysis

### 4. Model Interpretability (SHAP)
- **Summary Plots**: Global feature importance
- **Force Plots**: Individual prediction explanations
- **Dependence Plots**: Feature interaction analysis
- **Waterfall Plots**: Detailed contribution breakdown

### 5. Interactive Dashboard
- **Data Explorer**: Filter and visualize customer data
- **Model Performance**: Real-time metrics and charts
- **Churn Prediction**: Single and batch customer predictions
- **Feature Importance**: Interactive SHAP visualizations
- **Customer Insights**: Segment-wise churn analysis

## 📊 Dataset Features

The synthetic telecom dataset includes:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, InternetService, OnlineSecurity, TechSupport, etc.
- **Account**: Contract type, PaymentMethod, PaperlessBilling
- **Charges**: MonthlyCharges, TotalCharges
- **Tenure**: Months with the company
- **Target**: Churn (Yes/No)

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ Advanced feature engineering techniques
- ✅ Model training and hyperparameter tuning
- ✅ Model evaluation and comparison
- ✅ Explainable AI with SHAP
- ✅ Interactive dashboard development
- ✅ Best practices in ML project structure

## 📈 Expected Results

- **Model Accuracy**: 78-85% (depending on data quality)
- **ROC-AUC Score**: 0.80-0.88
- **Key Churn Indicators**: Contract type, tenure, monthly charges, internet service

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas & numpy**: Data manipulation
- **scikit-learn**: Machine learning
- **SHAP**: Model interpretability
- **matplotlib & seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **Streamlit**: Web dashboard

## 📝 Notes

- The project includes a synthetic dataset generator for demonstration
- All code is beginner-friendly with detailed comments
- Notebooks provide step-by-step explanations
- Dashboard is production-ready and can be deployed

## 🤝 Contributing

Feel free to enhance this project by:
- Adding more ML models (XGBoost, Neural Networks)
- Implementing additional features
- Improving the dashboard UI
- Adding automated testing

## 📄 License

This project is for educational and demonstration purposes.

---

**Happy Learning! 🚀**
