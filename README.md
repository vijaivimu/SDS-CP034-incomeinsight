# Welcome to the SuperDataScience Community Project!
Welcome to the **IcomeInsight: Predicting Earning Potential from Demographic & Employment Data** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:

## Project Overview
**IncomeInsight** uses the classic Adult Census dataset to predict whether a person earns more than $50K/year based on demographic, education, and work-related attributes.

Participants will explore the social and economic indicators behind income inequality while gaining experience in model development, experiment tracking, and real-world deployment.

This project is offered in two tracks:

- 🟢 **Beginner Track** – ML pipeline using classification models and deployment
- 🔴 **Advanced Track** – Deep learning on tabular data and SHAP interpretability

Link to dataset: https://www.kaggle.com/datasets/uciml/adult-census-income


## 🟢 Beginner Track: Income Classification with ML
### Objectives
#### Exploratory Data Analysis
- Handle missing values and inconsistent formatting (e.g., “?” entries)
- Encode categorical features and normalize continuous ones

**Key Questions to Answer**:
- What features show the strongest correlation with earning >$50K?
- How does income vary with education, marital status, or hours worked per week?
- Are there disparities across race, sex, or native country?
- Do capital gains/losses strongly impact the income label?

#### Model Development
- Train and compare classification models (Logistic Regression, Random Forest, XGBoost)
- Use `income` as the binary target variable
- Evaluate using accuracy, precision, recall, F1-score, and ROC-AUC
- Use **MLflow** to log metrics, hyperparameters, and model artifacts

#### Model Deployment
- Create a Streamlit app to input a person’s attributes and return an income prediction
- Deploy the app to Streamlit Community Cloud

### Technical Requirements
- **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `mlflow`
- **Deployment**: `streamlit`


## 🔴 Advanced Track: Deep Learning for Income Prediction
### Objectives
#### Exploratory Data Analysis
- Conduct in-depth correlation and multicollinearity analysis
- Examine feature sparsity and class imbalance
- Explore interactions like **education × hours/week** or **occupation × capital gains**

**Key Questions to Answer**:
- Can non-linear models capture interactions missed by tree-based models?
- Which features dominate prediction in different income brackets?
- Are embeddings more effective for high-cardinality variables like occupation or native country?
- Where does the model misclassify borderline cases?

#### Model Development
- Build a **Feedforward Neural Network (FFNN)** using **PyTorch** or **TensorFlow**
- Incorporate:
    - Embedding layers for high-cardinality categoricals
    - Dense layers with ReLU, dropout, and batch normalization
    - Binary cross-entropy loss and early stopping

- Use **MLflow** for tracking training metrics and architecture variations
- Evaluate using accuracy, F1-score, precision/recall, ROC-AUC
- Optionally compare with CatBoost or LightGBM baselines

#### Explainability
- Use **SHAP**, **LIME**, or **Integrated Gradients** to interpret predictions
- Visualize global and local feature contributions across the dataset

#### Model Deployment
- Deploy trained DL model in a Streamlit app
- Accept user inputs, return prediction, and display feature impact via SHAP plots
- Host on Streamlit Community Cloud or Hugging Face Spaces

### Technical Requirements
- **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Deep Learning**: `tensorflow` or `pytorch`, `mlflow`
- **Explainability**: `shap`, `lime`, `captum`
- **Deployment**: `streamlit`



## Workflow & Timeline (Both Tracks)

| Phase                     | Core Tasks                                                                | Duration      |
| ------------------------- | ------------------------------------------------------------------------- | ------------- |
| **1 · Setup + EDA**       | Set up repo, clean/transform data, visualize trends, answer EDA questions | **Week 1**    |
| **2 · Model Development** | Build and tune models, log experiments with MLflow, evaluate results      | **Weeks 2–4** |
| **3 · Deployment**        | Build and deploy Streamlit app                                            | **Week 5**    |
