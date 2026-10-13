# Bank Customer Churn: Predictive Modeling and Risk Scoring

A machine learning project that predicts which bank customers are likely to leave,
gives each one a churn probability (0 to 1) and a risk tier, and explains why.

**Live dashboard:** https://kalpesh2409-bank-churn-risk-scoring-dashboardapp-kc1urt.streamlit.app/


## Problem
The bank knows a customer has left only after they leave. This project builds a model
that tells in advance which customers may leave, so the bank can reach out early.

## Data
10,000 European bank customers, of whom 20.4% churned. Columns include credit score,
country, gender, age, tenure, balance, number of products, credit card, active member
status and estimated salary.

## Approach
1. Data cleaning and validation
2. Feature engineering (four business features, such as balance-to-salary ratio)
3. Stratified 80/20 train-test split, with scaling fitted on training data only
4. Models compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
5. Cut-off chosen from cross-validated training predictions
6. Explainability: feature importance, SHAP, partial dependence
7. Streamlit dashboard

## Key Results
- Best model: Gradient Boosting, ROC-AUC 0.867 on 2,000 unseen customers
- Lowering the cut-off from 0.5 to 0.30 raised churn recall from 0.484 to 0.654
  (precision 0.623)
- Risk tiers (Low below 0.30, Medium 0.30 to 0.60, High 0.60 and above) had real
  churn rates of 9.0%, 43.5% and 84.6%
- Age and number of products together account for 62% of the model's decisions
- The findings are associations, not proof of cause. Some churners still look low risk
  without behavioural data

## Dashboard
- Churn risk calculator for a single customer
- What-if simulator (change products or activity and see the score move)
- Risk score distribution with a tier table
- Feature importance chart

## Project Structure
```
data/raw/            original dataset
data/processed/      cleaned data and engineered features
notebooks/           01 cleaning, 02 features, 03 baseline, 04 tree models,
                     05 advanced models, 06 explainability, 07 EDA
src/churn_model.py   shared model and feature code
dashboard/app.py     Streamlit app
```

## Run Locally
```
pip install -r requirements.txt
streamlit run dashboard/app.py
```
