import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.churn_model import train_model, build_features, test_set_scores, CUTOFF

st.set_page_config(page_title="Bank Churn Risk Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem;}
    h1 {color: #1f4e79; font-weight: 800;}
    h2, h3 {color: #1c2733;}
    [data-testid="stMetric"] {
        background: #f3f6fa; border: 1px solid #dde3ec;
        border-radius: 12px; padding: 14px 18px;
    }
    .tier-card {
        background: #f3f6fa; border-radius: 10px; padding: 14px 18px;
        margin-top: 10px; border-left: 8px solid #888;
    }
    .tier-low {border-left-color: #2e7d32;}
    .tier-medium {border-left-color: #e08a00;}
    .tier-high {border-left-color: #c62828;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return train_model()


@st.cache_data
def load_scores(_model):
    return test_set_scores(_model)


model, feature_order = load_model()
scores, actual = load_scores(model)

st.title("Bank Customer Churn Risk Dashboard")
st.caption("Gradient Boosting model. Risk tiers: Low below 0.30, Medium 0.30 to 0.60, High 0.60 and above.")

st.header("Churn Risk Calculator")

col1, col2, col3 = st.columns(3)
with col1:
    geography = st.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Years with the bank", 0, 10, 5)
with col2:
    credit_score = st.slider("Credit score", 350, 850, 650)
    balance = st.number_input("Account balance", 0.0, 300000.0, 75000.0, step=1000.0)
    salary = st.number_input("Estimated yearly salary", 1000.0, 250000.0, 100000.0, step=1000.0)
with col3:
    num_products = st.slider("Number of products", 1, 4, 2)
    has_card = st.selectbox("Has a credit card?", ["Yes", "No"])
    is_active = st.selectbox("Active member?", ["Yes", "No"])

raw = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": int(has_card == "Yes"),
    "IsActiveMember": int(is_active == "Yes"),
    "EstimatedSalary": salary,
}
features = build_features(raw, feature_order)
prob = model.predict_proba(features)[0, 1]

if prob >= 0.60:
    tier, action = "High", "Personal call from a relationship manager."
elif prob >= CUTOFF:
    tier, action = "Medium", "Send a low-cost automated retention offer."
else:
    tier, action = "Low", "No action needed."

st.subheader("Result")
r1, r2 = st.columns(2)
r1.metric("Churn probability", f"{prob:.1%}")
r2.metric("Risk tier", tier)
st.markdown(
    f"<div class='tier-card tier-{tier.lower()}'><b>{tier} risk.</b> {action}</div>",
    unsafe_allow_html=True,
)

st.header("Risk Score Distribution")
st.caption("How the model's scores spread across 2,000 customers it never saw during training.")

left, right = st.columns([3, 2])

with left:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(scores[actual == 0], bins=bins, color="#9aa5b1", alpha=0.85, label="Actually stayed")
    ax.hist(scores[actual == 1], bins=bins, color="#d1495b", alpha=0.75, label="Actually churned")
    ax.axvline(CUTOFF, color="black", linestyle="--", linewidth=1, label="Cut-off 0.30")
    ax.axvline(prob, color="#1f4e79", linewidth=2.5, label="This customer")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Number of customers")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Risk tiers (2,000 test customers)")
    tier_labels = pd.cut(scores, bins=[0, CUTOFF, 0.60, 1.001], right=False,
                         labels=["Low", "Medium", "High"])
    summary = (pd.DataFrame({"tier": tier_labels, "churned": actual})
               .groupby("tier", observed=True)
               .agg(customers=("churned", "size"), real_churners=("churned", "sum")))
    summary["churn rate (%)"] = (summary["real_churners"] / summary["customers"] * 100).round(1)
    st.table(summary)