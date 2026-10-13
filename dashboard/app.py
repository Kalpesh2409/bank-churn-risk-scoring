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

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
})

FRIENDLY = {
    "CreditScore": "Credit score",
    "Age": "Age",
    "Tenure": "Years with the bank",
    "Balance": "Account balance",
    "NumOfProducts": "Number of products",
    "HasCrCard": "Has a credit card",
    "IsActiveMember": "Active member",
    "EstimatedSalary": "Estimated salary",
    "Geography_Germany": "Country: Germany",
    "Geography_Spain": "Country: Spain",
    "Gender_Male": "Gender: male",
    "BalanceSalaryRatio_log": "Balance-to-salary ratio",
    "ProductDensity": "Products per year",
    "EngagementProduct": "Active products",
    "TenureByAge": "Share of life with the bank",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp input, .stApp table,
    .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stCaptionContainer"], .stApp [data-baseweb="select"] div {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    }

    .block-container {max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem;}

    .hero {
        background: linear-gradient(135deg, #1f4e79 0%, #2f7bb6 100%);
        color: #ffffff; padding: 28px 32px; border-radius: 16px; margin-bottom: 1.5rem;
    }
    .hero-title {font-size: 2rem; font-weight: 800; letter-spacing: -0.5px;}
    .hero-sub {font-size: 1rem; opacity: 0.92; margin-top: 6px;}

    h2 {font-size: 1.55rem !important; font-weight: 700 !important; color: #1c2733;
        border-bottom: 2px solid #e3e8ef; padding-bottom: 0.4rem; margin-top: 1.5rem;}
    h3 {font-size: 1.25rem !important; font-weight: 600 !important; color: #1c2733;}
    h4 {font-size: 1.05rem !important; font-weight: 600 !important; color: #1c2733;}

    [data-testid="stCaptionContainer"] {font-size: 0.95rem; color: #5b6776;}
    [data-testid="stMetric"] {
        background: #f3f6fa; border: 1px solid #dde3ec; border-radius: 12px;
        padding: 14px 18px; box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    }
    [data-testid="stMetricLabel"] {font-size: 0.95rem; color: #5b6776;}
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: 700; color: #1f4e79;}

    .tier-card {
        background: #f3f6fa; border-radius: 10px; padding: 14px 18px;
        margin-top: 10px; border-left: 8px solid #888; font-size: 1rem;
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


def risk_tier(p):
    if p >= 0.60:
        return "High", "Personal call from a relationship manager."
    if p >= CUTOFF:
        return "Medium", "Send a low-cost automated retention offer."
    return "Low", "No action needed."


model, feature_order = load_model()
scores, actual = load_scores(model)

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">Bank Customer Churn Risk Dashboard</div>
      <div class="hero-sub">Gradient Boosting model &middot; Risk tiers: Low below 0.30
      &middot; Medium 0.30 to 0.60 &middot; High 0.60 and above</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
tier, action = risk_tier(prob)

st.subheader("Result")
r1, r2 = st.columns(2)
r1.metric("Churn probability", f"{prob:.1%}")
r2.metric("Risk tier", tier)
st.markdown(
    f"<div class='tier-card tier-{tier.lower()}'><b>{tier} risk.</b> {action}</div>",
    unsafe_allow_html=True,
)

st.header("What-If Scenario Simulator")
st.caption(
    "Change the engagement or product values to see how this customer's risk score moves. "
    "All other details stay as entered above. This shows how the model's score changes, "
    "not proof that the change would make the customer stay."
)

w1, w2 = st.columns(2)
with w1:
    new_products = st.slider("Scenario: number of products", 1, 4, int(num_products))
with w2:
    new_active = st.selectbox(
        "Scenario: active member?", ["Yes", "No"],
        index=0 if is_active == "Yes" else 1,
    )

scenario = dict(raw)
scenario["NumOfProducts"] = new_products
scenario["IsActiveMember"] = int(new_active == "Yes")
prob2 = model.predict_proba(build_features(scenario, feature_order))[0, 1]
tier2, action2 = risk_tier(prob2)
change = (prob2 - prob) * 100

s1, s2, s3 = st.columns(3)
s1.metric("Current churn probability", f"{prob:.1%}")
delta_text = None if abs(change) < 0.05 else f"{change:+.1f} points"
s2.metric("Scenario churn probability", f"{prob2:.1%}",
          delta=delta_text, delta_color="inverse")
s3.metric("Scenario risk tier", tier2)

if tier2 != tier:
    st.write(f"In this scenario the tier changes from **{tier}** to **{tier2}**. {action2}")
else:
    st.write(f"In this scenario the tier stays **{tier}**.")

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
    st.markdown("#### Risk tiers (2,000 test customers)")
    tier_labels = pd.cut(scores, bins=[0, CUTOFF, 0.60, 1.001], right=False,
                         labels=["Low", "Medium", "High"])
    summary = (pd.DataFrame({"tier": tier_labels, "churned": actual})
               .groupby("tier", observed=True)
               .agg(customers=("churned", "size"), real_churners=("churned", "sum")))
    summary = summary.rename(columns={"customers": "Customers",
                                      "real_churners": "Real churners"})
    summary["Churn rate (%)"] = (
        summary["Real churners"] / summary["Customers"] * 100
    ).map("{:.1f}".format)
    summary.index.name = "Tier"
    st.table(summary)

st.header("What Drives Churn")

importance = pd.Series(model.feature_importances_, index=feature_order)
top = importance.sort_values(ascending=False)

fl, fr = st.columns([3, 2])

with fl:
    fig2, ax2 = plt.subplots(figsize=(7, 4.6))
    shown = importance.sort_values()
    ax2.barh([FRIENDLY[c] for c in shown.index], shown.values, color="#1f4e79")
    ax2.set_xlabel("Share of the model's decisions")
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)
    plt.close(fig2)

with fr:
    st.markdown("#### How to read this")
    st.write(
        f"**{FRIENDLY[top.index[0]]}** and **{FRIENDLY[top.index[1]]}** together account for "
        f"**{top.iloc[:2].sum():.0%}** of the model's decisions."
    )
    st.write(
        "This chart shows how much each factor matters, not which direction it pushes. "
        "In our analysis, risk is highest for customers aged about 45 to 65, "
        "for those holding 3 or 4 products, for customers in Germany, "
        "and for inactive customers."
    )
