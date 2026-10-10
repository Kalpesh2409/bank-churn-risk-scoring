from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "bank_churn_features.csv"
CUTOFF = 0.30


def train_model():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist()

def build_features(raw, feature_order):
    balance = raw["Balance"]
    salary = raw["EstimatedSalary"]

    row = {
        "CreditScore": raw["CreditScore"],
        "Age": raw["Age"],
        "Tenure": raw["Tenure"],
        "Balance": balance,
        "NumOfProducts": raw["NumOfProducts"],
        "HasCrCard": raw["HasCrCard"],
        "IsActiveMember": raw["IsActiveMember"],
        "EstimatedSalary": salary,
        "Geography_Germany": int(raw["Geography"] == "Germany"),
        "Geography_Spain": int(raw["Geography"] == "Spain"),
        "Gender_Male": int(raw["Gender"] == "Male"),
        "BalanceSalaryRatio_log": np.log1p(balance / salary),
        "ProductDensity": raw["NumOfProducts"] / (raw["Tenure"] + 1),
        "EngagementProduct": raw["IsActiveMember"] * raw["NumOfProducts"],
        "TenureByAge": raw["Tenure"] / raw["Age"],
    }
    return pd.DataFrame([row])[feature_order]