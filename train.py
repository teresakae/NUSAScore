"""
Requirements:
    pandas scikit-learn xgboost joblib

Assumptions:
    This script is in the same directory as 'credit.data'.
Outputs:
    - nusa_score_model.joblib  (trained classifier)
    - nusa_score_meta.joblib   (dict with training columns & dtypes)
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# -----------------------------
# 1) Load data (no headers)
# -----------------------------
CSV_PATH = Path("credit.data")
if not CSV_PATH.exists():
    raise FileNotFoundError("credit.data not found. Place it in the same folder as this script.")

colnames = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "telephone",
    "foreign_worker", "credit_risk"
]

# Note: credit.data is space-delimited with no header
df = pd.read_csv("credit.data", header=None, delim_whitespace=True, names=colnames)

# -----------------------------
# 2) Target transform & feature split
# -----------------------------
# Convert target: 1 (Good) -> 0 ; 2 (Bad) -> 1
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1}).astype(int)

y = df["credit_risk"]
X_raw = df.drop(columns=["credit_risk"])

# Detect categorical columns (object dtype) and numeric columns
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [c for c in X_raw.columns if c not in categorical_cols]

# -----------------------------
# 3) Train/Test split
# -----------------------------
X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------
# 4) One-hot encoding (replicated exactly in API)
# -----------------------------
X_train = pd.get_dummies(X_raw_train, columns=categorical_cols, drop_first=False)
training_columns = X_train.columns.tolist()

X_test = pd.get_dummies(X_raw_test, columns=categorical_cols, drop_first=False)
X_test = X_test.reindex(columns=training_columns, fill_value=0)

# -----------------------------
# 5) Train model
# -----------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# -----------------------------
# 6) Evaluate
# -----------------------------
y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print("Classification report on held-out test set:\n")
print(classification_report(y_test, y_pred, digits=4))

# -----------------------------
# 7) Persist model & metadata
# -----------------------------
joblib.dump(model, "nusa_score_model.joblib")

meta = {
    "training_columns": training_columns,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
}
joblib.dump(meta, "nusa_score_meta.joblib")

print("\nSaved:")
print(" - nusa_score_model.joblib")
print(" - nusa_score_meta.joblib")
