"""
FastAPI service that loads the trained model (German Credit) and serves predictions.

Run:
    uvicorn main:app --reload

Requirements:
    fastapi uvicorn pandas scikit-learn xgboost joblib pydantic

POST /predict
Body must include the 20 original input fields (all features except 'credit_risk').
Categorical fields expect the original code values from the German Credit docs (e.g., A11, A30, A40...).
"""
from pathlib import Path
from typing import Literal, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import UploadFile, File

MODEL_PATH = Path("nusa_score_model.joblib")
META_PATH = Path("nusa_score_meta.joblib")

app = FastAPI(title="NUSA Score API", version="1.0.0")

# --------- Pydantic data models ---------

class ApplicationData(BaseModel):
    # Codes follow the original dataset documentation (examples in comments)
    checking_status: str = Field(..., description="A11/A12/A13/A14")
    duration: int = Field(..., description="Months (e.g., 24)")
    credit_history: str = Field(..., description="A30/A31/A32/A33/A34")
    purpose: str = Field(..., description="A40/A41/A42/A43/A44/A45/A46/A48/A49/A410")
    credit_amount: int = Field(..., description="Loan amount (e.g., 5000)")
    savings_status: str = Field(..., description="A61/A62/A63/A64/A65")
    employment_since: str = Field(..., description="A71/A72/A73/A74/A75")
    installment_rate: int = Field(..., description="1-4 (as % of income)")
    personal_status: str = Field(..., description="A91/A92/A93/A94/A95")
    other_debtors: str = Field(..., description="A101/A102/A103")
    residence_since: int = Field(..., description="Years at residence")
    property: str = Field(..., description="A121/A122/A123/A124")
    age: int = Field(..., description="Age in years")
    other_installment_plans: str = Field(..., description="A141/A142/A143")
    housing: str = Field(..., description="A151/A152/A153")
    existing_credits: int = Field(..., description="Number of existing credits")
    job: str = Field(..., description="A171/A172/A173/A174")
    num_dependents: int = Field(..., description="Number of dependents")
    telephone: str = Field(..., description="A191/A192")
    foreign_worker: str = Field(..., description="A201/A202")

class PredictionResult(BaseModel):
    risk_probability: float
    suggested_decision: Literal["APPROVE", "FLAG FOR REVIEW", "DECLINE"]

# --------- Load model & metadata on startup ---------

model = None
meta = None

@app.on_event("startup")
def _load_artifacts():
    global model, meta
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Artifacts missing. Run 'train.py' first.")
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    if "training_columns" not in meta or "categorical_cols" not in meta:
        raise RuntimeError("Metadata incomplete.")

# --------- Preprocessing (must mirror training) ---------

def preprocess_for_inference(payload: ApplicationData) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "checking_status": payload.checking_status,
        "duration": payload.duration,
        "credit_history": payload.credit_history,
        "purpose": payload.purpose,
        "credit_amount": payload.credit_amount,
        "savings_status": payload.savings_status,
        "employment_since": payload.employment_since,
        "installment_rate": payload.installment_rate,
        "personal_status": payload.personal_status,
        "other_debtors": payload.other_debtors,
        "residence_since": payload.residence_since,
        "property": payload.property,
        "age": payload.age,
        "other_installment_plans": payload.other_installment_plans,
        "housing": payload.housing,
        "existing_credits": payload.existing_credits,
        "job": payload.job,
        "num_dependents": payload.num_dependents,
        "telephone": payload.telephone,
        "foreign_worker": payload.foreign_worker,
    }])

    # One-hot encode the same categorical columns used during training
    X = pd.get_dummies(raw, columns=meta["categorical_cols"], drop_first=False)

    # Align to the exact training feature space
    X = X.reindex(columns=meta["training_columns"], fill_value=0)

    # Defensive numeric casting for numeric columns
    for col in meta["numeric_cols"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X

def decision_from_probability(p: float) -> str:
    if p < 0.25:
        return "APPROVE"
    elif p < 0.50:
        return "FLAG FOR REVIEW"
    else:
        return "DECLINE"

# --------- Routes ---------

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    import io

    # Read Excel file into DataFrame
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    # Preprocess each row
    try:
        X = pd.get_dummies(df, columns=meta["categorical_cols"], drop_first=False)
        X = X.reindex(columns=meta["training_columns"], fill_value=0)

        # Predictions
        probas = model.predict_proba(X)[:, 1]
        decisions = [decision_from_probability(p) for p in probas]

        results = pd.DataFrame({
            "risk_probability": probas,
            "suggested_decision": decisions
        })

        return results.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {e}")

@app.get("/")
def root():
    return {"service": "NUSA Score API", "status": "ok"}

@app.post("/predict", response_model=PredictionResult)
def predict(data: ApplicationData):
    if model is None or meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        X = preprocess_for_inference(data)
        proba = float(model.predict_proba(X)[:, 1][0])  # probability of *bad* risk
        return PredictionResult(
            risk_probability=proba,
            suggested_decision=decision_from_probability(proba),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
