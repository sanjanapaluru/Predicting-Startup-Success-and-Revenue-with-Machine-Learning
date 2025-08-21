from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat
from typing import Dict, Any
import joblib, numpy as np, json, os
import pandas as pd
import uvicorn

app = FastAPI(title="Startup ML Service")

# -------------------------------------------------------------------
# Model registry
# -------------------------------------------------------------------
REG_PATH = "models/registry.json"
if not os.path.exists(REG_PATH):
    with open(REG_PATH, "w") as f:
        f.write('{"success_clf":{"path":"models/success_clf_latest.pkl"},"revenue_reg":{"path":"models/revenue_reg_latest.pkl"}}')

with open(REG_PATH) as f:
    reg = json.load(f)

def _safe_load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Please place your trained pipeline there.")
    return joblib.load(path)

_clf = None
_reg = None

def clf():
    global _clf
    if _clf is None:
        _clf = _safe_load(reg["success_clf"]["path"])
    return _clf

def regressor():
    global _reg
    if _reg is None:
        _reg = _safe_load(reg["revenue_reg"]["path"])
    return _reg

# -------------------------------------------------------------------
# Load feature schema exported from notebook
# -------------------------------------------------------------------
SCHEMA_PATH = "models/feature_schema.json"
if not os.path.exists(SCHEMA_PATH):
    # We keep running, but will error on predict if schema is required.
    FEATURE_SCHEMA = None
else:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        FEATURE_SCHEMA = json.load(f)

def prepare_df(input_row: Dict[str, Any]) -> pd.DataFrame:
    """
    Align incoming JSON to the exact columns used during training:
    - Add missing columns with defaults from schema
    - Drop unknown columns
    - Order columns to match training order
    """
    if FEATURE_SCHEMA is None:
        raise FileNotFoundError(
            "feature_schema.json not found in models/. "
            "Export it from your notebook so the API can align inputs."
        )

    cols = FEATURE_SCHEMA["columns"]
    ordered_names = [c["name"] for c in cols]
    defaults = {c["name"]: c["default"] for c in cols}
    types = {c["name"]: c["type"] for c in cols}

    # Start from defaults, then overlay provided values
    row_aligned = defaults.copy()
    for k, v in input_row.items():
        if k in row_aligned:
            row_aligned[k] = v  # accept provided value if it's a known feature

    # Build DataFrame in the exact training order
    df = pd.DataFrame([[row_aligned.get(n) for n in ordered_names]], columns=ordered_names)

    # Cast types (best effort)
    for n in ordered_names:
        if types[n] == "number":
            try:
                df[n] = pd.to_numeric(df[n])
            except Exception:
                df[n] = pd.to_numeric(df[n], errors="coerce").fillna(defaults[n])
        else:
            df[n] = df[n].astype(str)

    return df

# -------------------------------------------------------------------
# Optional small schema for request validation (not all fields shown)
# You can keep this minimal since we align to the full schema anyway.
# -------------------------------------------------------------------
class MinimalFeatures(BaseModel):
    Funding_Amount: confloat(ge=0) | None = None
    Employees_Count: conint(ge=1) | None = None
    Burn_Rate: confloat(ge=0) | None = None
    Customer_Retention_Rate: confloat(ge=0, le=1) | None = None
    Marketing_Expense: confloat(ge=0) | None = None
    # You can POST any extra keys; they'll be used if present in schema.

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/success")
def predict_success(body: Dict[str, Any]):
    try:
        df = prepare_df(body)   # align to training columns
        model = clf()
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(df)[0, 1])
            label = int(proba >= 0.5)
            return {"success_probability": proba, "predicted_success": label}
        label = int(model.predict(df)[0])
        return {"predicted_success": label}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}

@app.post("/predict/revenue")
def predict_revenue(body: Dict[str, Any]):
    try:
        df = prepare_df(body)
        model = regressor()
        yhat = float(model.predict(df)[0])
        return {"predicted_revenue": yhat}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
