"""
ML Service: Completion Time Prediction
--------------------------------------
A FastAPI microservice that trains and serves a regression model to estimate 
job completion duration (in days) based on SimPRO job metadata.

Key Features:
- Predicts 'completion_days' using a Random Forest Regressor
- Handles diverse features: Numeric (costs), Categorical (Customer), and Text (Description).
- Supports both Random and Temporal (cutoff date) validation splits.
"""

import os
import sys
import json
import math
import re
import traceback
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Iterable

from fastapi import FastAPI, Body
from pydantic import BaseModel

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Path Setup ---
# Ensure local modules (like transformers.py) can be imported
MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR and MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from transformers import RareCategoryCapper, Log1pTransformer

# --- Application Config ---

app = FastAPI(title="ML Service - Completion Time", version="1.0.0")

# Model Persistence Paths
DURATION_MODEL_PATH = os.environ.get("DURATION_MODEL_PATH", "model_duration.joblib")
DURATION_META_PATH = os.environ.get("DURATION_MODEL_META_PATH", "model_duration_meta.json")

# Feature Definitions
NUMERIC_COLS = [
    "revenue", "materials", "labour", "overhead", "cost_total",
    "job_age_days", "lead_time_days", "is_overdue"
]
CATEGORICAL_COLS = ["statusName", "jobType", "customerName", "siteName", "date_month", "date_dow"]
TEXT_COL = "descriptionText"
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS + [TEXT_COL]

# --- Data Models ---

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]
    test_size: Optional[float] = 0.15
    random_state: Optional[int] = 42
    max_tfidf_features: Optional[int] = 500
    rare_top_k: Optional[int] = 20
    use_text: Optional[bool] = True
    cutoff_date: Optional[str] = None

# --- Helper Functions (Type Conversion) ---

def to_num(x: Any) -> Optional[float]:
    """Robust conversion to float, handling currency strings and NaNs."""
    try:
        if x is None: return None
        if isinstance(x, (int, float)):
             return None if (isinstance(x, float) and np.isnan(x)) else float(x)
        
        if isinstance(x, str):
            s = x.strip()
            if not s: return None
            # Strip currency symbols ($ £) and separators
            s = re.sub(r'[^0-9eE\.\-+]', '', s)
            return float(s)
            
        return float(x)
    except Exception:
        return None

def to_text(value: Any, *, allow_blank: bool = False) -> Optional[str]:
    """Converts any value into a clean string suitable for categorical features."""
    if value is None:
        return "" if allow_blank else None

    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed: return trimmed
        return "" if allow_blank else None

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        # Prefer readable label keys if present
        for key in ("name", "Name", "label", "Label", "value", "Value", "description"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        # Join list items
        parts = [to_text(item, allow_blank=False) for item in value]
        parts = [p for p in parts if p]
        if parts: return " ".join(parts)
        return "" if allow_blank else None

    return str(value)

def to_bool(value: Any) -> bool:
    """Robust boolean conversion."""
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return bool(int(value) == 1)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}: return True
        if normalized in {"false", "0", "no", "n"}: return False
    return False

def to_serializable(value: Any) -> Any:
    """Prepares Python objects for JSON serialization (handles NaN, NumPy types)."""
    if value is None: return None
    
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
        
    # Handle Numpy types
    if isinstance(value, (np.floating, np.float32, np.float64)):
        val = float(value)
        return None if (math.isnan(val) or math.isinf(val)) else val
        
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
        
    if isinstance(value, (np.bool_,)):
        return bool(value)
        
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
        
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
        
    return str(value)

# --- DataSet Construction ---

def build_duration_dataset(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Transforms raw JSON records into a pandas DataFrame ready for Scikit-Learn.
    Returns: (X, y, meta_info)
    """
    rows = []
    durations = []
    is_completed_flags = []
    issued_dates = []
    job_ids = []

    for j in records:
        # 1. Feature Extraction
        revenue = to_num(j.get("revenue") or (j.get("Total", {}) or {}).get("IncTax"))
        materials = to_num(j.get("materials")) or 0.0
        labour = to_num(j.get("labour")) or 0.0
        overhead = to_num(j.get("overhead")) or 0.0

        # Cost fallback logic
        cost_total = to_num(j.get("cost_total"))
        if cost_total is None:
            cost_total = to_num(j.get("cost_est_total")) 
        if cost_total is None and (materials or labour or overhead):
            cost_total = materials + labour + overhead

        issued_date_value = (
            j.get("dateIssued") or j.get("DateIssued") or 
            j.get("issuedDate") or j.get("createdDate")
        )
        
        # 2. Construct Row
        row = {
            "revenue": revenue,
            "materials": materials if materials is not None else None,
            "labour": labour if labour is not None else None,
            "overhead": overhead if overhead is not None else None,
            "cost_total": cost_total,
            "job_age_days": to_num(j.get("job_age_days")) or 0.0,
            "lead_time_days": to_num(j.get("lead_time_days")) or 0.0,
            "is_overdue": to_num(j.get("is_overdue")) or 0.0,
            "statusName": to_text(j.get("statusName")),
            "jobType": to_text(j.get("jobType")),
            "customerName": to_text(j.get("customerName")),
            "siteName": to_text(j.get("siteName")),
            "descriptionText": to_text(j.get("descriptionText"), allow_blank=True) or "",
            "_id": j.get("ID") or j.get("id"),
            "dateIssued": issued_date_value,
        }

        # 3. Date Features (Month/Day of Week)
        if issued_date_value:
            try:
                dt = pd.to_datetime(issued_date_value)
                if pd.notnull(dt):
                    row["date_month"] = dt.strftime("%b")
                    row["date_dow"] = dt.strftime("%a")
                else:
                    row["date_month"] = "Unknown"; row["date_dow"] = "Unknown"
            except:
                row["date_month"] = "Unknown"; row["date_dow"] = "Unknown"
        else:
            row["date_month"] = "Unknown"; row["date_dow"] = "Unknown"

        # 4. Target Variable (Completion Days)
        is_completed = to_bool(j.get("is_completed"))
        completion_days = to_num(j.get("completion_days"))
        
        has_valid_target = (
            is_completed 
            and completion_days is not None 
            and float(completion_days) > 0
        )
        
        rows.append(row)
        job_ids.append(row["_id"])
        issued_dates.append(issued_date_value)
        is_completed_flags.append(is_completed)
        durations.append(float(completion_days) if has_valid_target else None)

    # Convert to Pandas
    df = pd.DataFrame(rows)
    y = pd.Series(durations, dtype="float") if rows else pd.Series(dtype="float")
    
    info = {
        "is_completed": pd.Series(is_completed_flags, dtype="bool") if is_completed_flags else pd.Series(dtype="bool"),
        "issued_dates": pd.Series(issued_dates, dtype="object") if issued_dates else pd.Series(dtype="object"),
        "job_ids": job_ids,
    }
    return df, y, info

# --- Model Pipeline Construction ---

def build_duration_pipeline(
    *,
    max_tfidf_features: int = 500,
    use_text: bool = True,
    rare_top_k: int = 20,
    estimator: Optional[Any] = None,
) -> Pipeline:
    """Constructs the Scikit-Learn preprocessing and regression pipeline."""
    
    # 1. Numeric Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", Log1pTransformer()),  # Handle skewed cost distributions
        ("scaler", RobustScaler()),   # Robust to outliers
    ])
    
    # 2. Categorical Preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ]

    # 3. Text Preprocessing
    if use_text:
        text_transformer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
        )
        transformers.append(("txt", text_transformer, TEXT_COL))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )

    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=300, random_state=42)

    # 4. Final Pipeline
    return Pipeline(steps=[
        ("rare_cap", RareCategoryCapper(columns=["customerName", "siteName"], top_k=rare_top_k)),
        ("pre", pre),
        ("reg", estimator),
    ])

def select_duration_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    *, use_text: bool, rare_top_k: int, max_tfidf_features: int,
) -> Dict[str, Any]:
    """
    Model Selection:
    Trains multiple model candidates (RF and Linear Regression) and selects the best one based on MAE.
    """
    estimators = [
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=300, random_state=42))
    ]
    # If data is very small, simple linear regression might generalize better
    if len(X_train) < 80:
        estimators.append(("LinearRegression", LinearRegression()))

    best_result = None

    for model_name, estimator in estimators:
        pipeline = build_duration_pipeline(
            max_tfidf_features=max_tfidf_features,
            use_text=use_text,
            rare_top_k=rare_top_k,
            estimator=estimator,
        )
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(math.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        
        result = {
            "model": pipeline,
            "model_name": model_name,
            "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        }
        
        if best_result is None or mae < best_result["metrics"]["MAE"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Model training failed (no candidates succeeded).")
        
    return best_result

# --- Persistence ---

def save_duration_model(model: Pipeline, meta: Dict[str, Any]) -> None:
    joblib.dump(model, DURATION_MODEL_PATH)
    with open(DURATION_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def load_duration_model() -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    if not os.path.exists(DURATION_MODEL_PATH):
        return None, None
    try:
        model = joblib.load(DURATION_MODEL_PATH)
        meta = None
        if os.path.exists(DURATION_META_PATH):
            with open(DURATION_META_PATH) as f:
                meta = json.load(f)
        return model, meta
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def build_prediction_diagnostics(
    exc: Exception,
    jobs: List[Dict[str, Any]],
    df: Optional[pd.DataFrame],
    model: Pipeline,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates detailed debug info for failed inference requests."""
    diagnostics = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "job_count": len(jobs),
        "model_steps": list(model.named_steps.keys()) if hasattr(model, "named_steps") else [],
        "expected_features": meta.get("features") if meta else list(ALL_FEATURES),
    }

    if df is not None:
        diagnostics.update({
            "dataframe_built": True,
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "received_columns": list(df.columns),
            "missing_features": [col for col in ALL_FEATURES if col not in df.columns],
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
        })
        
        # Serialize sample rows safely
        try:
            head_rows = df.head(3).to_dict(orient="records")
            diagnostics["sample_rows"] = [
                {k: to_serializable(v) for k, v in row.items()} for row in head_rows
            ]
        except Exception:
            diagnostics["sample_rows"] = "Error serializing sample rows"
    else:
        diagnostics["dataframe_built"] = False

    return diagnostics

# --- API Endpoints ---

@app.get("/", tags=["System"])
def root():
    return {
        "message": "ML Service - Completion Time Prediction",
        "endpoints": ["/health", "/predict_duration", "/train_duration", "/model/info"]
    }

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns metadata about the currently loaded active model."""
    model, meta = load_duration_model()
    if not model:
        return {"loaded": False}
    
    info = {
        "loaded": True,
        "model_name": meta.get("model_name"),
        "metrics": meta.get("metrics"),
        "features": meta.get("features"),
        "training_metadata": {
            "use_text": meta.get("use_text"),
            "rare_top_k": meta.get("rare_top_k"),
            "cutoff_date": meta.get("cutoff_date"),
        }
    }
    return info

@app.post("/train", tags=["Training"])
@app.post("/train_duration", tags=["Training"])
def train_duration(payload: TrainRequest = Body(...)):
    """
    Retrains the Duration prediction model.
    Optionally performs a temporal split if a cutoff_date is provided.
    """
    # 1. Prepare Dataset
    df, y_all, info = build_duration_dataset(payload.data)
    if df.empty:
        return {"ok": False, "error": "No data provided for training."}

    # Filter to only completed jobs (labelled data)
    labelled_mask = y_all.notna()
    X = df.loc[labelled_mask]
    y = y_all.loc[labelled_mask]

    if len(X) < 30:
        return {"ok": False, "error": f"Insufficient labelled data ({len(X)} rows). Need >= 30."}

    # 2. Validation Split Strategy
    issued_series = info.get("issued_dates")
    if isinstance(issued_series, pd.Series):
        issued_series = issued_series.loc[labelled_mask]
    
    split_strategy = "random"
    X_train, X_test, y_train, y_test = None, None, None, None

    # Temporal Split attempt
    if payload.cutoff_date:
        cutoff_ts = pd.to_datetime(payload.cutoff_date, errors="coerce")
        date_series = pd.to_datetime(issued_series, errors="coerce")
        
        if pd.notnull(cutoff_ts) and date_series.notna().any():
            split_strategy = "temporal"
            train_mask = (date_series < cutoff_ts) | date_series.isna()
            test_mask = date_series >= cutoff_ts
            
            X_train, y_train = X.loc[train_mask], y.loc[train_mask]
            X_test, y_test = X.loc[test_mask], y.loc[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                return {
                    "ok": False, 
                    "error": "Temporal split resulted in empty train or test set. Adjust cutoff date."
                }
        else:
            return {"ok": False, "error": f"Invalid cutoff date: {payload.cutoff_date}"}

    # Fallback to Random Split
    if split_strategy == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=payload.test_size or 0.15, 
            random_state=payload.random_state
        )

    # 3. Model Training & Selection
    try:
        # Default config
        use_text = True if payload.use_text is None else payload.use_text
        max_tfidf = payload.max_tfidf_features or 500
        rare_top = payload.rare_top_k or 20

        try:
            selection = select_duration_model(
                X_train, y_train, X_test, y_test,
                use_text=use_text, rare_top_k=rare_top, max_tfidf_features=max_tfidf
            )
        except ValueError as exc:
            # Retry without text if vocab is empty (common with small datasets)
            if use_text and "empty vocabulary" in str(exc).lower():
                selection = select_duration_model(
                    X_train, y_train, X_test, y_test,
                    use_text=False, rare_top_k=rare_top, max_tfidf_features=max_tfidf
                )
                use_text = False
            else:
                raise exc

        # Refit best model on full dataset
        final_model = selection["model"]
        final_model.fit(X, y)

        # 4. Save Artifacts
        meta = {
            "model_name": selection["model_name"],
            "metrics": selection["metrics"],
            "features": NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if use_text else []),
            "use_text": use_text,
            "cutoff_date": payload.cutoff_date if split_strategy == "temporal" else None,
            "split_strategy": split_strategy,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_samples": len(X),
        }
        
        save_duration_model(final_model, meta)
        
        return {
            "ok": True,
            "model_name": selection["model_name"],
            "metrics": selection["metrics"],
            "metadata": meta
        }

    except Exception as exc:
        return {"ok": False, "error": f"Training process failed: {str(exc)}"}

@app.post("/predict_duration", tags=["Inference"])
def predict_duration(payload: PredictRequest = Body(...)):
    """
    Generates completion time estimates (days) for a list of jobs.
    Returns: { "predictions": [ { "jobId": ..., "predicted_completion_days": ... } ] }
    """
    jobs = payload.data or []
    model, meta = load_duration_model()

    if not model:
        return {
            "predictions": [],
            "count": 0,
            "model_loaded": False,
            "error": "Model not trained. Please call /train first."
        }

    try:
        # 1. Build DataFrame
        df, _, info = build_duration_dataset(jobs)
        if df.empty:
            return {"predictions": [], "count": 0, "model_loaded": True}

        # 2. Run Inference
        preds = model.predict(df)
        
        # 3. Format Response
        results = []
        is_completed_series = info.get("is_completed", pd.Series())
        job_ids = info.get("job_ids", [])

        for idx, job in enumerate(jobs):
            job_id = job_ids[idx] if idx < len(job_ids) else (job.get("ID") or job.get("id"))
            
            # Skip jobs that are already finished
            if idx < len(is_completed_series) and is_completed_series.iloc[idx]:
                results.append({
                    "jobId": job_id, 
                    "skipped": True, 
                    "reason": "already_completed"
                })
            else:
                results.append({
                    "jobId": job_id,
                    "predicted_completion_days": float(preds[idx])
                })

        return {
            "predictions": results,
            "count": len(results),
            "model_loaded": True,
            "metrics": meta.get("metrics") if meta else None
        }

    except Exception as exc:
        return {
            "predictions": [],
            "error": f"Prediction failed: {str(exc)}",
            "diagnostics": build_prediction_diagnostics(exc, jobs, df, model, meta)
        }
