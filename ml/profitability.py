"""
ML Service: Profitability Classification
----------------------------------------
A FastAPI microservice that trains and serves a classification model to predict
job profitability categories (Low, Medium, High).

Key Features:
- Classification using Logistic Regression (with optional Calibration).
- Hyperparameter tuning via Cross-Validation.
- Supports fallback rules if the ML model is not available.
- Handles diverse features: Financials, Job Type, Description (NLP).
- Supports Temporal splitting for time-series validation.
"""

import os
import sys
import json
import math
import re
import traceback
import inspect
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV


# --- Path Setup ---
MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR and MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from transformers import RareCategoryCapper, Log1pTransformer

# --- Application Config ---

app = FastAPI(title="ML Service - Profitability", version="1.0.0")

# Model Persistence Paths
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
META_PATH = os.environ.get("MODEL_META_PATH", "model_meta.json")

# Feature Definitions
NUMERIC_COLS = [
    "revenue", "materials", "labour", "overhead", "cost_total",
    "job_age_days", "lead_time_days", "is_overdue"
]
CATEGORICAL_COLS = ["statusName", "jobType", "customerName", "siteName", "date_month", "date_dow"]
TEXT_COL = "descriptionText"
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS + [TEXT_COL]

# Labels
CLASS_LABELS = ["Low", "Medium", "High"]
DEFAULT_THRESHOLDS = {"low": 0.44, "high": 0.64}

# --- Data Models ---

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class Thresholds(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None

class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]
    test_size: Optional[float] = 0.15
    random_state: Optional[int] = 42
    max_tfidf_features: Optional[int] = 500
    rare_top_k: Optional[int] = 20
    use_text: Optional[bool] = True
    calibrate: Optional[bool] = False
    cutoff_date: Optional[str] = None
    thresholds: Optional[Thresholds] = None

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
            # Strip currency and separators
            s = re.sub(r'[^0-9eE\.\-+]', '', s)
            return float(s)
        return float(x)
    except Exception:
        return None

def to_text(value: Any, *, allow_blank: bool = False) -> Optional[str]:
    """Coerces arbitrary values to a clean string suitable for categorical/text features."""
    if value is None:
        return "" if allow_blank else None

    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed: return trimmed
        return "" if allow_blank else None

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        for key in ("name", "Name", "label", "Label", "value", "Value", "description"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        parts = [to_text(item, allow_blank=False) for item in value]
        parts = [p for p in parts if p]
        if parts: return " ".join(parts)
        return "" if allow_blank else None

    return str(value)

def to_serializable(value: Any) -> Any:
    """Prepares Python objects for JSON serialization (handles NaN, NumPy types)."""
    if value is None: return None
    
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
        
    if isinstance(value, (np.floating, np.float32, np.float64)):
        val = float(value)
        return None if (math.isnan(val) or math.isinf(val)) else val
        
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
        
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
        
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
        
    return str(value)

def resolve_thresholds(thresholds: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Merges user-provided thresholds with defaults."""
    resolved = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        if thresholds.get("low") is not None:
            resolved["low"] = float(thresholds["low"])
        if thresholds.get("high") is not None:
            resolved["high"] = float(thresholds["high"])
    return resolved

# --- Logic: Fallback & Heuristics ---

def fallback_classification(j: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic-based classification when no ML model is loaded.
    Uses simple margin thresholds.
    """
    revenue = to_num(j.get("revenue"))
    materials = to_num(j.get("materials")) or to_num(j.get("materials_cost_est")) or 0.0
    labour = to_num(j.get("labour")) or to_num(j.get("labor_cost_est")) or 0.0
    overhead = to_num(j.get("overhead")) or to_num(j.get("overhead_est")) or 0.0
    
    cost_total = to_num(j.get("cost_total")) or to_num(j.get("cost_est_total"))
    if cost_total is None:
        cost_total = materials + labour + overhead

    profit_est = j.get("profit_est")
    if profit_est is not None:
        profit_est = to_num(profit_est)
    elif revenue is not None and cost_total is not None:
        profit_est = revenue - cost_total
    
    margin_est = None
    if profit_est is not None and revenue and revenue > 0:
        margin_est = profit_est / revenue

    # Decision logic
    profitable = False
    if profit_est is not None:
         profitable = profit_est > 0
    elif margin_est is not None:
         profitable = margin_est > 0.10

    # Probability fallback (Sigmoid-ish scaler)
    if margin_est is None:
        prob = 0.5 if profitable else 0.3
    else:
        prob = 1 / (1 + math.exp(-6 * (margin_est)))
        prob = float(np.clip(prob, 0.01, 0.99))

    # Class assignment based on default rule-of-thumb
    klass = "Unknown"
    if margin_est is not None:
        if margin_est >= 0.20: klass = "High"
        elif margin_est >= 0.05: klass = "Medium"
        else: klass = "Low"

    return {
        "jobId": j.get("ID") or j.get("id"),
        "class": klass,
        "profitable": bool(profitable),
        "probability": prob,
        "profit_est": profit_est,
        "margin_est": margin_est
    }

def derive_label(row: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[str]:
    """Generates a ground-truth label based on realized financial performance."""
    p = to_num(row.get("netMarginPct"))
    if p is None:
        rev = to_num(row.get("revenue"))
        cost = to_num(row.get("cost_total"))
        if rev is not None and cost is not None and rev > 0:
            p = (rev - cost) / rev
            
    if p is None: return None
    
    high = thresholds.get("high", DEFAULT_THRESHOLDS["high"])
    low = thresholds.get("low", DEFAULT_THRESHOLDS["low"])
    
    if p > high: return "High"
    if p >= low: return "Medium"
    return "Low"

# --- DataSet Construction ---

def build_dataframe(
    records: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Transforms raw JSON records into a pandas DataFrame (X) and Label Series (y).
    """
    rows = []
    labels = []
    resolved_thresholds = resolve_thresholds(thresholds)

    for j in records:
        # 1. Financials
        revenue = to_num(j.get("revenue") or (j.get("Total", {}) or {}).get("IncTax"))
        materials = to_num(j.get("materials")) or 0.0
        labour = to_num(j.get("labour")) or 0.0
        overhead = to_num(j.get("overhead")) or 0.0
        
        cost_total = to_num(j.get("cost_total"))
        if cost_total is None: cost_total = to_num(j.get("cost_est_total"))
        if cost_total is None and (materials or labour or overhead):
            cost_total = materials + labour + overhead

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
            "dateIssued": j.get("dateIssued") or j.get("DateIssued"),
        }

        # 3. Date Features
        dval = row.get("dateIssued")
        if dval:
            try:
                dt = pd.to_datetime(dval)
                if pd.notnull(dt):
                    row["date_month"] = dt.strftime("%b")
                    row["date_dow"] = dt.strftime("%a")
                else:
                    row["date_month"] = "Unknown"; row["date_dow"] = "Unknown"
            except:
                row["date_month"] = "Unknown"; row["date_dow"] = "Unknown"
        else:
             row["date_month"] = "Unknown"; row["date_dow"] = "Unknown"

        rows.append(row)

        # 4. Labels
        lbl = j.get("profitability_class")
        if lbl is None:
            lbl = derive_label(row, resolved_thresholds)
        labels.append(lbl)

    df = pd.DataFrame(rows)
    if df.empty: return df, None

    # Only return y if we actually found labels
    if any(l is not None for l in labels):
        y = pd.Series(labels, dtype="object")
        return df, y
        
    return df, None

# --- Model Pipeline Construction ---

def build_calibrated_classifier(base, *, method: str = "sigmoid", cv: int = 3) -> CalibratedClassifierCV:
    """Wrapper that supports different sklearn versions for CalibratedClassifierCV."""
    params = inspect.signature(CalibratedClassifierCV.__init__).parameters
    kwargs = {"method": method, "cv": cv}
    if "estimator" in params:
        kwargs["estimator"] = base
    else:
        kwargs["base_estimator"] = base
    return CalibratedClassifierCV(**kwargs)

def build_pipeline(
    max_tfidf_features: int = 500,
    use_text: bool = True,
    rare_top_k: int = 20,
    C: float = 1.0,
) -> Pipeline:
    """Constructs the Classification Pipeline (LR)."""
    
    # 1. Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", Log1pTransformer()),
        ("scaler", RobustScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ]
    
    if use_text:
        text_transformer = TfidfVectorizer(max_features=max_tfidf_features, ngram_range=(1, 2))
        transformers.append(("txt", text_transformer, TEXT_COL))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    
    # 2. Estimator
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        multi_class="multinomial",
        C=C,
    )
    
    # 3. Pipeline
    return Pipeline(steps=[
        ("rare_cap", RareCategoryCapper(columns=["customerName", "siteName"], top_k=rare_top_k)),
        ("pre", pre),
        ("clf", lr),
    ])

# --- Persistence ---

def save_model(model: Any, meta: Dict[str, Any]) -> None:
    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def load_model() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        meta = None
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
        return model, meta
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def build_prediction_diagnostics(
    exc: Exception,
    jobs: List[Dict[str, Any]],
    df: Optional[pd.DataFrame],
    model: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates detailed debug info."""
    model_steps = []
    if hasattr(model, "named_steps"):
        model_steps = list(model.named_steps.keys())
    elif hasattr(model, "base_estimator") and hasattr(model.base_estimator, "named_steps"):
        model_steps = list(model.base_estimator.named_steps.keys())
    elif hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        model_steps = list(model.estimator.named_steps.keys())

    diagnostics = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "job_count": len(jobs),
        "model_steps": model_steps,
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
        try:
            head_rows = df.head(3).to_dict(orient="records")
            diagnostics["sample_rows"] = [
                {k: to_serializable(v) for k, v in row.items()} for row in head_rows
            ]
        except Exception:
            diagnostics["sample_rows"] = "Serialization error"
    else:
        diagnostics["dataframe_built"] = False
        
    return diagnostics

# --- API Endpoints ---

@app.get("/", tags=["System"])
def root():
    return {
        "message": "ML Service - Profitability Prediction",
        "endpoints": ["/health", "/predict", "/train", "/model/info"]
    }

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns metadata about the currently loaded model."""
    model, meta = load_model()
    if not model:
        return {"loaded": False}
    return {
        "loaded": True,
        "metrics": {
            "f1_macro": meta.get("f1_macro"),
            "chosen_C": meta.get("chosen_C"),
        },
        "config": {
            "use_text": meta.get("use_text"),
            "calibrated": meta.get("calibrated"),
            "thresholds": meta.get("thresholds"),
            "cutoff_date": meta.get("cutoff_date"),
        },
        "labels": meta.get("labels"),
        "features": meta.get("features"),
        "meta": meta,
    }

@app.post("/train", tags=["Training"])
def train(payload: TrainRequest = Body(...)):
    """
    Retrains the classification model using Logistic Regression.
    Performs hyperparameter tuning and Cross Validation.
    """
    # 1. Config & Data
    thresholds = resolve_thresholds(
        payload.thresholds.dict(exclude_none=True) if payload.thresholds else None
    )
    df, y_all = build_dataframe(payload.data, thresholds)

    if df.empty:
        return {"ok": False, "error": "No usable data rows."}
    if y_all is None:
        return {"ok": False, "error": "No labels found (missing netMarginPct/profitability_class)."}

    # 2. Filtering
    mask = y_all.notna()
    X = df.loc[mask]
    y = y_all.loc[mask]

    # Keep only known classes
    keep_mask = y.isin(CLASS_LABELS)
    X = X.loc[keep_mask]
    y = y.loc[keep_mask]

    if len(X) < 12 or y.nunique() < 2:
        return {"ok": False, "error": f"Insufficient data ({len(X)} rows, {y.nunique()} classes). Need >= 12 rows, 2 classes."}

    # 3. Validation Split
    split_strategy = "random"
    cutoff_date = payload.cutoff_date
    X_train, X_test, y_train, y_test = None, None, None, None

    if cutoff_date and "dateIssued" in X.columns and X["dateIssued"].notna().any():
        split_strategy = "temporal"
        date_series = pd.to_datetime(X["dateIssued"], errors="coerce")
        cutoff_ts = pd.to_datetime(cutoff_date, errors="coerce")
        
        if pd.notnull(cutoff_ts):
            train_mask = (date_series < cutoff_ts) | date_series.isna()
            test_mask = date_series >= cutoff_ts
            X_train, y_train = X.loc[train_mask], y.loc[train_mask]
            X_test, y_test = X.loc[test_mask], y.loc[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                 return {"ok": False, "error": "Temporal split resulted in empty sets. Check cutoff_date."}
        else:
             return {"ok": False, "error": f"Invalid cutoff_date: {cutoff_date}"}
    else:
        # Stratified Random Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=payload.test_size or 0.15,
            random_state=payload.random_state, stratify=y
        )

    if y_train.value_counts().min() < 3:
        return {"ok": False, "error": "Training set has < 3 samples for some classes. Cannot perform CV."}

    # 4. Hyperparameter Tuning (Grid Search manually for C)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=payload.random_state)
    candidate_cs = [0.25, 0.5, 1.0, 2.0]
    best_c = 1.0
    best_score = -np.inf
    cv_scores = {}

    max_tfidf = payload.max_tfidf_features or 500
    use_text = True if payload.use_text is None else payload.use_text
    rare_top = payload.rare_top_k or 20

    for c in candidate_cs:
        pipe = build_pipeline(max_tfidf_features=max_tfidf, use_text=use_text, rare_top_k=rare_top, C=c)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        mean_score = float(np.mean(scores))
        cv_scores[str(c)] = [float(s) for s in scores]
        
        if mean_score > best_score:
            best_score = mean_score
            best_c = c

    # 5. Final Model Training
    base_model = build_pipeline(max_tfidf_features=max_tfidf, use_text=use_text, rare_top_k=rare_top, C=best_c)
    
    if payload.calibrate:
        final_model = build_calibrated_classifier(base_model)
    else:
        final_model = base_model

    final_model.fit(X, y)

    # 6. Evaluation (on Test Set)
    # We need a separate eval model trained only on TRAIN split to report fair metrics
    eval_model = build_calibrated_classifier(base_model) if payload.calibrate else base_model
    base_model_eval = build_pipeline(max_tfidf_features=max_tfidf, use_text=use_text, rare_top_k=rare_top, C=best_c)
    model_for_eval = build_calibrated_classifier(base_model_eval) if payload.calibrate else base_model_eval
    
    model_for_eval.fit(X_train, y_train)
    y_pred = model_for_eval.predict(X_test)
    
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS).tolist()

    # 7. Save
    meta = {
        "model_name": "LogisticRegression",
        "version": "1.0.0",
        "metrics": {"f1_macro": f1_macro},
        "report": report,
        "confusion_matrix": cm,
        "chosen_C": best_c,
        "calibrated": payload.calibrate,
        "use_text": use_text,
        "thresholds": thresholds,
        "labels": CLASS_LABELS,
        "features": NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if use_text else []),
        "split_strategy": split_strategy,
        "cutoff_date": cutoff_date if split_strategy == "temporal" else None,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
    save_model(final_model, meta)

    return {
        "ok": True,
        "metrics": meta["metrics"],
        "metadata": meta
    }

@app.post("/predict", tags=["Inference"])
def predict(payload: PredictRequest = Body(...)):
    """
    Predicts profitability class (Low, Medium, High) for a list of jobs.
    Returns class prediction and confidence score.
    """
    jobs = payload.data or []
    model, meta = load_model()

    # 1. Fallback if no model
    if not model:
        preds = [fallback_classification(j) for j in jobs]
        return {
            "predictions": preds, 
            "count": len(preds), 
            "model_loaded": False,
            "note": "Using heuristic fallback."
        }
    
    try:
        # 2. Feature Engineering
        thresholds = meta.get("thresholds") if meta else None
        df, _ = build_dataframe(jobs, thresholds)

        if df.empty:
            return {"predictions": [], "count": 0, "model_loaded": True}

        # 3. Inference
        y_pred = model.predict(df)
        
        proba = None
        if hasattr(model, "predict_proba"):
             proba = model.predict_proba(df)

        results = []
        for i, j in enumerate(jobs):
            jid = j.get("ID") or j.get("id")
            predicted_class = str(y_pred[i])
            
            row = {
                "jobId": jid, 
                "class": predicted_class
            }
            
            # Extract confidence
            if proba is not None:
                # Find index of predicted class
                if predicted_class in CLASS_LABELS:
                    # Model knows these labels, find index
                    # Note: model.classes_ might not match CLASS_LABELS order exactly
                    classes = list(model.classes_)
                    if predicted_class in classes:
                        idx = classes.index(predicted_class)
                        row["confidence"] = float(proba[i][idx])
                else:
                    # Fallback: max prob
                    row["confidence"] = float(np.max(proba[i]))
            
            results.append(row)

        return {
            "predictions": results,
            "count": len(results),
            "model_loaded": True,
            "metrics": {"f1_macro": meta.get("metrics", {}).get("f1_macro")} if meta else None
        }
    
    except Exception as exc:
        # 4. Error Fallback
        preds = [fallback_classification(j) for j in jobs]
        return {
            "predictions": preds,
            "count": len(preds),
            "model_loaded": False,
            "error": f"Model inference failed: {str(exc)}",
            "diagnostics": build_prediction_diagnostics(exc, jobs, df, model, meta),
        }
