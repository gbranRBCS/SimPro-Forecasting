#Profitability Classification ML Service


import os
import sys
import json
import math
import traceback
import inspect
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

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


# Path Setup
MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR and MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from transformers import RareCategoryCapper, Log1pTransformer
from common import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TEXT_COL,
    ALL_FEATURES,
    to_num,
    to_text,
    parse_job_features,
    PredictRequest,
    TrainRequest as CommonTrainRequest,
)

# Application Config

app = FastAPI(title="ML Service - Profitability", version="1.0.0")

# Model Persistence Paths
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
META_PATH = os.environ.get("MODEL_META_PATH", "model_meta.json")

TEXT_FEATURE = TEXT_COL[0] if isinstance(TEXT_COL, list) else TEXT_COL

# Labels
CLASS_LABELS = ["Low", "Medium", "High"]
DEFAULT_THRESHOLDS = {"low": 0.44, "high": 0.64}

class Thresholds(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None

class TrainRequest(CommonTrainRequest):
    thresholds: Optional[Thresholds] = None

def to_serializable(value: Any) -> Any:
    if value is None:
        return None
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
    resolved = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        if thresholds.get("low") is not None:
            resolved["low"] = float(thresholds["low"])
        if thresholds.get("high") is not None:
            resolved["high"] = float(thresholds["high"])
    return resolved

def fallback_classification(j: Dict[str, Any]) -> Dict[str, Any]:
    revenue = to_num(j.get("revenue"))
    materials = to_num(j.get("materials")) or 0.0
    labour = to_num(j.get("labour")) or 0.0
    overhead = to_num(j.get("overhead")) or 0.0
    cost_total = to_num(j.get("cost_total"))
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

    profitable = (profit_est or 0) > 0 if profit_est is not None else (margin_est or 0) > 0.10
    if margin_est is None:
        prob = 0.5 if profitable else 0.3
    else:
        prob = 1 / (1 + math.exp(-6 * margin_est))
        prob = float(np.clip(prob, 0.01, 0.99))

    klass = "Unknown"
    if margin_est is not None:
        if margin_est >= 0.20:
            klass = "High"
        elif margin_est >= 0.05:
            klass = "Medium"
        else:
            klass = "Low"

    return {
        "jobId": j.get("id"),
        "class": klass,
        "profitable": bool(profitable),
        "probability": prob,
        "profit_est": profit_est,
        "margin_est": margin_est
    }

def derive_label(row: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[str]:
    p = to_num(row.get("netMarginPct"))
    if p is None:
        rev = to_num(row.get("revenue"))
        cost = to_num(row.get("cost_total"))
        if rev and cost and rev > 0:
            p = (rev - cost) / rev
    if p is None:
        return None

    high = thresholds.get("high", DEFAULT_THRESHOLDS["high"])
    low = thresholds.get("low", DEFAULT_THRESHOLDS["low"])
    if p > high:
        return "High"
    if p >= low:
        return "Medium"
    return "Low"

def build_dataframe(
    records: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    rows = []
    labels = []
    resolved_thresholds = resolve_thresholds(thresholds)

    for j in records:
        row = parse_job_features(j)

        row["descriptionText"] = to_text(j.get("descriptionText")) or ""

        try:
            dt = pd.to_datetime(row.get("dateIssued"))
            row["date_month"] = dt.strftime("%b") if pd.notnull(dt) else "Unknown"
            row["date_dow"] = dt.strftime("%a") if pd.notnull(dt) else "Unknown"
        except:
            row["date_month"] = "Unknown"
            row["date_dow"] = "Unknown"

        rows.append(row)
        lbl = j.get("profitability_class")
        if lbl is None:
            lbl = derive_label(row, resolved_thresholds)
        labels.append(lbl)

    df = pd.DataFrame(rows)
    if df.empty or not any(l is not None for l in labels):
        return df, None
    return df, pd.Series(labels, dtype="object")

def build_calibrated_classifier(base, *, method: str = "sigmoid", cv: int = 3) -> CalibratedClassifierCV:
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
        transformers[-1] = ("txt", text_transformer, TEXT_FEATURE)

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        multi_class="multinomial",
        C=C,
    )
    return Pipeline(steps=[
        ("rare_cap", RareCategoryCapper(columns=["customerName", "siteName"], top_k=rare_top_k)),
        ("pre", pre),
        ("clf", lr),
    ])

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
    except Exception:
        return None, None

def build_prediction_diagnostics(
    exc: Exception,
    jobs: List[Dict[str, Any]],
    df: Optional[pd.DataFrame],
    model: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    thresholds = resolve_thresholds(
        payload.thresholds.dict(exclude_none=True) if payload.thresholds else None
    )
    df, y_all = build_dataframe(payload.data, thresholds)

    if df.empty:
        return {"ok": False, "error": "No usable data rows."}
    if y_all is None:
        return {"ok": False, "error": "No labels found."}

    mask = y_all.notna()
    X, y = df.loc[mask], y_all.loc[mask]
    keep_mask = y.isin(CLASS_LABELS)
    X, y = X.loc[keep_mask], y.loc[keep_mask]

    if len(X) < 12 or y.nunique() < 2:
        return {"ok": False, "error": f"Insufficient data: {len(X)} rows, {y.nunique()} classes."}
    split_strategy = "random"
    cutoff_date = payload.cutoff_date
    if cutoff_date and "dateIssued" in X.columns and X["dateIssued"].notna().any():
        split_strategy = "temporal"
        date_series = pd.to_datetime(X["dateIssued"], errors="coerce")
        cutoff_ts = pd.to_datetime(cutoff_date, errors="coerce")
        if not pd.notnull(cutoff_ts):
            return {"ok": False, "error": f"Invalid cutoff_date: {cutoff_date}"}
        train_mask = (date_series < cutoff_ts) | date_series.isna()
        test_mask = date_series >= cutoff_ts
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        if len(X_train) == 0 or len(X_test) == 0:
            return {"ok": False, "error": "Temporal split created empty sets."}
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=payload.test_size or 0.15,
            random_state=payload.random_state, stratify=y
        )

    if y_train.value_counts().min() < 3:
        return {"ok": False, "error": "Insufficient samples per class."}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=payload.random_state)
    candidate_cs = [0.25, 0.5, 1.0, 2.0]
    best_c = 1.0
    best_score = -np.inf

    max_tfidf = payload.max_tfidf_features or 500
    use_text = True if payload.use_text is None else payload.use_text
    rare_top = payload.rare_top_k or 20

    for c in candidate_cs:
        pipe = build_pipeline(max_tfidf_features=max_tfidf, use_text=use_text, rare_top_k=rare_top, C=c)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_c = c
    base_model = build_pipeline(max_tfidf_features=max_tfidf, use_text=use_text, rare_top_k=rare_top, C=best_c)
    final_model = build_calibrated_classifier(base_model) if payload.calibrate else base_model
    final_model.fit(X, y)

    base_model_eval = build_pipeline(max_tfidf_features=max_tfidf, use_text=use_text, rare_top_k=rare_top, C=best_c)
    model_for_eval = build_calibrated_classifier(base_model_eval) if payload.calibrate else base_model_eval
    model_for_eval.fit(X_train, y_train)
    y_pred = model_for_eval.predict(X_test)
    
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS).tolist()
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
        "features": NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_FEATURE] if use_text else []),
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
    jobs = payload.data or []
    model, meta = load_model()

    if not model:
        preds = [fallback_classification(j) for j in jobs]
        return {"predictions": preds, "count": len(preds), "model_loaded": False}
    
    try:
        thresholds = meta.get("thresholds") if meta else None
        df, _ = build_dataframe(jobs, thresholds)
        if df.empty:
            return {"predictions": [], "count": 0, "model_loaded": True}

        y_pred = model.predict(df)
        proba = model.predict_proba(df) if hasattr(model, "predict_proba") else None

        results = []
        for i, j in enumerate(jobs):
            jid = j.get("id")
            predicted_class = str(y_pred[i])
            row = {"jobId": jid, "class": predicted_class}
            
            if proba is not None and predicted_class in CLASS_LABELS:
                classes = list(model.classes_)
                if predicted_class in classes:
                    idx = classes.index(predicted_class)
                    row["confidence"] = float(proba[i][idx])
            results.append(row)

        return {
            "predictions": results,
            "count": len(results),
            "model_loaded": True,
            "metrics": {"f1_macro": meta.get("metrics", {}).get("f1_macro")} if meta else None
        }
    except Exception as exc:
        preds = [fallback_classification(j) for j in jobs]
        return {
            "predictions": preds,
            "count": len(preds),
            "model_loaded": False,
            "error": str(exc),
            "diagnostics": build_prediction_diagnostics(exc, jobs, df, model, meta),
        }
