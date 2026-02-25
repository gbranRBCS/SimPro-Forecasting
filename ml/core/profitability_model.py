import os
import json
import math
import traceback
import inspect
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from pydantic import BaseModel

from .transformers import RareCategoryCapper, Log1pTransformer
from .common import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TEXT_COL,
    ALL_FEATURES,
    to_num,
    to_text,
    parse_job_features,
)

import sys
from . import transformers as _transformers_module
sys.modules.setdefault("transformers", _transformers_module)

# model persistence paths
_core_dir = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.dirname(_core_dir)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_ml_dir, "model.joblib"))
META_PATH = os.environ.get("MODEL_META_PATH", os.path.join(_ml_dir, "model_meta.json"))

TEXT_FEATURE = TEXT_COL[0] if isinstance(TEXT_COL, list) else TEXT_COL

# labels
CLASS_LABELS = ["Low", "Medium", "High"]
DEFAULT_THRESHOLDS = {"low": 0.44, "high": 0.64}

class Thresholds(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None

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
    # simplified fallback logic
    # trusts revenue and cost total are handled by utils
    
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
        "confidence": prob,
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
        # use common parser
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
        transformers.append(("txt", text_transformer, TEXT_FEATURE))

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
