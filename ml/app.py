from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple, Iterable
import numpy as np
import math
import os
import sys
import json
import re
import traceback
import inspect
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR and MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from transformers import RareCategoryCapper

app = FastAPI(title="ML Service", version="1.0.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
META_PATH = os.environ.get("MODEL_META_PATH", "model_meta.json")
DURATION_MODEL_PATH = os.environ.get("DURATION_MODEL_PATH", "model_duration.joblib")
DURATION_META_PATH = os.environ.get("DURATION_MODEL_META_PATH", "model_duration_meta.json")

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

def to_num(x):
    try:
        # fast path for numerical values
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return None if (isinstance(x, float) and np.isnan(x)) else float(x)

        # handle strings
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            # strip currency symbols and thousands separators
            s = re.sub(r'[^0-9eE\.\-+]', '', s)
            return float(s)

        # fallback: try simple casting
        return float(x)
    except Exception:
        return None

def to_text(value: Any, *, allow_blank: bool = False) -> Optional[str]:
    """Safely coerce arbitrary values to a usable text string."""
    if value is None:
        return "" if allow_blank else None

    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
        return "" if allow_blank else None

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        # prefer human readable keys
        for key in ("name", "Name", "label", "Label", "value", "Value", "description", "Description"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        parts = []
        for item in value:
            text = to_text(item, allow_blank=False)
            if text:
                parts.append(text)
        if parts:
            return " ".join(parts)
        return "" if allow_blank else None

    return str(value)


def to_serializable(value: Any) -> Any:
    """convert numpy/pandas values into python types."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, (np.floating, np.float32, np.float64)):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    return str(value)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value) == 1)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return False


def fallback_classification(j: Dict[str, Any]):
    revenue = to_num(j.get("revenue"))
    materials = to_num(j.get("materials")) or to_num(j.get("materials_cost_est")) or 0.0
    labour = to_num(j.get("labour")) or to_num(j.get("labor_cost_est")) or 0.0
    overhead = to_num(j.get("overhead")) or to_num(j.get("overhead_est")) or 0.0
    cost_total = to_num(j.get("cost_total")) or to_num(j.get("cost_est_total"))
    if cost_total is None:
        cost_total = materials + labour + overhead

    profit_est = j.get("profit_est")
    profit_est = to_num(profit_est) if profit_est is not None else (revenue - cost_total if revenue is not None else None)
    margin_est = None
    if profit_est is not None and revenue and revenue > 0:
        margin_est = profit_est / revenue

    # decision
    if profit_est is not None:
        profitable = profit_est > 0
    elif margin_est is not None:
        profitable = margin_est > 0.10
    else:
        profitable = False

    # estimate probability from margin
    if margin_est is None:
        prob = 0.5 if profitable else 0.3
    else:
        # sigmoid function to generate probability from margin
        prob = 1 / (1 + math.exp(-6*(margin_est)))
        prob = float(np.clip(prob, 0.01, 0.99))

    klass = "High" if margin_est is not None and margin_est >= 0.20 else ("Medium" if margin_est is not None and margin_est >= 0.05 else ("Low" if margin_est is not None else "Unknown"))

    return {
        "jobId": j.get("ID") or j.get("id"),
        "class": klass,
        "profitable": bool(profitable),
        "probability": prob,
        "profit_est": float(profit_est) if profit_est is not None else None,
        "margin_est": float(margin_est) if margin_est is not None else None
    }

NUMERIC_COLS = [
    "revenue","materials","labour","overhead","cost_total",
    "job_age_days","lead_time_days","is_overdue"
]
CATEGORICAL_COLS = ["statusName","jobType","customerName","siteName"]
TEXT_COL = "descriptionText"
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS + [TEXT_COL]

CLASS_LABELS = ["Low","Medium","High"]

DEFAULT_THRESHOLDS = {"low": 0.44, "high": 0.64}

def resolve_thresholds(thresholds: Optional[Dict[str, Any]]) -> Dict[str, float]:
    resolved = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        low = thresholds.get("low")
        high = thresholds.get("high")
        if low is not None:
            resolved["low"] = float(low)
        if high is not None:
            resolved["high"] = float(high)
    return resolved

def derive_label(row: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[str]:
    """Create a 3-class profitability label using provided thresholds."""
    p = to_num(row.get("netMarginPct"))
    if p is None:
        rev = to_num(row.get("revenue"))
        cost = to_num(row.get("cost_total"))
        if rev is not None and cost is not None and rev > 0:
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

def build_calibrated_classifier(base, *, method: str = "sigmoid", cv: int = 3) -> CalibratedClassifierCV:
    """Wrap CalibratedClassifierCV to support estimator/base_estimator keyword across sklearn versions."""
    params = inspect.signature(CalibratedClassifierCV.__init__).parameters
    kwargs = {"method": method, "cv": cv}
    if "estimator" in params:
        kwargs["estimator"] = base
    else:
        kwargs["base_estimator"] = base
    return CalibratedClassifierCV(**kwargs)


def build_dataframe(
    records: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    build a pandas DataFrame with expected columns from enriched job dictionaries
    returns (X, y) where y may be None if labels are not available
    """
    rows: List[Dict[str, Any]] = []
    labels: List[Optional[str]] = []
    resolved_thresholds = resolve_thresholds(thresholds)

    for j in records:
        # convert numbers and include fallbacks
        revenue = to_num(j.get("revenue") or (j.get("Total", {}) or {}).get("IncTax"))
        materials = to_num(j.get("materials")) or 0.0
        labour = to_num(j.get("labour")) or 0.0
        overhead = to_num(j.get("overhead")) or 0.0

        cost_total = to_num(j.get("cost_total"))
        if cost_total is None:
            # accept alternative estimate field name then fallback to sum of components if necessary
            cost_total = to_num(j.get("cost_est_total"))
        if cost_total is None and (materials or labour or overhead):
            cost_total = materials + labour + overhead

        job_age = to_num(j.get("job_age_days"))
        if job_age is None:
            job_age = 0.0

        lead_time = to_num(j.get("lead_time_days"))
        if lead_time is None:
            lead_time = 0.0

        overdue = to_num(j.get("is_overdue"))
        if overdue is None:
            overdue = 0.0


        row = {
            "revenue": revenue,
            "materials": materials if materials is not None else None,
            "labour": labour if labour is not None else None,
            "overhead": overhead if overhead is not None else None,
            "cost_total": cost_total,
            "job_age_days": job_age,
            "lead_time_days": lead_time,
            "is_overdue": overdue,
            "statusName": to_text(j.get("statusName")),
            "jobType": to_text(j.get("jobType")),
            "customerName": to_text(j.get("customerName")),
            "siteName": to_text(j.get("siteName")),
            "descriptionText": to_text(j.get("descriptionText"), allow_blank=True) or "",
            "_id": j.get("ID") or j.get("id"),
            "dateIssued": j.get("dateIssued") or j.get("DateIssued"),
        }
        rows.append(row)

        # label handling: prefer provided class, else take from the row (which has fallbacks)
        lbl = j.get("profitability_class")
        if lbl is None:
            lbl = derive_label(row, resolved_thresholds)
        labels.append(lbl)

    df = pd.DataFrame(rows)

    if df.shape[0] == 0:
        return df, None

    if any(l is not None for l in labels):
        y = pd.Series(labels, dtype="object")
        return df, y
    return df, None

def build_pipeline(
    max_tfidf_features: int = 500,
    use_text: bool = True,
    rare_top_k: int = 20,
    C: float = 1.0,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ]
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
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        multinomial="auto",
        C=C,
    )
    pipe = Pipeline(
        steps=[
            (
                "rare_cap",
                RareCategoryCapper(columns=["customerName", "siteName"], top_k=rare_top_k),
            ),
            ("pre", pre),
            ("clf", lr),
        ]
    )
    return pipe


def build_duration_dataset(
    records: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    durations: List[Optional[float]] = []
    is_completed_flags: List[bool] = []
    issued_dates: List[Any] = []
    job_ids: List[Any] = []

    for j in records:
        revenue = to_num(j.get("revenue") or (j.get("Total", {}) or {}).get("IncTax"))
        materials = to_num(j.get("materials")) or 0.0
        labour = to_num(j.get("labour")) or 0.0
        overhead = to_num(j.get("overhead")) or 0.0

        cost_total = to_num(j.get("cost_total"))
        if cost_total is None:
            cost_total = to_num(j.get("cost_est_total"))
        if cost_total is None and (materials or labour or overhead):
            cost_total = materials + labour + overhead

        job_age = to_num(j.get("job_age_days"))
        if job_age is None:
            job_age = 0.0

        lead_time = to_num(j.get("lead_time_days"))
        if lead_time is None:
            lead_time = 0.0

        overdue = to_num(j.get("is_overdue"))
        if overdue is None:
            overdue = 0.0

        issued_date_value = (
            j.get("dateIssued")
            or j.get("DateIssued")
            or j.get("issuedDate")
            or j.get("IssuedDate")
            or j.get("createdDate")
            or j.get("CreatedDate")
        )

        row = {
            "revenue": revenue,
            "materials": materials if materials is not None else None,
            "labour": labour if labour is not None else None,
            "overhead": overhead if overhead is not None else None,
            "cost_total": cost_total,
            "job_age_days": job_age,
            "lead_time_days": lead_time,
            "is_overdue": overdue,
            "statusName": to_text(j.get("statusName")),
            "jobType": to_text(j.get("jobType")),
            "customerName": to_text(j.get("customerName")),
            "siteName": to_text(j.get("siteName")),
            "descriptionText": to_text(j.get("descriptionText"), allow_blank=True) or "",
            "_id": j.get("ID") or j.get("id"),
            "dateIssued": issued_date_value,
        }

        rows.append(row)
        job_ids.append(row["_id"])
        issued_dates.append(issued_date_value)

        is_completed_flag = to_bool(j.get("is_completed"))
        is_completed_flags.append(is_completed_flag)

        completion_days = to_num(j.get("completion_days"))
        if (
            is_completed_flag
            and completion_days is not None
            and float(completion_days) > 0
        ):
            durations.append(float(completion_days))
        else:
            durations.append(None)

    df = pd.DataFrame(rows)
    y = pd.Series(durations, dtype="float") if len(rows) else pd.Series(dtype="float")
    info = {
        "is_completed": pd.Series(is_completed_flags, dtype="bool")
        if len(is_completed_flags)
        else pd.Series(dtype="bool"),
        "issued_dates": pd.Series(issued_dates, dtype="object")
        if len(issued_dates)
        else pd.Series(dtype="object"),
        "job_ids": job_ids,
    }
    return df, y, info


def build_duration_pipeline(
    *,
    max_tfidf_features: int = 500,
    use_text: bool = True,
    rare_top_k: int = 20,
    estimator: Optional[Any] = None,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ]
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

    pipe = Pipeline(
        steps=[
            (
                "rare_cap",
                RareCategoryCapper(columns=["customerName", "siteName"], top_k=rare_top_k),
            ),
            ("pre", pre),
            ("reg", estimator),
        ]
    )
    return pipe


def select_duration_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    use_text: bool,
    rare_top_k: int,
    max_tfidf_features: int,
) -> Dict[str, Any]:
    estimators: List[Tuple[str, Any]] = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=300, random_state=42),
        )
    ]
    if len(X_train) < 80:
        estimators.append(("LinearRegression", LinearRegression()))

    best_result: Optional[Dict[str, Any]] = None

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
        raise RuntimeError("No regression model could be trained.")
    return best_result


def save_model(model: Pipeline, meta: Dict[str, Any]) -> None:
    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def save_duration_model(model: Pipeline, meta: Dict[str, Any]) -> None:
    joblib.dump(model, DURATION_MODEL_PATH)
    with open(DURATION_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_model() -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    meta = None
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    return model, meta


def load_duration_model() -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    if not os.path.exists(DURATION_MODEL_PATH):
        return None, None
    model = joblib.load(DURATION_MODEL_PATH)
    meta = None
    if os.path.exists(DURATION_META_PATH):
        with open(DURATION_META_PATH) as f:
            meta = json.load(f)
    return model, meta

@app.get("/")
def root():
    return {"message": "ML service is running", "endpoints": ["/health", "/predict", "/train", "/model/info"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(payload: TrainRequest = Body(...)):
    thresholds = resolve_thresholds(
        payload.thresholds.dict(exclude_none=True) if payload.thresholds else None
    )

    df, y_all = build_dataframe(payload.data, thresholds)

    if df.empty:
        return {
            "ok": False,
            "error": "No usable rows after preprocessing. Need either netMarginPct (or computable revenue+costs) or profitability_class.",
        }

    if y_all is None:
        return {
            "ok": False,
            "error": "Could not derive labels from provided data (need netMarginPct or profitability_class).",
        }

    mask = y_all.notna()
    X = df.loc[mask]
    y = y_all.loc[mask]

    keep_mask = y.isin(CLASS_LABELS)
    X = X.loc[keep_mask]
    y = y.loc[keep_mask]

    if len(X) < 12 or y.nunique() < 2:
        return {
            "ok": False,
            "error": "Not enough labelled data to train (need >=12 rows and at least 2 classes).",
        }

    rare_top_k = int(payload.rare_top_k) if payload.rare_top_k is not None else 20
    use_text = True if payload.use_text is None else bool(payload.use_text)
    calibrate = bool(payload.calibrate)
    max_tfidf = (
        int(payload.max_tfidf_features)
        if payload.max_tfidf_features is not None
        else 500
    )

    split_strategy = "random"
    cutoff_date = payload.cutoff_date
    if cutoff_date and "dateIssued" in X.columns and X["dateIssued"].notna().any():
        split_strategy = "temporal"
        date_series = pd.to_datetime(X["dateIssued"], errors="coerce")
        cutoff_ts = pd.to_datetime(cutoff_date, errors="coerce")
        if pd.isna(cutoff_ts):
            return {"ok": False, "error": f"Invalid cutoff_date '{cutoff_date}'."}
        train_mask = (date_series < cutoff_ts) | date_series.isna()
        test_mask = date_series >= cutoff_ts
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        if len(X_train) == 0 or len(X_test) == 0:
            return {
                "ok": False,
                "error": "Temporal split failed: adjust cutoff_date to ensure both train and test sets have data.",
            }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=payload.test_size,
            random_state=payload.random_state,
            stratify=y,
        )

    if y_train.value_counts().min() < 3:
        return {
            "ok": False,
            "error": "Need at least 3 samples per class in the training split for 3-fold CV. Provide more data or adjust cutoff_date.",
        }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=payload.random_state)
    candidate_cs = [0.25, 0.5, 1.0, 2.0]
    cv_scores: Dict[str, List[float]] = {}
    best_c = candidate_cs[0]
    best_score = -np.inf

    for c in candidate_cs:
        pipe = build_pipeline(
            max_tfidf_features=max_tfidf,
            use_text=use_text,
            rare_top_k=rare_top_k,
            C=c,
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        scores_list = [float(s) for s in scores]
        cv_scores[str(c)] = scores_list
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_c = mean_score, c

    base_model = build_pipeline(
        max_tfidf_features=max_tfidf,
        use_text=use_text,
        rare_top_k=rare_top_k,
        C=best_c,
    )

    if calibrate:
        model_for_eval: Any = build_calibrated_classifier(base_model)
    else:
        model_for_eval = base_model

    model_for_eval.fit(X_train, y_train)

    y_pred = model_for_eval.predict(X_test)
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS).tolist()

    final_base_model = build_pipeline(
        max_tfidf_features=max_tfidf,
        use_text=use_text,
        rare_top_k=rare_top_k,
        C=best_c,
    )
    if calibrate:
        final_model: Any = build_calibrated_classifier(final_base_model)
    else:
        final_model = final_base_model
    final_model.fit(X, y)

    features_used = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if use_text else [])

    meta = {
        "model_name": "LogisticRegression",
        "version": "1.0.0",
        "features": features_used,
        "labels": CLASS_LABELS,
        "f1_macro": f1_macro,
        "report": report,
        "confusion_matrix": cm,
        "chosen_C": best_c,
        "cv_scores": cv_scores,
        "calibrated": calibrate,
        "use_text": use_text,
        "rare_top_k": rare_top_k,
        "max_tfidf_features": max_tfidf,
        "thresholds": thresholds,
        "cutoff_date": cutoff_date if split_strategy == "temporal" else None,
        "split_strategy": split_strategy,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    save_model(final_model, meta)

    return {
        "ok": True,
        "metrics": {
            "f1_macro": f1_macro,
            "report": report,
            "confusion_matrix": cm,
            "chosen_C": best_c,
            "cv_scores": cv_scores,
            "calibrated": calibrate,
        },
        "thresholds": thresholds,
        "split": {
            "strategy": split_strategy,
            "cutoff_date": cutoff_date if split_strategy == "temporal" else None,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
    }

@app.get("/model/info")
def model_info():
    model, meta = load_model()
    if not model:
        return {"loaded": False}
    info = {
        "loaded": True,
        "f1_macro": meta.get("f1_macro") if meta else None,
        "labels": meta.get("labels") if meta else None,
        "features": meta.get("features") if meta else None,
        "chosen_C": meta.get("chosen_C") if meta else None,
        "calibrated": meta.get("calibrated") if meta else False,
        "use_text": meta.get("use_text") if meta else True,
        "rare_top_k": meta.get("rare_top_k") if meta else None,
        "thresholds": meta.get("thresholds") if meta else DEFAULT_THRESHOLDS,
        "meta": meta,
    }
    if meta and meta.get("cutoff_date"):
        info["cutoff_date"] = meta.get("cutoff_date")
    return info


@app.post("/predict")
def predict(payload: PredictRequest = Body(...)):
    jobs = payload.data or []
    model, meta = load_model()

    if not model:
        # fallback heuristic
        preds = [fallback_classification(j) for j in jobs]
        return {"predictions": preds, "count": len(preds), "model_loaded": False}
    
    df: Optional[pd.DataFrame] = None
    try:
        # build feature frame
        df, _ = build_dataframe(jobs, meta.get("thresholds") if meta else None)

        if df.empty:
            return {
                "predictions": [],
                "count": 0,
                "model_loaded": True,
                "metrics": {"f1_macro": meta.get("f1_macro") if meta else None},
            }

        X = df

        y_pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        results = []
        for i, j in enumerate(jobs):
            jid = j.get("ID") or j.get("id")
            row = {"jobId": jid, "class": str(y_pred[i])}
            if proba is not None:
                pred_idx = CLASS_LABELS.index(y_pred[i]) if y_pred[i] in CLASS_LABELS else int(np.argmax(proba[i]))
                row["confidence"] = float(proba[i][pred_idx])
            results.append(row)
        return {"predictions": results, "count": len(results), "model_loaded": True, "metrics": {"f1_macro": meta.get("f1_macro") if meta else None}}
    
    except Exception as exc:
        # if error with model, use fallback
        preds = [fallback_classification(j) for j in jobs]
        return {
            "predictions": preds,
            "count": len(preds),
            "model_loaded": False,
            "error": f"model prediction failed: {exc}",
            "diagnostics": build_prediction_diagnostics(exc, jobs, df, model, meta),
        }


def build_prediction_diagnostics(
    exc: Exception,
    jobs: List[Dict[str, Any]],
    df: Optional[pd.DataFrame],
    model: Pipeline,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """capture debug info when inference fails."""
    if hasattr(model, "named_steps"):
        model_steps = list(model.named_steps.keys())
    elif hasattr(model, "base_estimator") and hasattr(model.base_estimator, "named_steps"):
        model_steps = list(model.base_estimator.named_steps.keys())
    else:
        model_steps = []

    diagnostics: Dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "job_count": len(jobs),
        "model_steps": model_steps,
        "expected_features": meta.get("features") if meta else list(ALL_FEATURES),
    }

    if df is None:
        diagnostics["dataframe_built"] = False
        return diagnostics

    diagnostics["dataframe_built"] = True
    diagnostics["row_count"] = int(df.shape[0])
    diagnostics["column_count"] = int(df.shape[1])
    diagnostics["received_columns"] = list(df.columns)
    diagnostics["missing_features"] = [col for col in ALL_FEATURES if col not in df.columns]

    null_counts: Dict[str, int] = {}
    dtype_map: Dict[str, str] = {}
    for col in df.columns:
        try:
            null_counts[col] = int(pd.isna(df[col]).sum())
        except Exception:
            null_counts[col] = -1
        dtype_map[col] = str(df[col].dtype)
    diagnostics["null_counts"] = null_counts
    diagnostics["dtypes"] = dtype_map

    try:
        head_rows = [
            {col: to_serializable(val) for col, val in row.items()}
            for row in df.head(3).to_dict(orient="records")
        ]
    except Exception:
        head_rows = []
    diagnostics["sample_rows"] = head_rows

    # capture problematic job identifiers when available
    try:
        diagnostics["job_ids"] = [to_serializable(j.get("ID") or j.get("id")) for j in jobs[:5]]
    except Exception:
        diagnostics["job_ids"] = []

    return diagnostics


@app.post("/train_duration")
def train_duration(payload: TrainRequest = Body(...)):
    df, y_all, info = build_duration_dataset(payload.data)

    if df.empty:
        return {"ok": False, "error": "No rows provided for training."}

    labelled_mask = y_all.notna()
    X = df.loc[labelled_mask]
    y = y_all.loc[labelled_mask]

    if len(X) < 30:
        return {
            "ok": False,
            "error": "Not enough labelled data to train duration model (need >=30 rows).",
        }

    rare_top_k = int(payload.rare_top_k) if payload.rare_top_k is not None else 20
    use_text_flag = True if payload.use_text is None else bool(payload.use_text)
    max_tfidf = (
        int(payload.max_tfidf_features)
        if payload.max_tfidf_features is not None
        else 500
    )

    issued_series = info.get("issued_dates")
    if isinstance(issued_series, pd.Series):
        issued_series = issued_series.loc[labelled_mask]
    else:
        issued_series = pd.Series([], dtype="object")

    cutoff_date = payload.cutoff_date
    split_strategy = "random"

    if cutoff_date:
        cutoff_ts = pd.to_datetime(cutoff_date, errors="coerce")
        if pd.isna(cutoff_ts):
            return {"ok": False, "error": f"Invalid cutoff_date '{cutoff_date}'."}

        date_series = pd.to_datetime(issued_series, errors="coerce")
        if date_series.notna().any():
            split_strategy = "temporal"
            train_mask = (date_series < cutoff_ts) | date_series.isna()
            test_mask = date_series >= cutoff_ts
            X_train, y_train = X.loc[train_mask], y.loc[train_mask]
            X_test, y_test = X.loc[test_mask], y.loc[test_mask]
            if len(X_train) == 0 or len(X_test) == 0:
                return {
                    "ok": False,
                    "error": "Temporal split failed: adjust cutoff_date to ensure both train and test sets have data.",
                }
        else:
            split_strategy = "random"

    if split_strategy == "random":
        test_size = payload.test_size if payload.test_size is not None else 0.15
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=payload.random_state,
        )

    try:
        selection = select_duration_model(
            X_train,
            y_train,
            X_test,
            y_test,
            use_text=use_text_flag,
            rare_top_k=rare_top_k,
            max_tfidf_features=max_tfidf,
        )
    except ValueError as exc:
        if use_text_flag and "empty vocabulary" in str(exc).lower():
            use_text_flag = False
            selection = select_duration_model(
                X_train,
                y_train,
                X_test,
                y_test,
                use_text=use_text_flag,
                rare_top_k=rare_top_k,
                max_tfidf_features=max_tfidf,
            )
        else:
            return {"ok": False, "error": f"Training failed: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"Training failed: {exc}"}

    final_model: Pipeline = selection["model"]
    final_model.fit(X, y)

    features_used = NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_COL] if use_text_flag else [])

    meta = {
        "model_name": selection["model_name"],
        "metrics": selection["metrics"],
        "features": features_used,
        "use_text": use_text_flag,
        "cutoff_date": cutoff_date if split_strategy == "temporal" else None,
        "split_strategy": split_strategy,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_samples": int(len(X)),
        "max_tfidf_features": max_tfidf,
        "rare_top_k": rare_top_k,
    }

    save_duration_model(final_model, meta)

    return {
        "ok": True,
        "model_name": selection["model_name"],
        "metrics": selection["metrics"],
        "split": {
            "strategy": split_strategy,
            "cutoff_date": cutoff_date if split_strategy == "temporal" else None,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "use_text": use_text_flag,
    }


@app.post("/predict_duration")
def predict_duration(payload: PredictRequest = Body(...)):
    jobs = payload.data or []
    model, meta = load_duration_model()

    if not model:
        results = [
            {
                "jobId": job.get("ID") or job.get("id"),
                "skipped": True,
                "reason": "model_not_trained",
            }
            for job in jobs
        ]
        return {
            "predictions": results,
            "count": len(results),
            "model_loaded": False,
            "error": "Duration model not trained yet. Call /train_duration first.",
        }

    df: Optional[pd.DataFrame] = None
    try:
        df, _, info = build_duration_dataset(jobs)

        if df.empty:
            return {"predictions": [], "count": 0, "model_loaded": True}

        preds = model.predict(df)

        is_completed_series = info.get("is_completed")
        if not isinstance(is_completed_series, pd.Series):
            is_completed_series = pd.Series([False] * len(df))

        results = []
        for idx, job in enumerate(jobs):
            job_id = info.get("job_ids")[idx] if info.get("job_ids") else job.get("ID") or job.get("id")
            if idx < len(is_completed_series) and bool(is_completed_series.iloc[idx]):
                results.append({"jobId": job_id, "skipped": True, "reason": "already_completed"})
            else:
                predicted_value = float(preds[idx])
                results.append(
                    {
                        "jobId": job_id,
                        "predicted_completion_days": predicted_value,
                    }
                )

        response: Dict[str, Any] = {
            "predictions": results,
            "count": len(results),
            "model_loaded": True,
        }
        if meta and meta.get("metrics"):
            response["metrics"] = meta.get("metrics")
        return response

    except Exception as exc:
        return {
            "predictions": [],
            "count": 0,
            "model_loaded": False,
            "error": f"duration prediction failed: {exc}",
            "diagnostics": build_prediction_diagnostics(exc, jobs, df, model, meta),
        }
