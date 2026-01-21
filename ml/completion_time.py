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
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestRegressor

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR and MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from transformers import RareCategoryCapper, Log1pTransformer

app = FastAPI(title="ML Service - Completion Time", version="1.0.0")

DURATION_MODEL_PATH = os.environ.get("DURATION_MODEL_PATH", "model_duration.joblib")
DURATION_META_PATH = os.environ.get("DURATION_MODEL_META_PATH", "model_duration_meta.json")

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


# feature columns used by the completion time model
NUMERIC_COLS = [
    "revenue","materials","labour","overhead","cost_total",
    "job_age_days","lead_time_days","is_overdue"
]
CATEGORICAL_COLS = ["statusName","jobType","customerName","siteName","date_month","date_dow"]
TEXT_COL = "descriptionText"
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS + [TEXT_COL]


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

        # fallback - simple casting
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
        # prefer readable keys
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

        # enhance with date components
        dval = row.get("dateIssued")
        if dval:
            try:
                dt = pd.to_datetime(dval)
                if pd.notnull(dt):
                    row["date_month"] = dt.strftime("%b")
                    row["date_dow"] = dt.strftime("%a")
                else:
                    row["date_month"] = "Unknown"
                    row["date_dow"] = "Unknown"
            except Exception:
                row["date_month"] = "Unknown"
                row["date_dow"] = "Unknown"
        else:
             row["date_month"] = "Unknown"
             row["date_dow"] = "Unknown"

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
            ("log", Log1pTransformer()),
            ("scaler", RobustScaler()),
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


def save_duration_model(model: Pipeline, meta: Dict[str, Any]) -> None:
    joblib.dump(model, DURATION_MODEL_PATH)
    with open(DURATION_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_duration_model() -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    if not os.path.exists(DURATION_MODEL_PATH):
        return None, None
    model = joblib.load(DURATION_MODEL_PATH)
    meta = None
    if os.path.exists(DURATION_META_PATH):
        with open(DURATION_META_PATH) as f:
            meta = json.load(f)
    return model, meta


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

    try:
        diagnostics["job_ids"] = [to_serializable(j.get("ID") or j.get("id")) for j in jobs[:5]]
    except Exception:
        diagnostics["job_ids"] = []

    return diagnostics


@app.get("/")
def root():
    return {
        "message": "ML service - Completion Time Prediction",
        "endpoints": ["/health", "/predict_duration", "/train_duration", "/model/info"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    model, meta = load_duration_model()
    if not model:
        return {"loaded": False}
    
    info = {
        "loaded": True,
        "model_name": meta.get("model_name") if meta else None,
        "metrics": meta.get("metrics") if meta else None,
        "features": meta.get("features") if meta else None,
        "use_text": meta.get("use_text") if meta else True,
        "rare_top_k": meta.get("rare_top_k") if meta else None,
        "meta": meta,
    }
    if meta and meta.get("cutoff_date"):
        info["cutoff_date"] = meta.get("cutoff_date")
    return info


@app.post("/train")
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
