import os
import json
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .transformers import RareCategoryCapper, Log1pTransformer
from .common import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TEXT_COL,
    to_num,
    to_text,
    to_bool,
    parse_job_features,
)

import sys
from . import transformers as _transformers_module
sys.modules.setdefault("transformers", _transformers_module)

# model persistence paths
_core_dir = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.dirname(_core_dir)

DURATION_MODEL_PATH = os.environ.get("DURATION_MODEL_PATH", os.path.join(_ml_dir, "model_duration.joblib"))
DURATION_META_PATH = os.environ.get("DURATION_MODEL_META_PATH", os.path.join(_ml_dir, "model_duration_meta.json"))

TEXT_FEATURE = TEXT_COL[0] if isinstance(TEXT_COL, list) else TEXT_COL

# dataset building

def build_duration_dataset(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    # converts raw json records into pd dataframe
    rows = []
    durations = []
    is_completed_flags = []
    issued_dates = []
    job_ids = []

    for j in records:
        row = parse_job_features(j)
        
        row["descriptionText"] = to_text(j.get("descriptionText")) or ""

        issued_date_value = row.get("dateIssued")

        # Target Variable
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

    # Convert to pd
    df = pd.DataFrame(rows)
    y = pd.Series(durations, dtype="float") if rows else pd.Series(dtype="float")
    
    info = {
        "is_completed": pd.Series(is_completed_flags, dtype="bool") if is_completed_flags else pd.Series(dtype="bool"),
        "issued_dates": pd.Series(issued_dates, dtype="object") if issued_dates else pd.Series(dtype="object"),
        "job_ids": job_ids,
    }
    return df, y, info

# model pipeline construction

def build_duration_pipeline(
    *,
    max_tfidf_features: int = 500,
    use_text: bool = True,
    rare_top_k: int = 20,
    estimator: Optional[Any] = None,
) -> Pipeline:
    # scikit-learn regression pipeline
    
    # numeric 
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", Log1pTransformer()),
        ("scaler", RobustScaler()),
    ])
    
    # categorical 
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ]

    # Text 
    if use_text:
        text_transformer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
        )
        transformers.append(("txt", text_transformer, TEXT_FEATURE))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )

    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=300, random_state=42)

    # Final pipeline
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
    #Model Selection:
    #Trains Random Forest and Linear regression, and chooses best one accorind to mean absolute error score.
    estimators = [
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=300, random_state=42))
    ]
    # if data sample is small, linear regression could be better choice
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
        
        result = {
            "model": pipeline,
            "model_name": model_name,
            "metrics": {"MAE": mae},
        }
        
        if best_result is None or mae < best_result["metrics"]["MAE"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Model training failed (no candidates succeeded).")
        
    return best_result

# Persistence

def save_duration_model(model: Pipeline, meta: Dict[str, Any]) -> None:
    joblib.dump(model, DURATION_MODEL_PATH)
    with open(DURATION_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def load_model() -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
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
    #Generates bug info for failed requests
    diagnostics = {}
    diagnostics["error"] = str(exc)
    diagnostics["job_count"] = len(jobs)

    if df is not None:
        diagnostics["dataframe_exists"] = True
        diagnostics["row_count"] = len(df)
        
        # serialize sample rows for JSON
        try:
            sample = df.head(1).to_dict(orient="records")
            diagnostics["sample_row"] = sample[0] if sample else None
        except Exception:
            diagnostics["sample_row"] = None
    else:
        diagnostics["dataframe_exists"] = False

    return diagnostics
