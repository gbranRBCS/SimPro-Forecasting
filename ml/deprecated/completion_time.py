#ML Service: Completion Time Prediction

import os
import json
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, Body

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Path Setup
MODULE_DIR = os.path.dirname(__file__)

from transformers import RareCategoryCapper, Log1pTransformer
from common import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TEXT_COL,
    to_num,
    to_text,
    to_bool,
    parse_job_features,
    PredictRequest,
    TrainRequest,
)

# Application Config

app = FastAPI(title="ML Service - Completion Time", version="1.0.0")

# Model Persistence Paths
DURATION_MODEL_PATH = os.environ.get("DURATION_MODEL_PATH", "model_duration.joblib")
DURATION_META_PATH = os.environ.get("DURATION_MODEL_META_PATH", "model_duration_meta.json")

TEXT_FEATURE = TEXT_COL[0] if isinstance(TEXT_COL, list) else TEXT_COL

# Dataset building

def build_duration_dataset(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    # Converts raw JSON records into pd DataFrame.
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

# Model Pipeline Construction

def build_duration_pipeline(
    *,
    max_tfidf_features: int = 500,
    use_text: bool = True,
    rare_top_k: int = 20,
    estimator: Optional[Any] = None,
) -> Pipeline:
    # Scikit-Learn regression pipeline
    
    # Numeric 
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", Log1pTransformer()),
        ("scaler", RobustScaler()),
    ])
    
    # Categorical 
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
    # If data sample is  small, linear regression could be better choice
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
        return None
    try:
        model = joblib.load(DURATION_MODEL_PATH)
        meta = None
        if os.path.exists(DURATION_META_PATH):
            with open(DURATION_META_PATH) as f:
                meta = json.load(f)
        return model, meta
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

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

# API Endpoints

@app.get("/", tags=["System"])
def root():
    return {
        "message": "ML Service - Completion Time Prediction",
        "endpoints": ["/health", "/predict", "/predict_duration", "/train", "/train_duration", "/model/info"]
    }

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns metadata about the currently loaded active model."""
    model, meta = load_model()
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
    # Retrains the Duration prediction model.
    # option for temporal split if cutoff_date available

    # Prepare Dataset
    df, y_all, info = build_duration_dataset(payload.data)
    if df.empty:
        return {"ok": False, "error": "No data provided for training."}

    # Filter to only completed jobs
    labelled_mask = y_all.notna()
    X = df.loc[labelled_mask]
    y = y_all.loc[labelled_mask]

    if len(X) < 30:
        return {"ok": False, "error": f"Insufficient labelled data - ({len(X)} rows). Need >= 30."}

    # Validation Split
    issued_series = info.get("issued_dates")
    if isinstance(issued_series, pd.Series):
        issued_series = issued_series.loc[labelled_mask]
    
    split_strategy = "random"
    X_train, X_test, y_train, y_test = None, None, None, None

    # Temporal split 
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
                    "error": "Temporal split resulted in empty train or test set. Change cutoff date."
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

    # Model Training & Selection
    try:
        # Default configuration
        use_text = payload.use_text
        max_tfidf = payload.max_tfidf_features or 500
        rare_top = payload.rare_top_k or 20

        try:
            selection = select_duration_model(
                X_train, y_train, X_test, y_test,
                use_text=use_text, rare_top_k=rare_top, max_tfidf_features=max_tfidf
            )
        except ValueError as exc:
            # Retry without text if vocab is empty
            if use_text and "empty vocabulary" in str(exc).lower():
                selection = select_duration_model(
                    X_train, y_train, X_test, y_test,
                    use_text=False, rare_top_k=rare_top, max_tfidf_features=max_tfidf
                )
                use_text = False
            else:
                raise exc

        # Fit best model on full data
        final_model = selection["model"]
        final_model.fit(X, y)

        # Save metadata
        meta = {
            "model_name": selection["model_name"],
            "metrics": selection["metrics"],
            "features": NUMERIC_COLS + CATEGORICAL_COLS + ([TEXT_FEATURE] if use_text else []),
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

@app.post("/predict", tags=["Inference"])
@app.post("/predict_duration", tags=["Inference"])
def predict_duration(payload: PredictRequest = Body(...)):
    #Generates completion time estimates (days) for a list of jobs.

    jobs = payload.data or []
    model, meta = load_model()

    if not model:
        return {
            "predictions": [],
            "count": 0,
            "model_loaded": False,
            "error": "Model not trained. Please call /train first."
        }

    try:
        # Build DataFrame
        df, _, info = build_duration_dataset(jobs)
        if df.empty:
            return {"predictions": [], "count": 0, "model_loaded": True}

        # Run prediction
        preds = model.predict(df)
        
        # Format Response
        results = []
        is_completed_series = info.get("is_completed", pd.Series())
        job_ids = info.get("job_ids", [])

        for idx, job in enumerate(jobs):
            job_id = job_ids[idx] if idx < len(job_ids) else job.get("ID")
            
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
