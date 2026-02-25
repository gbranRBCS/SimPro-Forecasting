import os
import sys
import pandas as pd

from fastapi import FastAPI, Body
from sklearn.model_selection import train_test_split

# add parent directory to path if running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.common import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    PredictRequest,
    TrainRequest,
)

from core.completion_model import (
    build_duration_dataset,
    select_duration_model,
    save_duration_model,
    load_model,
    build_prediction_diagnostics,
    TEXT_FEATURE,
)

app = FastAPI(title="ML Service - Completion Time", version="1.0.0")

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
    # retrains the duration prediction model
    # option for temporal split if cutoff_date available

    # prepare dataset
    df, y_all, info = build_duration_dataset(payload.data)
    if df.empty:
        return {"ok": False, "error": "No data provided for training."}

    # filter to only completed jobs
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

    # temporal split 
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
        # build dataframe
        df, _, info = build_duration_dataset(jobs)
        if df.empty:
            return {"predictions": [], "count": 0, "model_loaded": True}

        # run prediction
        preds = model.predict(df)
        
        # Format Response
        results = []
        is_completed_series = info.get("is_completed", pd.Series())
        job_ids = info.get("job_ids", [])

        for idx, job in enumerate(jobs):
            job_id = job_ids[idx]
            
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
