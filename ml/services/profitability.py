import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

from fastapi import FastAPI, Body
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# add parent directory to path if running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.common import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    PredictRequest,
    TrainRequest as CommonTrainRequest,
)

from core.profitability_model import (
    Thresholds,
    resolve_thresholds,
    build_dataframe,
    build_pipeline,
    build_calibrated_classifier,
    save_model,
    load_model,
    build_prediction_diagnostics,
    fallback_classification,
    CLASS_LABELS,
    TEXT_FEATURE,
)

app = FastAPI(title="ML Service - Profitability", version="1.0.0")

class TrainRequest(CommonTrainRequest):
    thresholds: Optional[Thresholds] = None

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
