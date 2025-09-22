from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math
import os
import json
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

app = FastAPI(title="ML Service", version="1.0.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
META_PATH = os.environ.get("MODEL_META_PATH", "model_meta.json")

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]
    test_size: Optional[float] = 0.15
    random_state: Optional[int] = 42
    max_tfidf_features: Optional[int] = 500

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

def derive_label(row: Dict[str, Any]) -> Optional[str]:
    """
    create a 3-class profitability heuristic label from net margin percentage if present
    Thresholds: High > 0.64, Medium 0.44–0.64, Low < 0.44
    """
    # prefer explicit netMarginPct if present
    p = to_num(row.get("netMarginPct"))
    if p is None:
        rev = to_num(row.get("revenue"))
        cost = to_num(row.get("cost_total"))
        if rev is not None and cost is not None and rev > 0:
            p = (rev - cost) / rev
    if p is None:
        return None
    if p > 0.64:
        return "High"
    if p >= 0.44:
        return "Medium"
    return "Low"

def build_dataframe(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    build a pandas DataFrame with expected columns from enriched job dictionaries
    returns (X, y) where y may be None if labels are not available
    """
    rows: List[Dict[str, Any]] = []
    labels: List[Optional[str]] = []

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

        row = {
            "revenue": revenue,
            "materials": materials if materials is not None else None,
            "labour": labour if labour is not None else None,
            "overhead": overhead if overhead is not None else None,
            "cost_total": cost_total,
            "job_age_days": to_num(j.get("job_age_days")),
            "lead_time_days": to_num(j.get("lead_time_days")),
            "is_overdue": to_num(j.get("is_overdue")),
            "statusName": j.get("statusName"),
            "jobType": j.get("jobType"),
            "customerName": j.get("customerName"),
            "siteName": j.get("siteName"),
            "descriptionText": j.get("descriptionText") or "",
            "_id": j.get("ID") or j.get("id"),
        }
        rows.append(row)

        # label handling: prefer provided class, else take from the row (which has our fallbacks)
        lbl = j.get("profitability_class")
        if lbl is None:
            lbl = derive_label(row)
        labels.append(lbl)

    df = pd.DataFrame(rows)

    if df.shape[0] == 0:
        return df, None

    if any(l is not None for l in labels):
        y = pd.Series([l if l is not None else "Unknown" for l in labels])
        return df, y
    return df, None

def build_pipeline(max_tfidf_features: int = 500) -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    text_transformer = TfidfVectorizer(max_features=max_tfidf_features, ngram_range=(1,2))
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("txt", text_transformer, TEXT_COL)
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False
    )
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto")
    pipe = Pipeline(steps=[("pre", pre), ("clf", lr)])
    return pipe

def save_model(model: Pipeline, meta: Dict[str, Any]) -> None:
    joblib.dump(model, MODEL_PATH)
    with open(META_PATH, "w") as f:
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

@app.get("/")
def root():
    return {"message": "ML service is running", "endpoints": ["/health", "/predict", "/train", "/model/info"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(payload: TrainRequest = Body(...)):
    # frame + labels
    df, y_all = build_dataframe(payload.data)

    if df.empty:
        return {"ok": False, "error": "No usable rows after preprocessing. Need either netMarginPct (or computable revenue+costs) or profitability_class."}

    if y_all is None:
        return {"ok": False, "error": "Could not derive labels from provided data (need netMarginPct or profitability_class)."}

    # drop rows without labels
    mask = y_all.notna()
    X = df.loc[mask, ALL_FEATURES]
    y = y_all.loc[mask]

    # only keep known classes
    keep_mask = y.isin(CLASS_LABELS)
    X = X.loc[keep_mask]
    y = y.loc[keep_mask]

    if len(X) < 12 or y.nunique() < 2:
        return {"ok": False, "error": "Not enough labelled data to train (need >=12 rows and at least 2 classes)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=payload.test_size, random_state=payload.random_state, stratify=y
    )

    pipe = build_pipeline(max_tfidf_features=payload.max_tfidf_features)
    pipe.fit(X_train, y_train)

    # evaluate
    y_pred = pipe.predict(X_test)
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS).tolist()

    meta = {
        "model_name": "LogisticRegression",
        "version": "1.0.0",
        "features": ALL_FEATURES,
        "labels": CLASS_LABELS,
        "f1_macro": f1_macro,
        "report": report,
        "confusion_matrix": cm
    }
    save_model(pipe, meta)

    return {"ok": True, "metrics": {"f1_macro": f1_macro, "report": report, "confusion_matrix": cm}}

@app.get("/model/info")
def model_info():
    model, meta = load_model()
    if not model:
        return {"loaded": False}
    return {"loaded": True, "meta": meta}


@app.post("/predict")
def predict(payload: PredictRequest = Body(...)):
    jobs = payload.data or []
    model, meta = load_model()

    if not model:
        # fallback heuristic
        preds = [fallback_classification(j) for j in jobs]
        return {"predictions": preds, "count": len(preds), "model_loaded": False}

    # build feature frame
    df, _ = build_dataframe(jobs)
    X = df[ALL_FEATURES]

    try:
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
    except Exception:
        # if error with model, use fallback
        preds = [fallback_classification(j) for j in jobs]
        return {"predictions": preds, "count": len(preds), "model_loaded": False}