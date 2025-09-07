from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

app = FastAPI()

def to_number(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def extract_amount(job: Dict[str, Any]) -> Optional[float]:
    total = job.get("Total") if isinstance(job, dict) else getattr(job, "Total", None)
    if not total:
        return None
    return to_number(total.get("IncTax") if isinstance(total, dict) else getattr(total, "IncTax", None))

class JobTotal(BaseModel):
    ExTax: Optional[float] = None
    Tax: Optional[float] = None
    IncTax: Optional[float] = None

class Job(BaseModel):
    ID: Optional[int] = None
    Description: Optional[str] = None
    Total: Optional[JobTotal] = None

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def root():
    return {"message": "ML service is running", "endpoints": ["/health", "/predict"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictRequest = Body(...)):
    jobs = payload.data or []
    amounts: List[float] = []
    for j in jobs:
        v = extract_amount(j)
        if v is not None and np.isfinite(v) and v > 0:
            amounts.append(v)

    mean = float(np.mean(amounts)) if amounts else 0.0
    return {
        "predictions": [{"score": mean}],
        "count": len(amounts)
    }