from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import math

app = FastAPI()

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

def to_num(x):
    try: return float(x)
    except: return None

def classify_job(j: Dict[str, Any]):
    revenue = to_num(j.get("revenue"))
    materials = to_num(j.get("materials_cost_est")) or 0.0
    labour = to_num(j.get("labor_cost_est")) or 0.0
    overhead = to_num(j.get("overhead_est")) or 0.0
    cost_total = to_num(j.get("cost_est_total"))
    if cost_total is None: cost_total = materials + labour + overhead

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
        # sigmoid function to generate probaility from margin
        prob = 1 / (1 + math.exp(-6*(margin_est))) 
        prob = float(np.clip(prob, 0.01, 0.99))

    return {
        "jobId": j.get("ID"),
        "profitable": bool(profitable),
        "probability": prob,
        "profit_est": float(profit_est) if profit_est is not None else None,
        "margin_est": float(margin_est) if margin_est is not None else None
    }

@app.post("/predict")
def predict(payload: PredictRequest = Body(...)):
    jobs = payload.data or []
    preds = [classify_job(j) for j in jobs]
    return {"predictions": preds, "count": len(preds)}