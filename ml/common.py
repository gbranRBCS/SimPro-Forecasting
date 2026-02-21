import os
import math
import json
import traceback
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Shared Constants
NUMERIC_COLS = [
    "revenue", "materials", "labour", "overhead", "cost_total", "job_age_days", "lead_time_days", "is_overdue"
]
CATEGORICAL_COLS = [
    "statusName", "jobType", "customerName", "siteName", "date_month", "date_dow"
]
TEXT_COL = ['descriptionText']

ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS + TEXT_COL

# --- Helpers
def to_num(value):
    if value is None:
        return None
    
    try:
        # check if already a float, consider nan (counts as float)
        if type(value) == float:
            if math.isnan(value):
                return None
            return value
        
        # check for ints
        if type(value) == int:
            return float(value)
        
        # check for strings
        if type(value) == str:
            # remove whitespace
            s = value.strip()
            if s == '':
                return None
            
            # remove commas and currency
            s = s.replace(',', '').replace('$', '').replace('£', '')

            s_float = float(s)
            return s_float
        
        # try returning anything else
        return float(value)
    # if error, default to None
    except:
        return None
    
def to_text(value):
    if value is None:
        return None

    # check if already a string
    if type(value) == str:
        s = value.strip()
        if len(s) == 0:
            return None
        else:
            return s
    
    # check int / float
    if type(value) == int or type(value) == float:
        return str(value)
    
    # check bool
    if type(value) == bool:
        return str(value)
    
    # check dicts
    if type(value) == dict:
        # try to find name
        if 'name' in value:
            return str(value['name'])
        elif 'Name' in value:
            return str(value['Name'])
        elif 'label' in value:
            return str(value['label'])
        elif 'value' in value:
            return str(value['value'])
        else:
            # if no name, dump whole dict as string
            try:
                return json.dumps(value)
            except:
                return str(value)
            
    # check list
    if type(value) == List:
        string_list: List = []
        for i in value:
            s = to_text(i)
            if s is not None:
                string_list.append(s)

        if len(string_list) > 0:
            final_string = ''.join(string_list)
            return final_string
        
    return str(value)

def to_bool(value):
    if value == None:
        return False
    
    # check if already bool
    if type(value) == bool:
        return value
    
    # check str
    if type(value) == str:

        truthy = ['true', '1', 'yes']
        falsy = ['false', '0', 'no']
        
        if value.strip().lower() in truthy:
            return True
        elif value.strip().lower() in falsy:
            return False
        else:
            return False
        
    # check int / float
    if type(value) == int or type(value) == float:
        if int(value) == 1:
            return True
        else:
            return False
        
# --- Job Parser
def parse_job_features(job: Dict[str, Any]) -> Dict[str, Any]:
    # Extract ML features from job dictionary.
    
    # Numerical fields
    revenue = to_num(job.get('revenue'))

    nums = ['materials', 'labour', 'overhead']

    materials = to_num(job.get(nums[0])) or 0.0
    labour = to_num(job.get(nums[1])) or 0.0
    overhead = to_num(job.get(nums[2])) or 0.0

    # Cost Total Logic

    cost_total = to_num(job.get('cost_total'))
    if cost_total is None:
        cost_total = to_num(job.get("cost_est_total"))

    # if missing, calc manually
    if cost_total is None:
        cost_total = materials + labour + overhead
        if cost_total <= 0:
            cost_total = None
     
    # Date Parsing
    date_issued = job.get('dateIssued')

    # Time Features
    job_age_days = to_num(job.get('job_age_days'))
    lead_time_days = to_num(job.get('lead_time_days'))
    is_overdue = to_bool(job.get('is_overdue'))

    # Date Derivatives
    date_month = "Unknown"
    date_dow = "Unknown"
    
    if date_issued:
        try:
            dt = pd.to_datetime(date_issued)
            if pd.notnull(dt):
                date_month = dt.strftime("%b")
                date_dow = dt.strftime("%a")
        except:
            pass

    # Text / Categorical Features
    text_cat = [
        'statusName', 'customerName', 'siteName', 'descriptionText', 'jobType'
    ]
    status_name = to_text(job.get(text_cat[0]))
    customer_name = to_text(job.get(text_cat[1]))
    site_name = to_text(job.get(text_cat[2]))
    description = to_text(job.get(text_cat[3]))
    job_type = to_text(job.get(text_cat[4]))

    return {
        'revenue': revenue,
        'materials': materials,
        'labour': labour,
        'overhead': overhead,
        'cost_total': cost_total,
        'job_age_days': job_age_days,
        'lead_time_days': lead_time_days,
        'is_overdue': is_overdue,
        'dateIssued': date_issued,
        'date_month': date_month,
        'date_dow': date_dow,
        'statusName': status_name,
        'jobType': job_type,
        'customerName': customer_name,
        'siteName': site_name,
        'descriptionText': description,
        '_id': job.get('ID') or job.get('id')
    }

# --- Shared Pydantic Models
class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]
    test_size: float =0.15
    random_state: int = 42
    max_tfidf_features: int = 500
    rare_top_k: int = 20
    use_text: bool = True
    cutoff_date: Optional[str] = None
    # Profitability specific:
    thresholds: Optional[Dict[str, float]] = None
    calibrate: Optional[bool] = False

