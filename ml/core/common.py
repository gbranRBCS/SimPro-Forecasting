import math
import json
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# shared constants
NUMERIC_COLS = [
    "revenue", "materials", "labour", "overhead", "cost_total", "job_age_days", "lead_time_days", "is_overdue"
]
CATEGORICAL_COLS = [
    "statusName", "jobType", "customerName", "siteName", "date_month", "date_dow"
]
TEXT_COL = ['descriptionText']

ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS + TEXT_COL

# helpers
def to_num(value):
    if value is None:
        return None
    
    try:
        # check if already a float
        if type(value) == float:
            if math.isnan(value):
                return None
            return value
        
        # check ints
        if type(value) == int:
            return float(value)
        
        # check strings
        if type(value) == str:
            # remove whitespace
            s = value.strip()
            if s == '':
                return None
            
            # remove formatting chars
            s = s.replace(',', '').replace('$', '').replace('£', '')
            return float(s)
        
        # try converting anything else
        return float(value)
    except:
        return None
    
def to_text(value):
    if value is None:
        return None

    # check string
    if type(value) == str:
        s = value.strip()
        return None if len(s) == 0 else s
    
    # check numbers
    if type(value) in [int, float]:
        return str(value)
    
    # check bool
    if type(value) == bool:
        return str(value)
    
    # check dicts
    if type(value) == dict:
        # try to find name or label
        if 'name' in value:
            return str(value['name'])
        elif 'Name' in value:
            return str(value['Name'])
        elif 'label' in value:
            return str(value['label'])
        elif 'value' in value:
            return str(value['value'])
        else:
            # dump dict as string
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
            return ''.join(string_list)
        
    return str(value)

def to_bool(value):
    if value is None:
        return False
    
    # check bool
    if type(value) == bool:
        return value
    
    # check str
    if type(value) == str:
        truthy = ['true', '1', 'yes']
        falsy = ['false', '0', 'no']
        
        s = value.strip().lower()
        if s in truthy:
            return True
        elif s in falsy:
            return False
        return False
        
    # check numbers
    if type(value) in [int, float]:
        return True if int(value) == 1 else False
        
# job feature parser
def parse_job_features(job: Dict[str, Any]) -> Dict[str, Any]:
    # extract ml features
    
    # numerical fields
    revenue = to_num(job.get('revenue'))

    nums = ['materials', 'labour', 'overhead']

    materials = to_num(job.get(nums[0])) or 0.0
    labour = to_num(job.get(nums[1])) or 0.0
    overhead = to_num(job.get(nums[2])) or 0.0

    # total cost logic

    cost_total = to_num(job.get('cost_total'))
    if cost_total is None:
        cost_total = to_num(job.get("cost_est_total"))

    # fallback calculation
    if cost_total is None:
        cost_total = materials + labour + overhead
        if cost_total <= 0:
            cost_total = None
     
    # date parsing
    date_issued = job.get('dateIssued')

    # time features
    job_age_days = to_num(job.get('job_age_days'))
    lead_time_days = to_num(job.get('lead_time_days'))
    is_overdue = to_bool(job.get('is_overdue'))

    # date derivatives
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

    # text and categorical features
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

# shared models
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

