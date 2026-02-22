import api from "../../lib/api";

/**
Represents a job row from backend
 */
export interface Job {
  id: number;
  descriptionText: string;
  desc_len: number;
  
  customerName: string;
  siteName: string;
  status_name: string;
  stage: string;
  jobType: string;
  
  // Date strings in ISO format (YYYY-MM-DD)
  dateIssued: string;
  dateDue: string;
  dateCompleted: string;

  age_days: number | null;
  due_in_days: number | null;
  completion_days: number | null;
  
  is_completed: boolean;
  is_overdue: boolean;
  has_emergency: number; // 0 or 1
  
  // Financials
  revenue: number | null;
  materials_cost_est: number;
  labor_cost_est: number;
  overhead_est: number;
  labor_hours_est: number;
  cost_est_total: number;
  
  // Any extra fields
  [key: string]: any;
}

/**
Prediction result for a single job.
Used by both profitability and duration endpoints.
 */
export type Prediction = {
  jobId: number | string | null;
  
  // -- Profitability Outputs --
  class?: "Low" | "Medium" | "High";
  confidence?: number;      // 0 to 1
  probability?: number;
  
  // -- Duration Outputs --
  predicted_completion_days?: number; // In days
};

export interface SyncParams {
  from?: string;
  to?: string;
  force?: boolean;
  mode?: "update" | "full";
}

export interface GetJobsParams {
  page?: number;
  pageSize?: number;
  sortField?: string;
  order?: "asc" | "desc";
  minRevenue?: number;
  maxRevenue?: number;
  limit?: number;
  [key: string]: any;
}

export interface PredictParams {
  jobs?: Job[];
  jobIds?: (string | number)[];
  limit?: number;
}

export interface GetJobsResponse {
  jobs: Job[];
  total: number;
  page?: number;
  pageSize?: number;
  totalPages?: number;
}

export interface PredictResponse {
  predictions: Prediction[];
  count: number;
  model_loaded: boolean;
}

// --- API Functions ---

export async function syncJobs(params: SyncParams) {
  // sync query is passed through directly from dashboard state
  const syncQuery = {
    DateIssuedFrom: params.from,
    DateIssuedTo: params.to,
    force: params.force,
    mode: params.mode,
  };

  const res = await api.get("/data/sync", { params: syncQuery });
  return res.data;
}

export async function getJobs(params: GetJobsParams) {
  const res = await api.get<GetJobsResponse>(
    "/data/jobs", 
    { params }
  );
  return res.data;
}

export async function predictProfitability(body: PredictParams) {
  const res = await api.post<PredictResponse>("/data/predict", body);
  return res.data;
}

export async function predictDuration(body: PredictParams) {
  const res = await api.post<PredictResponse>("/data/predict_duration", body);
  return res.data;
}
