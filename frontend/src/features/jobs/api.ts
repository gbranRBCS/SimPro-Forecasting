import api from "../../lib/api";

/**
Represents a Job entity as returned by the API.
This combines raw SimPRO data with normalized fields added by the backend during sync.
 */
export interface Job {
  // SimPRO ID is usually 'ID' (int) or 'id' (string) depending on endpoint version
  id: string | number;
  ID?: number;
  
  // Raw SimPRO Description (often contains HTML)
  Description?: string;
  // Cleaned plain-text description added by backend
  descriptionText?: string;
  
  customerName?: string;
  siteName?: string;
  
  // Status object from SimPRO
  status?: { Name?: string; ID?: number; [key: string]: any };
  // Normalized status name
  status_name?: string;
  
  jobType?: string;
  stage?: string;
  
  // Dates (ISO strings)
  dateIssued?: string;
  dateDue?: string;
  dateCompleted?: string;
  
  // Financials
  revenue?: number;
  cost_est_total?: number;
  materials_cost_est?: number;
  labor_cost_est?: number;
  
  // Calculated Metrics
  profit_est?: number;
  margin_est?: number;         // e.g. 0.15 for 15%
  netMarginPct?: number;
  profitability_class?: "Low" | "Medium" | "High";
  
  // Flags & Counters
  is_completed?: boolean;
  is_overdue?: boolean;
  has_emergency?: number;     // 1 or 0
  age_days?: number;
  
  // Flexible index signature for other SimPRO fields not explicitly typed
  [key: string]: any;
}

/**
Prediction result for a single job.
Used for both profitability and duration predictions.
 */
export type Prediction = {
  jobId: number | string | null;
  
  // -- Profitability Outputs --
  class?: "Low" | "Medium" | "High";
  confidence?: number;      // 0 to 1
  probability?: number;    // Legacy confidence field
  
  // -- Heuristic Fallbacks --
  profitable?: boolean;
  profit_est?: number | null;
  margin_est?: number | null;
  
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
  const query = {
    DateIssuedFrom: params.from,
    DateIssuedTo: params.to,
    force: params.force ? "1" : undefined,
    mode: params.mode,
  };
  
  const res = await api.get("/data/sync", { params: query });
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
