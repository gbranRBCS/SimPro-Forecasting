import api from "../../lib/api";

export type ApiJob = Record<string, any>;
export type Prediction = {
  jobId: number | string | null;
  profitable: boolean;
  probability: number; // 0..1
  profit_est?: number | null;
  margin_est?: number | null;
};

export async function syncJobs(params: { from?: string; to?: string; force?: boolean }) {
  const q: Record<string, any> = {};
  if (params?.from) q.DateIssuedFrom = params.from;
  if (params?.to) q.DateIssuedTo = params.to;
  if (params?.force) q.force = "1";
  const r = await api.get("/data/sync", { params: q });
  return r.data;
}


export async function getJobs(params: {
  minRevenue?: number;
  maxRevenue?: number;
  sortField?: string;
  order?: "asc" | "desc";
  limit?: number;
}) : Promise<{ jobs: ApiJob[] }> {
  const r = await api.get("/data/jobs", { params });
  return r.data as { jobs: ApiJob[] };
}


export async function predict(params: {
  minRevenue?: number;
  maxRevenue?: number;
  sortField?: string;
  order?: "asc" | "desc";
  limit?: number;
}) : Promise<{ predictions: Prediction[]; count: number }> {
  const r = await api.post("/data/predict", null, { params });
  return r.data;
}
