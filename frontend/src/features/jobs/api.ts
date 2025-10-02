import api from "../../lib/api";

export type ApiJob = Record<string, any>;
export type Prediction = {
  jobId: number | string | null;
  // trained model fields
  class?: "Low" | "Medium" | "High";
  confidence?: number;
  // fallback heuristic fields (older shape)
  profitable?: boolean;
  probability?: number;
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


export async function getJobs(params: any) {
  const res = await api.get("/data/jobs", { params });
  return res.data;
}


export async function predict(body: Record<string, any>) {
  const payload: Record<string, any> = {};
  Object.entries(body || {}).forEach(([key, value]) => {
    if (value === undefined) return;
    payload[key] = value;
  });

  const res = await api.post("/data/predict", payload);
  return res.data;
}
