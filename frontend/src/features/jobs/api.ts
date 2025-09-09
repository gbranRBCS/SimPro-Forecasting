import api from "../../lib/api";

export async function syncJobs(params?: { from?: string; to?: string }) {
  const q = new URLSearchParams();
  q.set("force", "1");
  if (params?.from) q.set("DateIssuedFrom", params.from);
  if (params?.to)   q.set("DateIssuedTo", params.to);
  const { data } = await api.get(`/data/sync?${q.toString()}`);
  return data; // { message, jobs }
}

export async function getJobs(params?: {
  minRevenue?: number; maxRevenue?: number;
  sortField?: string; order?: "asc" | "desc"; limit?: number;
}) {
  const q = new URLSearchParams();
  if (params?.minRevenue != null) q.set("minRevenue", String(params.minRevenue));
  if (params?.maxRevenue != null) q.set("maxRevenue", String(params.maxRevenue));
  if (params?.sortField) q.set("sortField", params.sortField);
  if (params?.order)     q.set("order", params.order);
  if (params?.limit)     q.set("limit", String(params.limit));
  const { data } = await api.get(`/data/jobs?${q.toString()}`);
  return data.jobs as any[];
}

export async function predict(params?: {
  minRevenue?: number; maxRevenue?: number;
  sortField?: string; order?: "asc" | "desc"; limit?: number;
}) {
  const q = new URLSearchParams();
  if (params?.minRevenue != null) q.set("minRevenue", String(params.minRevenue));
  if (params?.maxRevenue != null) q.set("maxRevenue", String(params.maxRevenue));
  if (params?.sortField) q.set("sortField", params.sortField);
  if (params?.order)     q.set("order", params.order);
  if (params?.limit)     q.set("limit", String(params.limit));
  const { data } = await api.post(`/data/predict?${q.toString()}`);
  return data; // { predictions: [{ score }], count }
}