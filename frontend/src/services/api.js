const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5001";

function authHeaders() {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handle(res) {
  if (res.status === 401) {
    localStorage.removeItem("token");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(data?.error || res.statusText), { data, status: res.status });
  return data;
}

export async function login(username, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  const data = await handle(res);
  return data; 
}

export async function syncJobs({ force = false, dateFrom, dateTo, dateExact } = {}) {
  const params = new URLSearchParams();
  if (force) params.set("force", "1");
  if (dateExact) params.set("DateIssued", dateExact);
  if (dateFrom) params.set("DateIssuedFrom", dateFrom);
  if (dateTo) params.set("DateIssuedTo", dateTo);

  const res = await fetch(`${API_BASE}/data/sync?${params.toString()}`, {
    headers: { ...authHeaders() }
  });
  return handle(res); 
}

export async function getJobs({ limit, minRevenue, maxRevenue, sortField = "revenue", order = "desc" } = {}) {
  const params = new URLSearchParams();
  if (limit) params.set("limit", String(limit));
  if (minRevenue) params.set("minRevenue", String(minRevenue));
  if (maxRevenue) params.set("maxRevenue", String(maxRevenue));
  if (sortField) params.set("sortField", sortField);
  if (order) params.set("order", order);

  const res = await fetch(`${API_BASE}/data/jobs?${params.toString()}`, {
    headers: { ...authHeaders() }
  });
  return handle(res); 
}

export async function predict({ limit } = {}) {
  const params = new URLSearchParams();
  if (limit) params.set("limit", String(limit));

  const res = await fetch(`${API_BASE}/data/predict?${params.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({})
  });
  return handle(res);
}