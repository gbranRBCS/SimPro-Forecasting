import express from "express";
import axios from "axios";
import jwt from "jsonwebtoken";

function axiosDiag(err) {
  return {
    message: err.message,
    code: err.code,
    method: err.config?.method,
    url: err.config?.url,
    status: err.response?.status,
    responseData: err.response?.data,
  };
}

const router = express.Router();

// Store sync state
let lastSyncTime = null;
const syncInterval = parseInt(process.env.SYNC_INTERVAL_MINUTES) || 60;

// Cache current job data
let cachedJobs = [];

// JWT Authentication
function authRequired(req, res, next) {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];
  if (!token) return res.sendStatus(401);

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) return res.sendStatus(403);
    req.user = user;
    next();
  });
}

// OAuth token (cached)
let cachedToken = null;
let tokenExpiryMs = 0;

async function getSimproToken() {
  const now = Date.now();
  if (cachedToken && now < tokenExpiryMs - 60_000) return cachedToken; // reuse

  const form = new URLSearchParams();
  form.append("grant_type", process.env.GRANT_TYPE || "client_credentials");
  if (process.env.SCOPE) form.append("scope", process.env.SCOPE);
  if (process.env.AUDIENCE) form.append("audience", process.env.AUDIENCE);

  try {
    const resp = await axios.post(process.env.TOKEN_URL, form.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      auth: {
        username: process.env.SIMPRO_CLIENT_ID,
        password: process.env.SIMPRO_CLIENT_SECRET,
      },
      timeout: 15_000,
    });

    const { access_token, expires_in } = resp.data || {};
    if (!access_token) throw new Error("No access_token in token response");
    cachedToken = access_token;
    tokenExpiryMs = now + ((expires_in || 3600) * 1000);
    return cachedToken;
  } catch (err) {
    const d = axiosDiag(err);
    console.error("getSimproToken failed:", d);
    throw new Error("OAuth token fetch failed");
  }
}

// simPRO fetch
async function fetchSimPROJobs(params = {}) {
  const base = process.env.SIMPRO_API_BASE;
  const companyId = process.env.SIMPRO_COMPANY_ID || "0";
  let jobsPath =
    (process.env.SIMPRO_JOBS_PATH || "/api/v1.0/companies/{COMPANY_ID}/jobs/")
      .replace("{COMPANY_ID}", companyId);

  if (!jobsPath.startsWith("/api/")) {
    jobsPath = "/api" + (jobsPath.startsWith("/") ? jobsPath : "/" + jobsPath);
  }
  if (!jobsPath.endsWith("/")) {
    jobsPath += "/";
  }

  const url = new URL(jobsPath, base).toString();
  const token = await getSimproToken();

  let allJobs = [];
  let page = 1;
  const pageSize = 250; // simPRO max page size

  while (true) {
    const pageParams = { ...params, page, pageSize };
    try {
      const r = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}` },
        params: pageParams,
        timeout: 20_000,
      });
      const jobs = Array.isArray(r.data) ? r.data : r.data?.jobs || [];
      allJobs = allJobs.concat(jobs);

      const resultCount = Number(r.headers["result-count"] || jobs.length);
      if (resultCount < pageSize) break;
      page += 1;
    } catch (err) {
      const d = axiosDiag(err);
      console.error("fetchSimPROJobs() failed:", d);
      const e = new Error("simPRO jobs fetch failed");
      e.detail = d;
      throw e;
    }
  }
  return allJobs;
}

router.get("/sync", authRequired, async (req, res) => {
  const now = Date.now();
  if (!req.query.force && lastSyncTime && now - lastSyncTime < syncInterval * 60 * 1000) {
    return res.json({ message: "Already recently synced", jobs: cachedJobs });
  }

  
  const { DateIssued, DateIssuedFrom, DateIssuedTo } = req.query;
  const params = {};
  if (DateIssued) params.DateIssued = DateIssued;
  if (DateIssuedFrom) params.DateIssuedFrom = DateIssuedFrom;
  if (DateIssuedTo) params.DateIssuedTo = DateIssuedTo;

  
  const concurrency = Number.parseInt(process.env.ENRICH_CONCURRENCY || "5", 10);

  try {
    const list = await fetchSimPROJobs(params);

    // enrich jobs
    let jobs = list;
    if (Array.isArray(list) && list.length) {
      const ids = list.map(j => j.ID).filter(Boolean);
      const results = await inChunks(ids, concurrency, (id) => fetchJobDetail(id));

      const byId = new Map(list.map(j => [j.ID, j]));
      results.forEach((r, idx) => {
        const id = ids[idx];
        if (r.status === "fulfilled") byId.set(id, r.value);
      });
      jobs = Array.from(byId.values());
    }

    // cache
    cachedJobs = jobs.map(normaliseJob);
    lastSyncTime = now;

    res.json({
      message: "Sync + enrich successful",
      count: cachedJobs.length,
      enriched: Array.isArray(cachedJobs) ? cachedJobs.length : 0,
      jobs: cachedJobs,
    });
  } catch (err) {
    res.status(500).json({
      error: "Failed to fetch simPRO job data.",
      detail: err.detail || { message: err.message },
    });
  }
});

function getRevenue(j) {
  if (typeof j?.revenue === "number") return j.revenue;
  const inc = j?.Total?.IncTax;
  return typeof inc === "number" ? inc : (inc ? Number(inc) : null);
}

function filterAndSortJobs(jobs, query) {
  let { sortField = "revenue", order = "asc", minRevenue, maxRevenue, limit } = query;
  let filtered = [...jobs];

  if (minRevenue) filtered = filtered.filter(j => {
    const r = getRevenue(j);
    return r != null && r >= parseFloat(minRevenue);
  });
  if (maxRevenue) filtered = filtered.filter(j => {
    const r = getRevenue(j);
    return r != null && r <= parseFloat(maxRevenue);
  });

  filtered.sort((a, b) => {
    const A = sortField === "revenue" ? (getRevenue(a) ?? -Infinity) : a[sortField];
    const B = sortField === "revenue" ? (getRevenue(b) ?? -Infinity) : b[sortField];
    if (order === "desc") return A > B ? -1 : 1;
    return A > B ? 1 : -1;
  });

  if (limit) {
    const n = Number.parseInt(limit, 10);
    if (!Number.isNaN(n) && n > 0) filtered = filtered.slice(0, n);
  }
  return filtered;
}

function stripHtml(s = " ") {
  return String(s.replace(/<[^>]*>/g, "").trim());
}


function num(x, d = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : d;
}

function daysBetween(a, b) {
  const da = a ? new Date(a) : null;
  const db = b ? new Date(b) : null;
  if (!da || !db || Number.isNaN(+da) || Number.isNaN(+db)) return null;
  const ms = db - da;
  return Math.round(ms / 86400000);
}

function normaliseJob(j) {
  const incTax = j?.Total?.IncTax ?? null;
  const revenue = typeof incTax === "number" ? incTax : (incTax ? Number(incTax) : null);

  // costs: prefer estimates, fall back to actual values, otherwise set to 0
  const mats = j?.Totals?.MaterialsCost;
  const resCost = j?.Totals?.ResourcesCost;
  const labor = resCost?.Labor;
  const overhead = resCost?.Overhead;
  const laborHours = resCost?.LaborHours;

  const materials_cost_est = num(mats?.Estimate, num(mats?.Actual, 0));
  const labor_cost_est = num(labor?.Estimate, num(labor?.Actual, 0));
  const overhead_est = num(overhead?.Actual, 0);
  const labor_hours_est = num(laborHours?.Estimate, num(laborHours?.Actual, 0));
  const cost_est_total = materials_cost_est + labor_cost_est + overhead_est;

  const descriptionText = stripHtml(j?.Description ?? " ");
  const profit_est = revenue != null ? revenue - cost_est_total : null;
  const margin_est = (revenue && revenue > 0 && profit_est != null) ? (profit_est / revenue) : null;

  let netMarginPct = margin_est ?? null;
  let profitability_class = null;
  if (netMarginPct != null) {
    if (netMarginPct >= 0.20) profitability_class = "High";
    else if (netMarginPct >= 0.05) profitability_class = "Medium";
    else profitability_class = "Low";
  }

  const dateIssued = j?.DateIssued ?? null;
  const dateDue = j?.DueDate ?? null;
  const age_days = dateIssued ? daysBetween(dateIssued, new Date()) : null;
  const due_in_days = (dateIssued && dateDue) ? daysBetween(dateIssued, dateDue) : null;

  const statusName = j?.Status?.Name ?? null;
  const stage = j?.Stage ?? null;
  const jobType = j?.Type ?? null;

  const txt = descriptionText.toLowerCase();
  const has_emergency = /emergency|urgent|callout|call-out|call out/.test(txt) ? 1 : 0;

  return {
    ...j,
    descriptionText,
    revenue,
    status: j?.Status ?? null,
    dateIssued,
    dateDue,
    customerName: j?.Customer?.CompanyName ?? null,
    siteName: j?.Site?.Name ?? null,
    jobType,

    // custom numeric features
    materials_cost_est,
    labor_cost_est,
    overhead_est,
    labor_hours_est,
    cost_est_total,
    profit_est,
    margin_est,
    netMarginPct,
    profitability_class,
    age_days,
    due_in_days,

    // custom categorical/text flags
    status_name: statusName,
    stage,
    desc_len: descriptionText.length,
    has_emergency,
  };
}

async function fetchJobDetail(jobId) {
  const base = process.env.SIMPRO_API_BASE;
  const companyId = process.env.SIMPRO_COMPANY_ID || "0";
const url = new URL(`/api/v1.0/companies/${companyId}/jobs/${jobId}`, base).toString();
  const token = await getSimproToken();

  const r = await axios.get(url, { headers: { Authorization: `Bearer ${token}` }, timeout: 20_000 });
  return r.data;
}

async function inChunks(ids, size, fn) {
  const out = [];
  for (let i = 0; i < ids.length; i += size) {
    const slice = ids.slice(i, i + size);
    const results = await Promise.allSettled(slice.map(fn));
    out.push(...results);
  }
  return out;
}

router.get("/jobs", authRequired, (req, res) => {
  let jobs = filterAndSortJobs(cachedJobs, req.query);

  // pagination
  const page = Number.parseInt(req.query.page, 10) || 1;
  const pageSize = Number.parseInt(req.query.pageSize, 10) || jobs.length;
  const start = (page - 1) * pageSize;
  const paged = jobs.slice(start, start + pageSize);

  res.json({
    jobs: paged,
    total: jobs.length,
    page,
    pageSize,
    totalPages: Math.ceil(jobs.length / pageSize),
  });
});

// forward cleaned jobs to ML microservice
router.post("/predict", authRequired, async (req, res) => {
  const jobsToSend = filterAndSortJobs(cachedJobs, req.query);
  const limit = Number.parseInt(req.query.limit, 10);
  const jobsToSendLimited = Number.isFinite(limit) && limit > 0 ? jobsToSend.slice(0, limit) : jobsToSend;
  try {
    const response = await axios.post(`${process.env.ML_URL}/predict`, { data: jobsToSendLimited });
    res.json(response.data);
  } catch (err) {
    console.error("Error forwarding jobs to ML service:", err.message, err.response?.data);
    res.status(500).json({ error: "ML Prediction failed" });
  }
});

// token check
router.get("/oauth-test", async (_req, res) => {
  try {
    const token = await getSimproToken();
    return res.status(200).json({
      ok: true,
      tokenPreview: token ? token.slice(0, 24) + "..." : null,
      base: process.env.SIMPRO_API_BASE || null,
    });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e?.message || "unknown error" });
  }
});

export default router;