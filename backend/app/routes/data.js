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

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// generic retry with rate-limit handling, backs off exponentially
async function withRetries(fn, { maxRetries = 5, baseDelay = 1000 } = {}) {
  let attempt = 0;
  for (;;) {
    try {
      return await fn();
    } catch (err) {
      const status = err?.response?.status;
      const code = err?.code;
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      const retryableStatus = status === 429 || status === 502 || status === 503 || status === 504;
      const retryableCode = code === "ECONNRESET" || code === "ETIMEDOUT";
      if ((retryableStatus || retryableCode) && attempt < maxRetries) {
        const jitterMs = Number.parseInt(process.env.SIMPRO_RETRY_JITTER_MS || "250", 10);
        const base = Number.isFinite(retryAfterSec)
          ? retryAfterSec * 1000
          : baseDelay * Math.pow(2, attempt);
        const delay = base + Math.floor(Math.random() * Math.max(0, jitterMs));
        console.warn(
          `simPRO retry: status=${status ?? code}, waiting ${delay}ms (attempt ${attempt + 1}/${maxRetries})`
        );
        await sleep(delay);
        attempt++;
        continue;
      }
      console.error("simPRO request failed:", axiosDiag(err));
      throw err;
    }
  }
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
    const resp = await withRetries(
      () =>
        axios.post(process.env.TOKEN_URL, form.toString(), {
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          auth: {
            username: process.env.SIMPRO_CLIENT_ID,
            password: process.env.SIMPRO_CLIENT_SECRET,
          },
          timeout: 15_000,
        }),
      { maxRetries: 4, baseDelay: 1500 }
    );

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
// exported for tests if needed
export async function fetchSimPROJobs(params = {}) {
  const base = process.env.SIMPRO_API_BASE || process.env.SIMPRO_API_URL;
  const companyId = process.env.SIMPRO_COMPANY_ID || "0";
  const token = await getSimproToken();

  // max 250 per docs; allow override (capped at 250)
  const pageSizeEnv = parseInt(process.env.SIMPRO_PAGE_SIZE || "250", 10);
  const pageSize = Math.min(Number.isFinite(pageSizeEnv) ? pageSizeEnv : 250, 250);

  const pageDelayMs = parseInt(process.env.SIMPRO_PAGE_DELAY_MS || "200", 10) || 0;
  const pageJitterMs = parseInt(process.env.SIMPRO_PAGE_JITTER_MS || "150", 10) || 0;

  const url = new URL(`/api/v1.0/companies/${companyId}/jobs/`, base).toString();

  const allJobs = [];
  let page = 1;

  for (;;) {
    const pageParams = { ...params, page, pageSize };
    const resp = await withRetries(
      () =>
        axios.get(url, {
          headers: { Authorization: `Bearer ${token}` },
          params: pageParams,
          timeout: 20_000,
        }),
      { maxRetries: 5, baseDelay: 1000 }
    );

    const jobs = Array.isArray(resp.data) ? resp.data : resp.data?.jobs || [];

 
    if (!jobs.length) break;

    allJobs.push(...jobs);


    if (jobs.length < pageSize) break;

    page += 1;

    // soft throttle between pages
    if (pageDelayMs > 0 || pageJitterMs > 0) {
      const delay = pageDelayMs + Math.floor(Math.random() * Math.max(0, pageJitterMs));
      await sleep(delay);
    }
  }

  return allJobs;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// generic retry with rate-limit handling, backs off exponentially
async function withRetries(fn, { maxRetries = 5, baseDelay = 1000 } = {}) {
  let attempt = 0;
  for (;;) {
    try {
      return await fn();
    } catch (err) {
      const status = err?.response?.status;
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      if (status === 429 && attempt < maxRetries) {
        const delay = Number.isFinite(retryAfterSec)
          ? retryAfterSec * 1000
          : baseDelay * Math.pow(2, attempt);
        console.warn(
          `simPRO 429 - retrying in ${Math.round(delay)}ms (attempt ${
            attempt + 1
          }/${maxRetries})`
        );
        await sleep(delay);
        attempt++;
        continue;
      }
      console.error("simPRO request failed:", axiosDiag(err));
      throw err;
    }
  }
}

let syncing = false;

router.get("/sync", authRequired, async (req, res) => {
  if (syncing) return res.status(409).json({ message: "Sync already in progress" });

  const now = Date.now();
  if (lastSyncTime && now - lastSyncTime < syncInterval * 60 * 1000 && cachedJobs.length) {
    return res.json({ message: "Already recently synced", jobs: cachedJobs });
  }

  syncing = true;
  try {
    const list = await fetchSimPROJobs(/* optional filters */);
    cachedJobs = list;
    lastSyncTime = now;
    res.json({ message: "Sync was successful", jobs: cachedJobs });
  } catch (err) {
    console.error("Sync failed:", axiosDiag(err) || err);
    res.status(502).json({ error: "Failed to fetch simPRO data" });
  } finally {
    syncing = false;
  }
});

function getRevenue(j) {
  if (typeof j?.revenue === "number") return j.revenue;
  const inc = j?.Total?.IncTax;
  return typeof inc === "number" ? inc : (inc ? Number(inc) : null);
}

function filterAndSortJobs(jobs, query) {
  let { sortField = "revenue", order = "asc", minRevenue, maxRevenue, limit, page, pageSize } = query;
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

  // only limit if no pagination
  if (!page && !pageSize && limit) {
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

  const r = await withRetries(() =>
    axios.get(url, { headers: { Authorization: `Bearer ${token}` }, timeout: 20_000 })
  );
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