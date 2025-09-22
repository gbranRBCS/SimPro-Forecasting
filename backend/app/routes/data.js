import express from "express";
import axios from "axios";
import jwt from "jsonwebtoken";

const router = express.Router();

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

// Store sync state
let lastSyncTime = null;
const syncInterval = parseInt(process.env.SYNC_INTERVAL_MINUTES, 10) || 60;

// Cache current job data
let cachedJobs = [];
let syncing = false;

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
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withRetries(fn, { maxRetries = 5, baseDelay = 1000 } = {}) {
  let attempt = 0;
  for (;;) {
    try {
      return await fn();
    } catch (err) {
      const status = err?.response?.status;
      const code = err?.code;
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      const retryableStatus =
        status === 429 || status === 502 || status === 503 || status === 504;
      const retryableCode = code === "ECONNRESET" || code === "ETIMEDOUT";

      if ((retryableStatus || retryableCode) && attempt < maxRetries) {
        const jitterMs = Number.parseInt(
          process.env.SIMPRO_RETRY_JITTER_MS || "250",
          10,
        );
        const base = Number.isFinite(retryAfterSec)
          ? retryAfterSec * 1000
          : baseDelay * Math.pow(2, attempt);
        const delay = base + Math.floor(Math.random() * Math.max(0, jitterMs));
        console.warn(
          `simPRO retry: status=${status ?? code}, waiting ${delay}ms (attempt ${attempt + 1}/${maxRetries})`,
        );
        await sleep(delay);
        attempt += 1;
        continue;
      }

      console.error("simPRO request failed:", axiosDiag(err));
      throw err;
    }
  }
}

const requestGapMs = Math.max(
  0,
  parseInt(process.env.SIMPRO_REQUEST_INTERVAL_MS || "500", 10) || 0,
);
const throttlePenaltyMs = Math.max(
  requestGapMs,
  parseInt(process.env.SIMPRO_THROTTLE_PENALTY_MS || `${requestGapMs * 4}`, 10) ||
    requestGapMs,
);
let simproQueue = Promise.resolve();
let nextAllowedRequestAt = 0;

function scheduleSimproRequest(makeRequest, label = "simpro") {
  const run = simproQueue.then(async () => {
    const now = Date.now();
    const waitFor = Math.max(0, nextAllowedRequestAt - now);
      await sleep(waitFor);

    try {
      const result = await makeRequest();
      nextAllowedRequestAt = Date.now() + requestGapMs;
      return result;
    } catch (err) {
      const status = err?.response?.status;
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      if (Number.isFinite(retryAfterSec) && retryAfterSec > 0) {
        nextAllowedRequestAt = Date.now() + retryAfterSec * 1000;
      } else if (status === 429) {
        nextAllowedRequestAt = Date.now() + throttlePenaltyMs;
      } else {
        nextAllowedRequestAt = Date.now() + requestGapMs;
      }
      throw err;
    }
  });

  simproQueue = run.catch(() => {});
  return run;
}

// OAuth token (cached)
let cachedToken = null;
let tokenExpiryMs = 0;

async function getSimproToken() {
  const now = Date.now();
  if (cachedToken && now < tokenExpiryMs - 60_000) return cachedToken;

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
      { maxRetries: 4, baseDelay: 1500 },
    );

    const { access_token, expires_in } = resp.data || {};
    if (!access_token) throw new Error("No access_token in token response");
    cachedToken = access_token;
    tokenExpiryMs = now + (expires_in || 3600) * 1000;
    return cachedToken;
  } catch (err) {
    const d = axiosDiag(err);
    console.error("getSimproToken failed:", d);
    throw new Error("OAuth token fetch failed");
  }
}

function buildJobsUrl() {
  const base = process.env.SIMPRO_API_BASE || process.env.SIMPRO_API_URL;
  if (!base) throw new Error("SIMPRO_API_BASE is not configured");
  const companyId = process.env.SIMPRO_COMPANY_ID || "0";
  let jobsPath = (
    process.env.SIMPRO_JOBS_PATH || "/api/v1.0/companies/{COMPANY_ID}/jobs/"
  ).replace("{COMPANY_ID}", companyId);

  if (!jobsPath.startsWith("/")) jobsPath = `/${jobsPath}`;
  if (!jobsPath.endsWith("/")) jobsPath += "/";

  return new URL(jobsPath, base).toString();
}

// simPRO fetch
export async function fetchSimPROJobs(params = {}) {
  const url = buildJobsUrl();
  const token = await getSimproToken();

  const pageSizeEnv = parseInt(process.env.SIMPRO_PAGE_SIZE || "250", 10);
  const pageSize = Math.min(Number.isFinite(pageSizeEnv) ? pageSizeEnv : 250, 250);

  const pageDelayMs = parseInt(process.env.SIMPRO_PAGE_DELAY_MS || "0", 10) || 0;
  const pageJitterMs = parseInt(process.env.SIMPRO_PAGE_JITTER_MS || "0", 10) || 0;

  // LIMIT FOR TESTING: stop after fetching up to determined maximum
  const MAX_SYNC_JOBS = 200;

  const allJobs = [];

  // if historical window is requested, fetch from the oldest pages first.
  const historicalWindow = !!(params?.DateIssuedFrom || params?.DateIssuedTo);

  if (historicalWindow) {
    // First request to read total pages from headers
    const headResp = await withRetries(
      () =>
        axios.get(url, {
          headers: { Authorization: `Bearer ${token}` },
          params: { ...params, page: 1, pageSize },
          timeout: 20_000,
        }),
      { maxRetries: 5, baseDelay: 1000 }
    );

    const totalPages =
      parseInt(headResp.headers["result-pages"] ?? headResp.headers["Result-Pages"] ?? "1", 10) || 1;

    let page = totalPages;
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

      if (allJobs.length >= MAX_SYNC_JOBS) {
        allJobs.length = MAX_SYNC_JOBS;
        break;
      }

      page -= 1;
      if (page < 1) break;

      if (pageDelayMs > 0 || pageJitterMs > 0) {
        const delay = pageDelayMs + Math.floor(Math.random() * Math.max(0, pageJitterMs));
        await sleep(delay);
      }
    }

    return allJobs;
  }

  // default - newest-first forward paging 
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

    if (allJobs.length >= MAX_SYNC_JOBS) {
      allJobs.length = MAX_SYNC_JOBS;
      break;
    }

    if (jobs.length < pageSize) break;
    page += 1;

    if (pageDelayMs > 0 || pageJitterMs > 0) {
      const delay = pageDelayMs + Math.floor(Math.random() * Math.max(0, pageJitterMs));
      await sleep(delay);
    }
  }

  return allJobs;
}

function getRevenue(j) {
  if (typeof j?.revenue === "number") return j.revenue;
  const inc = j?.Total?.IncTax;
  return typeof inc === "number" ? inc : inc ? Number(inc) : null;
}

function filterAndSortJobs(jobs, query) {
  let {
    sortField = "revenue",
    order = "asc",
    minRevenue,
    maxRevenue,
    limit,
    page,
    pageSize,
  } = query;
  let filtered = [...jobs];

  if (minRevenue)
    filtered = filtered.filter((j) => {
      const r = getRevenue(j);
      return r != null && r >= parseFloat(minRevenue);
    });
  if (maxRevenue)
    filtered = filtered.filter((j) => {
      const r = getRevenue(j);
      return r != null && r <= parseFloat(maxRevenue);
    });

  filtered.sort((a, b) => {
    const A =
      sortField === "revenue" ? (getRevenue(a) ?? -Infinity) : a[sortField];
    const B =
      sortField === "revenue" ? (getRevenue(b) ?? -Infinity) : b[sortField];
    if (order === "desc") return A > B ? -1 : 1;
    return A > B ? 1 : -1;
  });

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
  const revenue =
    typeof incTax === "number" ? incTax : incTax ? Number(incTax) : null;

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
  const margin_est =
    revenue && revenue > 0 && profit_est != null ? profit_est / revenue : null;

  let netMarginPct = margin_est ?? null;
  let profitability_class = null;
  if (netMarginPct != null) {
    if (netMarginPct >= 0.2) profitability_class = "High";
    else if (netMarginPct >= 0.05) profitability_class = "Medium";
    else profitability_class = "Low";
  }

  const dateIssued = j?.DateIssued ?? null;
  const dateDue = j?.DueDate ?? null;
  const age_days = dateIssued ? daysBetween(dateIssued, new Date()) : null;
  const due_in_days =
    dateIssued && dateDue ? daysBetween(dateIssued, dateDue) : null;

  const statusName = j?.Status?.Name ?? null;
  const stage = j?.Stage ?? null;
  const jobType = j?.Type ?? null;

  const txt = descriptionText.toLowerCase();
  const has_emergency = /emergency|urgent|callout|call-out|call out/.test(txt)
    ? 1
    : 0;

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
    status_name: statusName,
    stage,
    desc_len: descriptionText.length,
    has_emergency,
  };
}

async function fetchJobDetail(jobId) {
  const base = process.env.SIMPRO_API_BASE || process.env.SIMPRO_API_URL;
  if (!base) throw new Error("SIMPRO_API_BASE is not configured");
  const companyId = process.env.SIMPRO_COMPANY_ID || "0";
  const url = new URL(
    `/api/v1.0/companies/${companyId}/jobs/${jobId}`,
    base,
  ).toString();
  const token = await getSimproToken();

  const resp = await withRetries(
    () =>
      scheduleSimproRequest(
        () =>
          axios.get(url, {
            headers: { Authorization: `Bearer ${token}` },
            timeout: 20_000,
          }),
        `job-${jobId}`,
      ),
    { maxRetries: 4, baseDelay: 1200 },
  );

  return resp.data;
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

router.get("/sync", authRequired, async (req, res) => {
  if (syncing)
    return res.status(409).json({ message: "Sync already in progress" });

  const now = Date.now();
  if (
    !req.query.force &&
    lastSyncTime &&
    now - lastSyncTime < syncInterval * 60 * 1000 &&
    cachedJobs.length
  ) {
    return res.json({ message: "Already recently synced", jobs: cachedJobs });
  }

  const { DateIssued, DateIssuedFrom, DateIssuedTo } = req.query;
  const params = {};
  if (DateIssued) params.DateIssued = DateIssued;
  if (DateIssuedFrom) params.DateIssuedFrom = DateIssuedFrom;
  if (DateIssuedTo) params.DateIssuedTo = DateIssuedTo;


  const concurrency = Number.parseInt(
    process.env.ENRICH_CONCURRENCY || "5",
    10,
  );

  syncing = true;
  try {
    const list = await fetchSimPROJobs(params);

    // enrich every job, but stop once enough enriched jobs are collected
    const KEEP_MAX = Number.parseInt(process.env.SIMPRO_SYNC_MAX || "100", 10);
    const idsAll = Array.isArray(list) ? list.map((j) => j.ID).filter(Boolean) : [];
    const byIdBase = new Map(Array.isArray(list) ? list.map((j) => [j.ID, j]) : []);

    const kept = [];
    for (let i = 0; i < idsAll.length && kept.length < KEEP_MAX; i += concurrency) {
      const slice = idsAll.slice(i, i + concurrency);
      // fetch details for this chunk (with retries inside fetchJobDetail)
      const settled = await Promise.allSettled(slice.map((id) => fetchJobDetail(id)));
      for (let s = 0; s < settled.length && kept.length < KEEP_MAX; s++) {
        const id = slice[s];
        const base = byIdBase.get(id) || {};
        const detail = settled[s].status === "fulfilled" ? settled[s].value : {};
        const merged = { ...base, ...(detail || {}) };
        const norm = normaliseJob(merged);
        if (norm && (norm.netMarginPct != null || norm.profitability_class != null)) {
          kept.push(norm);
        }
      }
      // small gap between chunks to reduce burstiness (honour requestGapMs)
      if (requestGapMs > 0) await sleep(requestGapMs);
    }

    const excludedCount = Array.isArray(list) ? list.length - kept.length : 0;
    if (excludedCount > 0) {
      console.info(`Sync: excluded ${excludedCount} jobs without netMarginPct/profitability_class (kept ${kept.length})`);
    }

    cachedJobs = kept;
     lastSyncTime = now;

    res.json({
      message: "Sync + enrich successful",
      count: cachedJobs.length,
      excluded: excludedCount,
      jobs: cachedJobs,
    });
  } catch (err) {
    console.error("Sync failed:", axiosDiag(err) || err);
    res.status(502).json({
      error: "Failed to fetch simPRO data",
      detail: axiosDiag(err),
    });
  } finally {
    syncing = false;
  }
});

router.get("/jobs", authRequired, (req, res) => {
  const jobs = filterAndSortJobs(cachedJobs, req.query);

  const page = Number.parseInt(req.query.page, 10) || 1;
  const pageSize = Number.parseInt(req.query.pageSize, 10) || jobs.length || 1;
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

router.post("/predict", authRequired, async (req, res) => {
  const jobsToSend = filterAndSortJobs(cachedJobs, req.query);
  const limit = Number.parseInt(req.query.limit, 10);
  const jobsToSendLimited =
    Number.isFinite(limit) && limit > 0
      ? jobsToSend.slice(0, limit)
      : jobsToSend;

  try {
    const response = await axios.post(`${process.env.ML_URL}/predict`, {
      data: jobsToSendLimited,
    });
    res.json(response.data);
  } catch (err) {
    console.error(
      "Error forwarding jobs to ML service:",
      err.message,
      err.response?.data,
    );
    res.status(500).json({ error: "ML Prediction failed" });
  }
});

router.get("/oauth-test", async (_req, res) => {
  try {
    const token = await getSimproToken();
    return res.status(200).json({
      ok: true,
      tokenPreview: token ? `${token.slice(0, 24)}...` : null,
      base: process.env.SIMPRO_API_BASE || process.env.SIMPRO_API_URL || null,
    });
  } catch (err) {
    return res
      .status(500)
      .json({ ok: false, error: err?.message || "unknown error" });
  }
});

export default router;
