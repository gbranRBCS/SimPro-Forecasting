import express from "express";
import axios from "axios";
import jwt from "jsonwebtoken";

import { clearJobs, getLatestIssuedDate, loadJobs, upsertJobs } from "../db/jobs.js";

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

// Cache current job data
let cachedJobs = loadJobs();
let lastSyncTime = cachedJobs.length ? Date.now() : null;
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

const DEFAULT_FULL_HISTORY_START = "2025-01-01T00:00:00Z";
const DEFAULT_UPDATE_HISTORY_START = "2010-01-01T00:00:00Z";

function parseStartDate(raw, fallback) {
  const source = raw || fallback;
  const isoLike = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(source)
    ? `${source}T00:00:00Z`
    : source;
  const parsed = new Date(isoLike);
  return Number.isFinite(+parsed) ? parsed : new Date(fallback);
}

const FULL_HISTORY_START = parseStartDate(
  process.env.SIMPRO_FULL_HISTORY_START || process.env.SIMPRO_HISTORY_START,
  DEFAULT_FULL_HISTORY_START,
);
const UPDATE_HISTORY_START = parseStartDate(
  process.env.SIMPRO_UPDATE_HISTORY_START || process.env.SIMPRO_MIN_ISSUED_DATE,
  DEFAULT_UPDATE_HISTORY_START,
);
const LOOKBACK_HOURS = Math.max(
  0,
  parseInt(process.env.SIMPRO_SYNC_LOOKBACK_HOURS || "24", 10) || 24,
);

function toDate(value) {
  if (!value) return null;
  const d = value instanceof Date ? new Date(value.getTime()) : new Date(value);
  if (!Number.isFinite(+d)) return null;
  return d;
}

function toIsoString(value) {
  const d = toDate(value);
  return d ? d.toISOString() : null;
}

function toIsoDate(value) {
  const d = toDate(value);
  if (!d) return null;
  const year = d.getUTCFullYear();
  const month = `${d.getUTCMonth() + 1}`.padStart(2, "0");
  const day = `${d.getUTCDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function computeIssuedFromOverride() {
  const latestIssued = getLatestIssuedDate();
  const latestDate = toDate(latestIssued);
  const lookbackMs = LOOKBACK_HOURS * 60 * 60 * 1000;
  if (!latestDate) {
    return toIsoDate(UPDATE_HISTORY_START);
  }

  const lowerBound = UPDATE_HISTORY_START.getTime();
  const from = new Date(latestDate.getTime() - lookbackMs);
  const clamped = new Date(Math.max(lowerBound, from.getTime()));
  return toIsoDate(clamped);
}

function buildJobRow(job) {
  if (!job) return null;
  const jobId = job?.ID ?? job?.Id ?? job?.id;
  if (jobId == null) return null;

  const issuedCandidate = job?.dateIssued ?? job?.DateIssued ?? null;
  const completedCandidate = job?.completed_date ?? job?.CompletedDate ?? job?.DateCompleted;
  const updatedCandidate =
    job?.DateUpdated ??
    job?.UpdatedDate ??
    job?.LastUpdated ??
    job?.updated_at ??
    job?.Updated ??
    completedCandidate ??
    issuedCandidate;

  // validate issued date is not in the future (data quality issue affecting syncing)
  let validatedIssuedDate = null;
  if (issuedCandidate) {
    const parsed = toDate(issuedCandidate);
    if (parsed) {
      const now = new Date();
      // allow a 1 day tolerance
      const oneDayAhead = new Date(now.getTime() + 24 * 60 * 60 * 1000);
      if (parsed.getTime() <= oneDayAhead.getTime()) {
        validatedIssuedDate = toIsoDate(parsed);
      } else {
        console.warn(
          `Job ${jobId}: issued date ${issuedCandidate} is in the future, nullifying to preserve sync integrity`
        );
      }
    }
  }

  return {
    job_id: String(jobId),
    issued_date: validatedIssuedDate,
    completed_date: toIsoString(completedCandidate),
    updated_at: toIsoString(updatedCandidate),
    payload: JSON.stringify(job),
  };
}

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
export async function fetchSimPROJobs(params = {}, { historical = false } = {}) {
  const url = buildJobsUrl();
  const token = await getSimproToken();

  const pageSizeEnv = parseInt(process.env.SIMPRO_PAGE_SIZE || "250", 10);
  const pageSize = Math.min(Number.isFinite(pageSizeEnv) ? pageSizeEnv : 250, 250);

  const pageDelayMs = parseInt(process.env.SIMPRO_PAGE_DELAY_MS || "0", 10) || 0;
  const pageJitterMs = parseInt(process.env.SIMPRO_PAGE_JITTER_MS || "0", 10) || 0;
  const maxSyncDefaultRaw = parseInt(process.env.SIMPRO_FETCH_MAX || "500", 10);
  const maxSyncDefault =
    Number.isFinite(maxSyncDefaultRaw) && maxSyncDefaultRaw > 0 ? maxSyncDefaultRaw : null;

  const allJobs = [];

  const queryParams = { ...params };

  // if historical window is requested, fetch from the oldest pages first.
  const historicalWindow = Boolean(historical);
  
  // for update sync, remove the fetch cap to ensure we get ALL jobs matching the date filter
  const hasDateFilter = params.DateIssued || params.DateIssuedFrom;
  const maxJobs = historicalWindow ? null : (hasDateFilter ? null : maxSyncDefault);

  if (params.DateIssuedFrom) {
    const fromDate = params.DateIssuedFrom;
    queryParams.DateIssued = `ge(${fromDate})`;
    delete queryParams.DateIssuedFrom;
    
    // Also filter by DateModified to catch recently updated jobs
    queryParams.DateModified = `ge(${fromDate})`;
    console.log(`[sync] Update mode: DateIssued=ge(${fromDate}), DateModified=ge(${fromDate}), no fetch cap`);
  }

  // default - newest-first forward paging 
  let page = 1;
  for (;;) {
    const pageParams = { ...queryParams, page, pageSize };
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

    if (maxJobs && allJobs.length >= maxJobs) {
      allJobs.length = maxJobs;
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

  const {
    DateIssued,
    DateIssuedFrom,
    DateIssuedTo,
    mode: syncModeRaw,
    syncMode: syncModeAlias,
  } = req.query;

  const requestedMode = (syncModeRaw || syncModeAlias || "").toString().toLowerCase();
  const syncMode = requestedMode === "full" ? "full" : "update";

  const params = {};
  let historicalRange = false;
  let incrementalFrom = null;

  if (syncMode === "full") {
    params.DateIssuedFrom = toIsoDate(FULL_HISTORY_START);
    historicalRange = true;
  } else {
    if (DateIssued) params.DateIssued = DateIssued;
    if (DateIssuedFrom) params.DateIssuedFrom = DateIssuedFrom;
    if (DateIssuedTo) params.DateIssuedTo = DateIssuedTo;
    historicalRange = Boolean(DateIssuedFrom || DateIssuedTo);

    if (!historicalRange && !DateIssued) {
      incrementalFrom = computeIssuedFromOverride();
      params.DateIssuedFrom = incrementalFrom;
    }
  }
  const concurrency = Number.parseInt(
    process.env.ENRICH_CONCURRENCY || "5",
    10,
  );

  syncing = true;
  try {
    console.log(`[sync] mode=${syncMode}, params=`, JSON.stringify(params, null, 2));
    const listRaw = await fetchSimPROJobs(params, { historical: historicalRange });

    const earliestAllowedMs = (
      syncMode === "full" ? FULL_HISTORY_START : UPDATE_HISTORY_START
    ).getTime();
    const incrementalFromDate =
      incrementalFrom != null ? toDate(`${incrementalFrom}T00:00:00Z`) : null;

    const rawCount = Array.isArray(listRaw) ? listRaw.length : 0;
    const list = Array.isArray(listRaw) ? listRaw : [];

    // enrich every job, but stop once enough enriched jobs are collected
    const keepMaxRaw = Number.parseInt(process.env.SIMPRO_SYNC_MAX || "500", 10);
    const keepMaxConfigured =
      Number.isFinite(keepMaxRaw) && keepMaxRaw > 0 ? keepMaxRaw : null;
    const KEEP_MAX = keepMaxConfigured;
  const getId = (j) => j?.ID ?? j?.Id ?? j?.id ?? null;
  const idsAll = list.map((j) => getId(j)).filter(Boolean);
  const byIdBase = new Map(list.map((j) => [getId(j), j]));

    const kept = [];
    let detailFailures = 0;
    for (
      let i = 0;
      i < idsAll.length && (!KEEP_MAX || kept.length < KEEP_MAX);
      i += concurrency
    ) {
      const slice = idsAll.slice(i, i + concurrency);
      // fetch details for this chunk (with retries inside fetchJobDetail)
      const settled = await Promise.allSettled(slice.map((id) => fetchJobDetail(id)));
      for (let s = 0; s < settled.length && (!KEEP_MAX || kept.length < KEEP_MAX); s++) {
        const id = slice[s];
        const base = byIdBase.get(id) || {};
        const detail = settled[s].status === "fulfilled" ? settled[s].value : {};
        if (settled[s].status === "rejected") {
          const reason = settled[s].reason;
          const diag =
            typeof reason === "object" && reason
              ? {
                  status: reason?.response?.status ?? null,
                  code: reason?.code ?? null,
                  message: reason?.message ?? String(reason),
                }
              : { message: String(reason) };
          detailFailures += 1;
          console.warn(`sync: job detail fetch failed for ID ${id}`, diag);
        }
        const merged = { ...base, ...(detail || {}) };
        const norm = normaliseJob(merged);
        if (!norm) continue;
        const dateCandidate =
          norm.dateIssued ??
          norm.dateDue ??
          norm.DateCompleted ??
          norm.completed_date ??
          null;
        const d = toDate(dateCandidate);
        if (d) {
          const ms = d.getTime();
          if (ms < earliestAllowedMs) continue;
          if (incrementalFromDate && ms < incrementalFromDate.getTime()) continue;
        } else if (syncMode === "full") {
          // drop undated in full mode to enforce 2025 data onwards
          continue;
        }

        if (norm.netMarginPct != null || norm.profitability_class != null) {
          kept.push(norm);
        }
      }
      // small gap between chunks to reduce burstiness (honour requestGapMs)
      if (requestGapMs > 0) await sleep(requestGapMs);
    }

    const filteredCount = kept.length;
    const excludedCount = list.length - filteredCount;
    if (excludedCount > 0) {
      console.info(
        `Sync: excluded ${excludedCount} jobs without netMarginPct/profitability_class (kept ${kept.length})`,
      );
    }

    const rows = kept
      .map((job) => buildJobRow(job))
      .filter(Boolean);

    if (syncMode === "full") {
      clearJobs();
    }

    if (rows.length) upsertJobs(rows);

    cachedJobs = loadJobs();
    lastSyncTime = Date.now();

    res.json({
      message: syncMode === "full" ? "Full sync complete" : "Update sync complete",
      count: cachedJobs.length,
      excluded: excludedCount,
      fetched: rawCount,
      filtered: filteredCount,
      detailFailures,
      upserted: rows.length,
      params,
      mode: syncMode,
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

export function refreshCachedJobsFromDb() {
  cachedJobs = loadJobs();
  return cachedJobs;
}

router.post("/predict", authRequired, async (req, res) => {
  const bodyJobs = Array.isArray(req.body?.jobs) ? req.body.jobs : null;
  const bodyJobIds = Array.isArray(req.body?.jobIds) ? req.body.jobIds : null;

  const limitFromBody = Number.parseInt(req.body?.limit, 10);
  const limitFromQuery = Number.parseInt(req.query?.limit, 10);
  const effectiveLimit = Number.isFinite(limitFromBody)
    ? limitFromBody
    : Number.isFinite(limitFromQuery)
    ? limitFromQuery
    : null;

  let jobsToSend = [];

  if (bodyJobs?.length) {
    jobsToSend = bodyJobs;
  } else if (bodyJobIds?.length) {
    const idSet = new Set(
      bodyJobIds
        .map((id) => {
          if (id === null || id === undefined) return null;
          try {
            return String(id);
          } catch (_) {
            return null;
          }
        })
        .filter(Boolean),
    );

    jobsToSend = cachedJobs.filter((job) => {
      const id = job?.ID ?? job?.id;
      return id != null && idSet.has(String(id));
    });
  } else {
    jobsToSend = filterAndSortJobs(cachedJobs, req.query);
  }

  const jobsToSendLimited =
    Number.isFinite(effectiveLimit) && effectiveLimit > 0
      ? jobsToSend.slice(0, effectiveLimit)
      : jobsToSend;

  if (!jobsToSendLimited?.length) {
    return res.json({ predictions: [], count: 0, model_loaded: false });
  }

  if (!process.env.ML_URL) {
    return res.status(500).json({
      error: "ML service URL not configured",
    });
  }

  try {
    console.info(
      `[predict] forwarding ${jobsToSendLimited.length} jobs to ML at ${process.env.ML_URL}/predict`,
    );
    const response = await axios.post(`${process.env.ML_URL}/predict`, {
      data: jobsToSendLimited,
    });
    return res.json(response.data);
  } catch (err) {
    console.error(
      "Error forwarding jobs to ML service:",
      axiosDiag(err),
      "jobsCount:",
      jobsToSendLimited.length,
    );

    const mlStatus = err.response?.status ?? null;
    const mlBody = err.response?.data ?? { message: err.message };

    return res.status(502).json({
      error: "ML Prediction failed",
      mlStatus,
      mlBody,
    });
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
