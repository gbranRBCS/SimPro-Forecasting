import express from "express";
import axios from "axios";
import jwt from "jsonwebtoken";

import { clearJobs, getLatestIssuedDate, loadJobs, upsertJobs } from "../db/jobs.js";

const router = express.Router();

/**
Helper to extract useful error details from an Axios error object
Returns clean object with status, URL, method, and response data
 */
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

// In-memory cache for jobs to avoid hitting the DB constantly.
let cachedJobs = loadJobs();

// Track when we last synced with SimPRO.
let lastSyncTime = cachedJobs.length ? Date.now() : null;

// Flag to prevent overlapping sync operations.
let syncing = false;

// Middleware to protect routes with JWT authentication.
function authRequired(req, res, next) {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];
  
  // No token -> Access Denied
  if (!token) return res.sendStatus(401);

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    // If the token is invalid or expired, forbid access
    if (err) return res.sendStatus(403);
    
    // Attach user info to request and proceed
    req.user = user;
    next();
  });
}

// Simple pause function
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
Wraps an async function with retry logic
Handles rate limits (429) & network glitches
Uses exponential backoff with to avoid overwhelming the API
 */
async function withRetries(fn, { maxRetries = 5, baseDelay = 1000 } = {}) {
  let attempt = 0;
  
  while (true) {
    try {
      // Try to run the function
      return await fn();
    } catch (err) {
      // Failed, Check if we should retry
      const status = err?.response?.status;
      const code = err?.code;
      
      // If the server wait time provided, utilise it
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      
      // Common temporary failure codes.
      const isRateLimited = status === 429;
      const isServerIssues = status === 502 || status === 503 || status === 504;
      const isNetworkIssues = code === "ECONNRESET" || code === "ETIMEDOUT";

      const shouldRetry = (isRateLimited || isServerIssues || isNetworkIssues) && attempt < maxRetries;

      if (shouldRetry) {
        // Calculate wait time: Base delay * 2^attempt (exponential) + random jitter.
        const jitterMs = parseInt(process.env.SIMPRO_RETRY_JITTER_MS || "250", 10);
        
        let waitTimeMs = 0;
        if (Number.isFinite(retryAfterSec)) {
          // Respect the server's request if it gave a Retry-After header.
          waitTimeMs = retryAfterSec * 1000;
        } else {
          // Otherwise, back off exponentially.
          const exponentialBackoff = baseDelay * Math.pow(2, attempt);
          const randomJitter = Math.floor(Math.random() * Math.max(0, jitterMs));
          waitTimeMs = exponentialBackoff + randomJitter;
        }

        console.warn(
          `simPRO retry: status=${status ?? code}, waiting ${waitTimeMs}ms (attempt ${attempt + 1}/${maxRetries})`
        );
        
        await sleep(waitTimeMs);
        attempt++;
        continue; // Loop back and try again.
      }

      // If we shouldn't retry (or ran out of retries), log and throw again.
      console.error("simPRO request failed:", axiosDiag(err));
      throw err;
    }
  }
}

// --- Rate Limiting & Throttling Configuration ---
// Minimum gap between requests to avoid hitting rate limits.
const requestGapMs = Math.max(
  0,
  parseInt(process.env.SIMPRO_REQUEST_INTERVAL_MS || "500", 10) || 0
);

// Penalty wait time if we hit a 429 (Too Many Requests).
const throttlePenaltyMs = Math.max(
  requestGapMs,
  parseInt(process.env.SIMPRO_THROTTLE_PENALTY_MS || `${requestGapMs * 4}`, 10) || requestGapMs
);

// Queue promise to chain requests sequentially.
let simproQueue = Promise.resolve();
// Timestamp when the next request is allowed to fire.
let nextAllowedRequestAt = 0;


// --- History & Date Configuration ---
const DEFAULT_FULL_HISTORY_START = "2025-01-01T00:00:00Z";
const DEFAULT_UPDATE_HISTORY_START = "2010-01-01T00:00:00Z";

/**
Parses a date string into a Date object.
Returns the fallback date if parsing fails.
Ensures strict ISO formatting for input like "YYYY-MM-DD" with RegEx.
 */
function parseStartDate(raw, fallback) {
  const source = raw || fallback;
  // If user provided a simple date "2023-01-01", treat it as UTC midnight.
  const isoLike = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(source)
    ? `${source}T00:00:00Z`
    : source;
    
  const parsed = new Date(isoLike);
  return Number.isFinite(+parsed) ? parsed : new Date(fallback);
}

const FULL_HISTORY_START = parseStartDate(
  process.env.SIMPRO_FULL_HISTORY_START || process.env.SIMPRO_HISTORY_START,
  DEFAULT_FULL_HISTORY_START
);

const UPDATE_HISTORY_START = parseStartDate(
  process.env.SIMPRO_UPDATE_HISTORY_START || process.env.SIMPRO_MIN_ISSUED_DATE,
  DEFAULT_UPDATE_HISTORY_START
);

// How far back to look when doing an incrmental sync (hours).
const LOOKBACK_HOURS = Math.max(
  0,
  parseInt(process.env.SIMPRO_SYNC_LOOKBACK_HOURS || "24", 10) || 24
);

// --- Date Helpers ---

function toDate(value) {
  if (!value) return null;
  const d = value instanceof Date ? new Date(value.getTime()) : new Date(value);
  return Number.isFinite(+d) ? d : null;
}

function toIsoString(value) {
  const d = toDate(value);
  return d ? d.toISOString() : null;
}

// Returns just the YYYY-MM-DD part.
function toIsoDate(value) {
  const d = toDate(value);
  if (!d) return null;
  const year = d.getUTCFullYear();
  const month = `${d.getUTCMonth() + 1}`.padStart(2, "0");
  const day = `${d.getUTCDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/**
Determines the start date for an incremental sync.
Looks at the most recent job, subtracts the lookback window, and clamps it to the minimum history start date.
 */
function computeIssuedFromOverride() {
  const latestIssued = getLatestIssuedDate();
  const latestDate = toDate(latestIssued);
  const lookbackMs = LOOKBACK_HOURS * 60 * 60 * 1000;
  
  // If we have no local data, start from the configured beginning of time.
  if (!latestDate) {
    return toIsoDate(UPDATE_HISTORY_START);
  }

  const lowerBound = UPDATE_HISTORY_START.getTime();
  const from = new Date(latestDate.getTime() - lookbackMs);
  
  // Ensure we don't go back further than allowed.
  const clamped = new Date(Math.max(lowerBound, from.getTime()));
  return toIsoDate(clamped);
}

/**
Extracts and validates key fields from a SimPRO job object to prepare it for database storage.
 */
function buildJobRow(job) {
  if (!job) return null;
  
  // SimPRO is inconsistent with casing sometimes, so check multiple variations.
  const jobId = job?.ID ?? job?.Id ?? job?.id;
  if (jobId == null) return null;

  const issuedCandidate = job?.dateIssued ?? job?.DateIssued ?? null;
  const completedCandidate = job?.completed_date ?? job?.CompletedDate ?? job?.DateCompleted;
  
  // Try to find ANY date that indicates when this job was valuable/updated.
  const updatedCandidate =
    job?.DateUpdated ??
    job?.UpdatedDate ??
    job?.LastUpdated ??
    job?.updated_at ??
    job?.Updated ??
    completedCandidate ??
    issuedCandidate;

  // Validate issued date is not in the future
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
          `Job ${jobId}: issued date ${issuedCandidate} is in the future. Nullifying to protect data integrity.`
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

/**
Schedules a request to SimPRO, ensuring we don't exceed rate limits.
All requests through this function are sequential.
 */
function scheduleSimproRequest(makeRequest, label = "simpro") {
  const run = simproQueue.then(async () => {
    // Check if we need to wait before firing the next request.
    const now = Date.now();
    const waitFor = Math.max(0, nextAllowedRequestAt - now);
    if (waitFor > 0) {
      await sleep(waitFor);
    }

    try {
      const result = await makeRequest();
      // Success - Set the next allowed time based on standard gap.
      nextAllowedRequestAt = Date.now() + requestGapMs;
      return result;
    } catch (err) {
      // Failure logic.
      const status = err?.response?.status;
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      
      // Update the delay for the next request in the queue according to result
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

  // Keep the queue chain alive, even if this request fails.
  simproQueue = run.catch(() => {});
  return run;
}

// Cached token
let cachedToken = null;
// Time (in ms) when the current token expires.
let tokenExpiryMs = 0;

/**
Retrieves a OAuth 2.0 access token for SimPRO.
Uses client credentials flow. Returns cached token if still valid.
 */
async function getSimproToken() {
  const now = Date.now();
  // Reuse token if it has more than 60 seconds of life left.
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
      { maxRetries: 4, baseDelay: 1500 }
    );

    const { access_token, expires_in } = resp.data || {};
    if (!access_token) throw new Error("No access_token in token response");
    
    cachedToken = access_token;
    // expires_in is usually in seconds. Default to 1 hour if missing.
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

  // Normalize slashes
  if (!jobsPath.startsWith("/")) jobsPath = `/${jobsPath}`;
  if (!jobsPath.endsWith("/")) jobsPath += "/";

  return new URL(jobsPath, base).toString();
}

/**
Fetches specific jobs or a list of jobs from SimPRO.
Handles pagination automatically.
 */
export async function fetchSimPROJobs(params = {}, { historical = false } = {}) {
  const url = buildJobsUrl();
  const token = await getSimproToken();

  // Pagination config
  const pageSizeEnv = parseInt(process.env.SIMPRO_PAGE_SIZE || "250", 10);
  const pageSize = Math.min(Number.isFinite(pageSizeEnv) ? pageSizeEnv : 250, 250);
  const pageDelayMs = parseInt(process.env.SIMPRO_PAGE_DELAY_MS || "0", 10) || 0;
  const pageJitterMs = parseInt(process.env.SIMPRO_PAGE_JITTER_MS || "0", 10) || 0;
  
  // Safety cap to prevent memory overflow during large syncs
  const maxSyncDefaultRaw = parseInt(process.env.SIMPRO_FETCH_MAX || "500", 10);
  const maxSyncDefault =
    Number.isFinite(maxSyncDefaultRaw) && maxSyncDefaultRaw > 0 ? maxSyncDefaultRaw : null;

  const allJobs = [];
  const queryParams = { ...params };
  
  // Update mode requires fetching everything that changed, no matter how many.
  // Historical/Regular syncs might be capped.
  const hasDateFilter = params.DateIssued || params.DateIssuedFrom;
  const shouldCapResults = !historical && !hasDateFilter;
  const maxJobs = shouldCapResults ? maxSyncDefault : null; // Null means no limit

  // If fetching since a date, apply the SimPRO specific query syntax `ge(...)`.
  if (params.DateIssuedFrom) {
    const fromDate = params.DateIssuedFrom;
    queryParams.DateIssued = `ge(${fromDate})`;
    delete queryParams.DateIssuedFrom;
    
    // Also filter by DateModified to catch recently updated jobs
    queryParams.DateModified = `ge(${fromDate})`;
    console.log(`[sync] Update mode: DateIssued=ge(${fromDate}), DateModified=ge(${fromDate}), no fetch cap`);
  }

  // Iterate through pages until we run out of data or hit our limit.
  let page = 1;
  while (true) {
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

    // SimPRO results can be directly an array or wrapped in an object.
    const jobs = Array.isArray(resp.data) ? resp.data : resp.data?.jobs || [];
    if (!jobs.length) {
      break; // No more data
    }

    allJobs.push(...jobs);

    // Stop if we've reached our safety cap.
    if (maxJobs && allJobs.length >= maxJobs) {
      allJobs.length = maxJobs; // Trim excess
      break;
    }

    // Stop if the page wasn't full (means it's the last page).
    if (jobs.length < pageSize) {
      break;
    }
    
    page++;

    // Optional delay between pages to minimise API load.
    if (pageDelayMs > 0 || pageJitterMs > 0) {
      const delay = pageDelayMs + Math.floor(Math.random() * Math.max(0, pageJitterMs));
      await sleep(delay);
    }
  }

  return allJobs;
}

// --- Filters & Sorting ---

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
  
  // Clone array to avoid changing original.
  let filtered = [...jobs];

  // Apply revenue filters.
  if (minRevenue) {
    filtered = filtered.filter((j) => {
      const r = getRevenue(j);
      return r != null && r >= parseFloat(minRevenue);
    });
  }
  
  if (maxRevenue) {
    filtered = filtered.filter((j) => {
      const r = getRevenue(j);
      return r != null && r <= parseFloat(maxRevenue);
    });
  }

  // Sort.
  filtered.sort((a, b) => {
    const A = sortField === "revenue" ? (getRevenue(a) ?? -Infinity) : a[sortField];
    const B = sortField === "revenue" ? (getRevenue(b) ?? -Infinity) : b[sortField];
    
    if (order === "desc") return A > B ? -1 : 1;
    return A > B ? 1 : -1;
  });

  // Apply limit if no pagination is requested
  if (!page && !pageSize && limit) {
    const n = Number.parseInt(limit, 10);
    if (!Number.isNaN(n) && n > 0) filtered = filtered.slice(0, n);
  }
  return filtered;
}

function stripHtml(s = " ") {
  if (!s) return "";
  let text = String(s);
  
  // 1. Remove HTML tags (replace with space to prevent word concatenation)
  text = text.replace(/<[^>]*>/g, " ");
  
  // 2. Decode common entities
  const entities = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&copy;": "(c)",
    "&reg;": "(r)"
  };
  
  text = text.replace(/&[a-z0-9#]+;/gi, (match) => {
    // Handle hex/decimal entities if needed, but for now just common ones
    return entities[match.toLowerCase()] || " "; 
  });
  
  // 3. Normalize whitespace
  return text.replace(/\s+/g, " ").trim();
}

/**
Safely converts value to number, defaulting to `d` if invalid.
 */
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

// --- Data Normalization ---

/**
Calculates financial metrics for a job.
 */
function calculateFinancials(j) {
  const incTax = j?.Total?.IncTax ?? null;
  const revenue = typeof incTax === "number" ? incTax : incTax ? Number(incTax) : null;

  const mats = j?.Totals?.MaterialsCost;
  const resCost = j?.Totals?.ResourcesCost;
  const labor = resCost?.Labor;
  const overhead = resCost?.Overhead;
  const laborHours = resCost?.LaborHours;

  // Use Actual cost if available, otherwise fallback to Estimate.
  const materials_cost_est = num(mats?.Estimate, num(mats?.Actual, 0));
  const labor_cost_est = num(labor?.Estimate, num(labor?.Actual, 0));
  const overhead_est = num(overhead?.Actual, 0);
  const labor_hours_est = num(laborHours?.Estimate, num(laborHours?.Actual, 0));
  
  const cost_est_total = materials_cost_est + labor_cost_est + overhead_est;
  const profit_est = revenue != null ? revenue - cost_est_total : null;
  const margin_est = (revenue && revenue > 0 && profit_est != null) 
    ? profit_est / revenue 
    : null;

  return {
    revenue,
    materials_cost_est,
    labor_cost_est,
    overhead_est,
    labor_hours_est,
    cost_est_total,
    profit_est,
    margin_est,
    netMarginPct: margin_est ?? null
  };
}

/**
Transforms raw SimPRO job data into a standardized format for the frontend and ML models.
 */
function normaliseJob(j) {
  const financials = calculateFinancials(j);
  
  // Categorize profitability.
  let profitability_class = null;
  if (financials.netMarginPct != null) {
    if (financials.netMarginPct >= 0.2) profitability_class = "High";
    else if (financials.netMarginPct >= 0.05) profitability_class = "Medium";
    else profitability_class = "Low";
  }

  const descriptionText = stripHtml(j?.Description ?? " ");

  const dateIssued = j?.DateIssued ?? null;
  const dateDue = j?.DueDate ?? null;
  
  // SimPRO has multiple fields for completion date.
  const dateCompleted = j?.DateCompleted ?? j?.CompletedDate ?? null;
  
  const age_days = dateIssued ? daysBetween(dateIssued, new Date()) : null;
  const due_in_days = dateIssued && dateDue ? daysBetween(dateIssued, dateDue) : null;
  const completion_days = dateIssued && dateCompleted ? daysBetween(dateIssued, dateCompleted) : null;
  
  const isCompleted = dateCompleted != null;
  
  // Job is overdue if:
  // 1. It's completed later than due date.
  // 2. It's NOT completed and today is past due date.
  const isOverdue = dateDue ? (
    dateCompleted 
      ? new Date(dateCompleted) > new Date(dateDue)
      : new Date() > new Date(dateDue)
  ) : false;

  const txt = descriptionText.toLowerCase();
  const has_emergency = /emergency|urgent|callout|call-out|call out/.test(txt) ? 1 : 0;

  return {
    ...j,
    // Text fields
    descriptionText,
    desc_len: descriptionText.length,
    customerName: j?.Customer?.CompanyName ?? null,
    siteName: j?.Site?.Name ?? null,
    status_name: j?.Status?.Name ?? null,
    status: j?.Status ?? null,
    stage: j?.Stage ?? null,
    jobType: j?.Type ?? null,

    // Dates & Times
    dateIssued,
    dateDue,
    dateCompleted,
    age_days,
    due_in_days,
    completion_days,
    is_completed: isCompleted,
    is_overdue: isOverdue,
    
    // Financials
    ...financials,
    profitability_class,
    
    // Flags
    has_emergency,
  };
}

async function fetchJobDetail(jobId) {
  const base = process.env.SIMPRO_API_BASE || process.env.SIMPRO_API_URL;
  if (!base) throw new Error("SIMPRO_API_BASE is not configured");
  
  const companyId = process.env.SIMPRO_COMPANY_ID || "0";
  const url = new URL(
    `/api/v1.0/companies/${companyId}/jobs/${jobId}`,
    base
  ).toString();
  
  const token = await getSimproToken();

  // We wrap this in `scheduleSimproRequest`
  const resp = await withRetries(
    () =>
      scheduleSimproRequest(
        () =>
          axios.get(url, {
            headers: { Authorization: `Bearer ${token}` },
            timeout: 20_000,
          }),
        `job-${jobId}` // Label for debugging
      ),
    { maxRetries: 4, baseDelay: 1200 }
  );

  return resp.data;
}

// --- Sync Logic ---

/**
Determines the parameters for the sync operation based on the request.
 */
function getSyncParams(req) {
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
    // Full sync: Start from the absolute beginning of our history window.
    params.DateIssuedFrom = toIsoDate(FULL_HISTORY_START);
    historicalRange = true;
  } else {
    // Update sync: Use provided dates or calculate incremental start date.
    if (DateIssued) params.DateIssued = DateIssued;
    if (DateIssuedFrom) params.DateIssuedFrom = DateIssuedFrom;
    if (DateIssuedTo) params.DateIssuedTo = DateIssuedTo;
    
    // If user provided specific dates, it's a historical/custom range.
    historicalRange = Boolean(DateIssuedFrom || DateIssuedTo);

    // If no dates provided, auto-calculate from last known job.
    if (!historicalRange && !DateIssued) {
      incrementalFrom = computeIssuedFromOverride();
      params.DateIssuedFrom = incrementalFrom;
    }
  }

  return { params, syncMode, historicalRange, incrementalFrom };
}

/**
Process a batch of Job IDs: fetch details, normalize, and filter them.
 */
async function processJobBatch(ids, byIdBase, earliestAllowedMs, incrementalFromDate, syncMode) {
  const kept = [];
  let detailFailures = 0;

  // Fetch all details in parallel (Promise.allSettled handles individual failures)
  const settled = await Promise.allSettled(ids.map((id) => fetchJobDetail(id)));

  for (let i = 0; i < settled.length; i++) {
    const id = ids[i];
    const baseJob = byIdBase.get(id) || {};
    
    // All details retrieved succesfully?
    const detail = settled[i].status === "fulfilled" ? settled[i].value : {};
    
    if (settled[i].status === "rejected") {
      const reason = settled[i].reason;
      const diag = axiosDiag(reason) || { message: String(reason) };
      detailFailures++;
      console.warn(`sync: job detail fetch failed for ID ${id}`, diag);
    }

    // Merge basic list data with detailed data
    const merged = { ...baseJob, ...(detail || {}) };
    
    // Normalize and add business logic metrics
    const norm = normaliseJob(merged);
    if (!norm) continue;

    // Filter by date (if strict date checking is needed)
    const dateCandidate =
      norm.dateIssued ??
      norm.dateDue ??
      norm.DateCompleted ??
      norm.completed_date ??
      null;
      
    const d = toDate(dateCandidate);
    
    if (d) {
      const ms = d.getTime();
      // Drop if older than our absolute history start
      if (ms < earliestAllowedMs) continue;
      // Drop if older than our incremental start (double check)
      if (incrementalFromDate && ms < incrementalFromDate.getTime()) continue;
    } else if (syncMode === "full") {
      // In full mode, if it has no date,
      continue;
    }

    // Only keep jobs where we could calculate profitability
    if (norm.netMarginPct != null || norm.profitability_class != null) {
      kept.push(norm);
    }
  }

  return { kept, detailFailures };
}

router.get("/sync", authRequired, async (req, res) => {
  if (syncing) {
    return res.status(409).json({ message: "Sync already in progress" });
  }

  const { params, syncMode, historicalRange, incrementalFrom } = getSyncParams(req);
  
  // How many detailed lookups to do in parallel.
  const concurrency = parseInt(process.env.ENRICH_CONCURRENCY || "5", 10);

  // Timing metrics
  const timings = {
    start: Date.now(),
    fetchStart: null,
    fetchEnd: null,
    enrichmentStart: null,
    enrichmentEnd: null,
    dbStart: null,
    dbEnd: null,
    end: null,
  };

  syncing = true;
  try {
    console.log(`[sync] mode=${syncMode}, params=`, JSON.stringify(params, null, 2));
    
    // 1. Fetch the list of jobs
    timings.fetchStart = Date.now();
    const listRaw = await fetchSimPROJobs(params, { historical: historicalRange });
    const list = Array.isArray(listRaw) ? listRaw : [];
    timings.fetchEnd = Date.now();
    
    const fetchDurationMs = timings.fetchEnd - timings.fetchStart;
    console.info(`[sync] fetched ${list.length} raw jobs in ${fetchDurationMs}ms. Starting enrichment...`);

    const earliestAllowedMs = (
      syncMode === "full" ? FULL_HISTORY_START : UPDATE_HISTORY_START
    ).getTime();
    
    const incrementalFromDate = incrementalFrom != null 
      ? toDate(`${incrementalFrom}T00:00:00Z`) 
      : null;

    // Configuration for how many jobs we actually want to save.
    const keepMaxRaw = parseInt(process.env.SIMPRO_SYNC_MAX || "500", 10);
    const KEEP_MAX = (Number.isFinite(keepMaxRaw) && keepMaxRaw > 0) ? keepMaxRaw : null;
    
    // Extract IDs.
    const getId = (j) => j?.ID ?? j?.Id ?? j?.id ?? null;
    const idsAll = list.map((j) => getId(j)).filter(Boolean);
    const byIdBase = new Map(list.map((j) => [getId(j), j]));

    const allKept = [];
    let totalDetailFailures = 0;
    const batchTimings = [];

    // 2. Fetch details in chunks to avoid overwhelming the server or API
    timings.enrichmentStart = Date.now();
    for (
      let i = 0;
      i < idsAll.length && (!KEEP_MAX || allKept.length < KEEP_MAX);
      i += concurrency
    ) {
      const batchStart = Date.now();
      const slice = idsAll.slice(i, i + concurrency);
      
      const { kept, detailFailures } = await processJobBatch(
        slice, 
        byIdBase, 
        earliestAllowedMs, 
        incrementalFromDate, 
        syncMode
      );
      
      const batchEnd = Date.now();
      batchTimings.push({
        batchIndex: Math.floor(i / concurrency),
        jobsProcessed: slice.length,
        jobsKept: kept.length,
        durationMs: batchEnd - batchStart,
      });
      
      allKept.push(...kept);
      totalDetailFailures += detailFailures;

      if (requestGapMs > 0) await sleep(requestGapMs);
    }
    timings.enrichmentEnd = Date.now();

    const filteredCount = allKept.length;
    const excludedCount = list.length - filteredCount;
    
    if (excludedCount > 0) {
      console.info(
        `Sync: excluded ${excludedCount} jobs without netMarginPct/profitability_class (kept ${allKept.length})`
      );
    }

    // 3. Prepare for DB
    timings.dbStart = Date.now();
    const rows = allKept.map((job) => buildJobRow(job)).filter(Boolean);

    if (syncMode === "full") {
      clearJobs(); // Wipe clean for a full sync
    }

    if (rows.length) {
      upsertJobs(rows);
    }

    // Refresh memory cache
    cachedJobs = loadJobs();
    lastSyncTime = Date.now();
    timings.dbEnd = Date.now();
    timings.end = Date.now();

    const enrichmentDurationMs = timings.enrichmentEnd - timings.enrichmentStart;
    const dbDurationMs = timings.dbEnd - timings.dbStart;
    const totalDurationMs = timings.end - timings.start;
    const avgBatchDurationMs = batchTimings.length > 0
      ? batchTimings.reduce((sum, b) => sum + b.durationMs, 0) / batchTimings.length
      : 0;

    console.info(
      `[sync] Complete in ${totalDurationMs}ms (fetch: ${fetchDurationMs}ms, enrichment: ${enrichmentDurationMs}ms, db: ${dbDurationMs}ms)`
    );

    res.json({
      message: syncMode === "full" ? "Full sync complete" : "Update sync complete",
      count: cachedJobs.length,
      excluded: excludedCount,
      fetched: list.length,
      filtered: filteredCount,
      detailFailures: totalDetailFailures,
      upserted: rows.length,
      params,
      mode: syncMode,
      jobs: cachedJobs,
      timings: {
        totalMs: totalDurationMs,
        fetchMs: fetchDurationMs,
        enrichmentMs: enrichmentDurationMs,
        dbMs: dbDurationMs,
        avgBatchMs: Math.round(avgBatchDurationMs * 100) / 100,
        batchCount: batchTimings.length,
        batches: batchTimings,
      },
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

// --- Jobs API ---

router.get("/jobs", authRequired, (req, res) => {
  // Filter jobs based on query params (revenue, etc.)
  const jobs = filterAndSortJobs(cachedJobs, req.query);

  // Manual pagination for the frontend table.
  const page = parseInt(req.query.page, 10) || 1;
  const pageSize = parseInt(req.query.pageSize, 10) || jobs.length || 1;
  
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

/**
Reloads the in-memory cache from the database.
Used when other processes update the DB directly.
 */
export function refreshCachedJobsFromDb() {
  cachedJobs = loadJobs();
  return cachedJobs;
}

// --- ML Prediction Logic ---

/**
 * Helper to determine which jobs to send to the ML service.
 * Can select by specific IDs, explicit job objects, or a filter query.
 */
function selectJobsForPrediction(req, cachedJobs) {
  const bodyJobs = Array.isArray(req.body?.jobs) ? req.body.jobs : null;
  const bodyJobIds = Array.isArray(req.body?.jobIds) ? req.body.jobIds : null;

  // Determine the limit.
  const limitFromBody = parseInt(req.body?.limit, 10);
  const limitFromQuery = parseInt(req.query?.limit, 10);
  
  const effectiveLimit = Number.isFinite(limitFromBody)
    ? limitFromBody
    : Number.isFinite(limitFromQuery)
    ? limitFromQuery
    : null;

  let jobsToSend = [];

  if (bodyJobs?.length) {
    // 1. User provided the full job objects directly.
    jobsToSend = bodyJobs;
  } else if (bodyJobIds?.length) {
    // 2. User provided a list of IDs. Look them up in cache.
    const idSet = new Set(
      bodyJobIds
        .map((id) => (id == null ? null : String(id)))
        .filter(Boolean)
    );

    jobsToSend = cachedJobs.filter((job) => {
      const id = job?.ID ?? job?.id;
      return id != null && idSet.has(String(id));
    });
  } else {
    // 3. Fallback: Use standard filters (revenue, etc.) from query string.
    jobsToSend = filterAndSortJobs(cachedJobs, req.query);
  }

  // Apply limit if specified.
  if (Number.isFinite(effectiveLimit) && effectiveLimit > 0) {
    return jobsToSend.slice(0, effectiveLimit);
  }
  
  return jobsToSend;
}

/**
Generic handler for proxying requests to an ML service.
 */
async function handlePredictionRequest(req, res, serviceUrl, endpointName) {
  const jobsToSend = selectJobsForPrediction(req, cachedJobs);

  if (!jobsToSend?.length) {
    return res.json({ predictions: [], count: 0, model_loaded: false });
  }

  if (!serviceUrl) {
    return res.status(500).json({
      error: `ML service URL for ${endpointName} not configured.`,
    });
  }

  try {
    console.info(
      `[${endpointName}] forwarding ${jobsToSend.length} jobs to ML service at ${serviceUrl}`
    );
    
    const response = await axios.post(serviceUrl, {
      data: jobsToSend,
    });
    
    return res.json(response.data);
  } catch (err) {
    console.error(
      `Error forwarding jobs to ML service (${endpointName}):`,
      axiosDiag(err),
      "jobsCount:",
      jobsToSend.length
    );

    const mlStatus = err.response?.status ?? null;
    const mlBody = err.response?.data ?? { message: err.message };

    return res.status(502).json({
      error: `ML Prediction failed (${endpointName})`,
      mlStatus,
      mlBody,
    });
  }
}

// Predict profitability (margin, class)
router.post("/predict", authRequired, async (req, res) => {
  const url = process.env.ML_PROFITABILITY_URL 
    ? `${process.env.ML_PROFITABILITY_URL}/predict` 
    : null;
    
  return handlePredictionRequest(req, res, url, "predict-profitability");
});

// Predict completion time (duration)
router.post("/predict_duration", authRequired, async (req, res) => {
  const url = process.env.ML_DURATION_URL 
    ? `${process.env.ML_DURATION_URL}/predict_duration` 
    : null;``
    
  return handlePredictionRequest(req, res, url, "predict-duration");
});

// Diagnostic endpoint to check token health.
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
