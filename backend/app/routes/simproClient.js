/**
 * SimPRO API client
 * Handles OAuth authentication, rate limiting, pagination, and API requests
 */
import axios from "axios";
import { getLatestIssuedDate } from "../db/jobs.js";

// --- Utilities ---

/**
 * Helper to extract useful error details from an Axios error object
 */
export function axiosDiag(err) {
  return {
    message: err.message,
    code: err.code,
    method: err.config?.method,
    url: err.config?.url,
    status: err.response?.status,
    responseData: err.response?.data,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wraps an async function with retry logic
 * Handles rate limits (429) & network glitches
 * Uses exponential backoff to avoid overwhelming the API
 */
export async function withRetries(fn, { maxRetries = 5, baseDelay = 1000 } = {}) {
  let attempt = 0;
  
  while (true) {
    try {
      return await fn();
    } catch (err) {
      const status = err?.response?.status;
      const code = err?.code;
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      
      const isRateLimited = status === 429;
      const isServerIssues = status === 502 || status === 503 || status === 504;
      const isNetworkIssues = code === "ECONNRESET" || code === "ETIMEDOUT";

      const shouldRetry = (isRateLimited || isServerIssues || isNetworkIssues) && attempt < maxRetries;

      if (shouldRetry) {
        const jitterMs = parseInt(process.env.SIMPRO_RETRY_JITTER_MS || "250", 10);
        
        let waitTimeMs = 0;
        if (Number.isFinite(retryAfterSec)) {
          waitTimeMs = retryAfterSec * 1000;
        } else {
          const exponentialBackoff = baseDelay * Math.pow(2, attempt);
          const randomJitter = Math.floor(Math.random() * Math.max(0, jitterMs));
          waitTimeMs = exponentialBackoff + randomJitter;
        }

        console.warn(
          `simPRO retry: status=${status ?? code}, waiting ${waitTimeMs}ms (attempt ${attempt + 1}/${maxRetries})`
        );
        
        await sleep(waitTimeMs);
        attempt++;
        continue;
      }

      console.error("simPRO request failed:", axiosDiag(err));
      throw err;
    }
  }
}

// --- Rate Limiting & Throttling ---

const requestGapMs = Math.max(
  0,
  parseInt(process.env.SIMPRO_REQUEST_INTERVAL_MS || "500", 10) || 0
);

const throttlePenaltyMs = Math.max(
  requestGapMs,
  parseInt(process.env.SIMPRO_THROTTLE_PENALTY_MS || `${requestGapMs * 4}`, 10) || requestGapMs
);

let simproQueue = Promise.resolve();
let nextAllowedRequestAt = 0;

/**
 * Schedules a request to SimPRO, ensuring we don't exceed rate limits.
 * All requests through this function are sequential.
 */
export function scheduleSimproRequest(makeRequest, label = "simpro") {
  const run = simproQueue.then(async () => {
    const now = Date.now();
    const waitFor = Math.max(0, nextAllowedRequestAt - now);
    if (waitFor > 0) {
      await sleep(waitFor);
    }

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

// --- OAuth Token Management ---

let cachedToken = null;
let tokenExpiryMs = 0;

/**
 * Retrieves an OAuth 2.0 access token for SimPRO.
 * Uses client credentials flow. Returns cached token if still valid.
 */
export async function getSimproToken() {
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
      { maxRetries: 4, baseDelay: 1500 }
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

// --- URL Builders ---

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

// --- Public API ---

/**
 * Fetches specific jobs or a list of jobs from SimPRO.
 * Handles pagination automatically.
 */
export async function fetchSimPROJobs(params = {}, { historical = false, maxJobs = null } = {}) {
  const url = buildJobsUrl();
  const token = await getSimproToken();

  const pageSizeEnv = parseInt(process.env.SIMPRO_PAGE_SIZE || "250");
  const pageSize = Math.min(Number.isFinite(pageSizeEnv) ? pageSizeEnv : 250, 250);
  const pageDelayMs = parseInt(process.env.SIMPRO_PAGE_DELAY_MS || "0") || 0;
  const pageJitterMs = parseInt(process.env.SIMPRO_PAGE_JITTER_MS || "0") || 0;
  
  const maxSyncDefaultRaw = 1000;
  const maxSyncDefault =
    Number.isFinite(maxSyncDefaultRaw) && maxSyncDefaultRaw > 0 ? maxSyncDefaultRaw : null;

  const allJobs = [];
  const queryParams = { ...params };
  
  const hasDateFilter = params.DateIssued || params.DateIssuedFrom;
  const shouldCapResults = !historical && !hasDateFilter;
  const requestedMaxJobs = Number.isFinite(maxJobs) && maxJobs > 0 ? maxJobs : null;
  const appliedMaxJobs = requestedMaxJobs ?? (shouldCapResults ? maxSyncDefault : null);

  if (params.DateIssuedFrom) {
    const fromDate = params.DateIssuedFrom;
    queryParams.DateIssued = `ge(${fromDate})`;
    delete queryParams.DateIssuedFrom;
    queryParams.DateModified = `ge(${fromDate})`;
    console.log(`[sync] Update mode: DateIssued=ge(${fromDate}), DateModified=ge(${fromDate}), no fetch cap`);
  }

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

    const jobs = Array.isArray(resp.data) ? resp.data : resp.data?.jobs || [];
    if (!jobs.length) {
      break;
    }

    allJobs.push(...jobs);

    if (appliedMaxJobs && allJobs.length >= appliedMaxJobs) {
      allJobs.length = appliedMaxJobs;
      break;
    }

    if (jobs.length < pageSize) {
      break;
    }
    
    page++;

    if (pageDelayMs > 0 || pageJitterMs > 0) {
      const delay = pageDelayMs + Math.floor(Math.random() * Math.max(0, pageJitterMs));
      await sleep(delay);
    }
  }

  return allJobs;
}

/**
 * Fetches detailed information for a single job from SimPRO
 */
export async function fetchJobDetail(jobId) {
  const base = process.env.SIMPRO_API_BASE || process.env.SIMPRO_API_URL;
  if (!base) throw new Error("SIMPRO_API_BASE is not configured");
  
  const companyId = process.env.SIMPRO_COMPANY_ID || "0";
  const url = new URL(
    `/api/v1.0/companies/${companyId}/jobs/${jobId}`,
    base
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
        `job-${jobId}`
      ),
    { maxRetries: 4, baseDelay: 1200 }
  );

  return resp.data;
}
