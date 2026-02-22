/**
 * simpro API client for fetching job data.
 * handles oauth tokens, rate limits, retries, and pagination.
 */
import axios from "axios";
import { getLatestIssuedDate } from "../db/jobs.js";

// utility functions

/**
 * extracts useful error info from axios errors for logging.
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

// retry logic for network failures and rate limits

/**
 * wraps a function with automatic retry logic
 * handles rate limits and other errors
 * uses exponential to slow down requests
 */
export async function withRetries(fn, { maxRetries = 5, baseDelay = 1000 } = {}) {
  let attempt = 0;
  
  while (true) {
    try {
      const result = await fn();
      return result;
    } catch (err) {
      const status = err?.response?.status;
      const code = err?.code;
      const retryAfterSec = Number(err?.response?.headers?.["retry-after"]);
      
      // check if this error is worth retrying
      const isRateLimited = status === 429;
      const isServerIssues = status === 502 || status === 503 || status === 504;
      const isNetworkIssues = code === "ECONNRESET" || code === "ETIMEDOUT";

      const canRetry = attempt < maxRetries;
      const shouldRetry = (isRateLimited || isServerIssues || isNetworkIssues) && canRetry;

      if (shouldRetry) {
        const jitterMs = parseInt(process.env.SIMPRO_RETRY_JITTER_MS);
        
        let waitTimeMs = 0;
        if (Number.isFinite(retryAfterSec)) {
          waitTimeMs = retryAfterSec * 1000;
        } else {
          const exponentialBackoff = baseDelay * Math.pow(2, attempt);
          const randomJitter = Math.floor(Math.random() * Math.max(0, jitterMs));
          waitTimeMs = exponentialBackoff + randomJitter;
        }

        console.warn(
          `simPRO retry: status=${status || code}, waiting ${waitTimeMs}ms (attempt ${attempt + 1}/${maxRetries})`
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

// rate limiting to avoid hitting simpro api limits

const requestGapMs = parseInt(process.env.SIMPRO_REQUEST_INTERVAL_MS)

// penalty delay when rate limited
const throttlePenaltyMs = parseInt(process.env.SIMPRO_THROTTLE_PENALTY_MS);

let simproQueue = Promise.resolve();
let nextAllowedRequestAt = 0;

/**
 * schedules a request to simpro with rate limiting
 * all requests go through a single queue so they are sequential
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

// oauth token management

let cachedToken = null;
let tokenExpiryMs = 0;

// gets an oauth token for simpro API access.
export async function getSimproToken() {
  const now = Date.now();
  const tokenStillValid = cachedToken && now < tokenExpiryMs - 60_000;
  
  if (tokenStillValid) {
    return cachedToken;
  }

  const form = new URLSearchParams();
  form.append("grant_type", process.env.GRANT_TYPE);

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

    const data = resp.data;
    const access_token = data.access_token;
    const expires_in = data.expires_in;
    
    if (!access_token) {
      throw new Error("No access_token in token response");
    }
    
    cachedToken = access_token;
    tokenExpiryMs = now + expires_in * 1000;
    
    return cachedToken;
  } catch (err) {
    const d = axiosDiag(err);
    console.error("getSimproToken failed:", d);
    throw new Error("OAuth token fetch failed");
  }
}

// url builders

function buildJobsUrl() {
  const base = process.env.SIMPRO_API_BASE;
  if (!base) {
    throw new Error("SIMPRO_API_BASE is not configured");
  }
  
  const companyId = process.env.SIMPRO_COMPANY_ID;
  let jobsPath = process.env.SIMPRO_JOBS_PATH;
  
  // replace company id placeholder
  jobsPath = jobsPath.replace("{COMPANY_ID}", companyId);

  if (!jobsPath.startsWith("/")) jobsPath = `/${jobsPath}`;
  if (!jobsPath.endsWith("/")) jobsPath += "/";

  return new URL(jobsPath, base).toString();
}

// main api functions

// fetches jobs from simpro with automatic pagination
export async function fetchSimPROJobs(params = {}, { historical = false, maxJobs = null } = {}) {
  const url = buildJobsUrl();
  const token = await getSimproToken();

  const pageSizeEnv = parseInt(process.env.SIMPRO_PAGE_SIZE);
  const pageSize = Math.min(pageSizeEnv, 250);
  const pageDelayMs = parseInt(process.env.SIMPRO_PAGE_DELAY_MS);
  const pageJitterMs = parseInt(process.env.SIMPRO_PAGE_JITTER_MS);
  
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

    // response can be array or object with jobs property
    let jobs = [];
    if (Array.isArray(resp.data)) {
      jobs = resp.data;
    } else if (resp.data?.jobs) {
      jobs = resp.data.jobs;
    }
    
    if (!jobs.length) {
      break;
    }

    allJobs.push(...jobs);

    if (appliedMaxJobs && allJobs.length >= appliedMaxJobs) {
      allJobs.length = appliedMaxJobs;
      break;
    }

    // check if last page
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

// fetches detailed info for a single job by id.
export async function fetchJobDetail(jobId) {
  const base = process.env.SIMPRO_API_BASE;
  if (!base) {
    throw new Error("SIMPRO_API_BASE is not configured in .env");
  }
  
  const companyId = process.env.SIMPRO_COMPANY_ID;
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
