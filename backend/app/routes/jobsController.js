/**
 * Jobs controller
 * Handles API routes for jobs listing and ML predictions
 */
import axios from "axios";
import { getCachedJobs } from "./jobsRepo.js";
import { axiosDiag } from "./simproClient.js";
import { getSimproToken } from "./simproClient.js";

// --- Filters & Sorting ---

function getRevenue(j) {
  return typeof j?.revenue === "number" ? j.revenue : null;
}

/**
 * Filters and sorts jobs based on query parameters
 */
export function filterAndSortJobs(jobs, query) {
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

  filtered.sort((a, b) => {
    const A = sortField === "revenue" ? (getRevenue(a) ?? -Infinity) : a[sortField];
    const B = sortField === "revenue" ? (getRevenue(b) ?? -Infinity) : b[sortField];
    
    if (order === "desc") return A > B ? -1 : 1;
    return A > B ? 1 : -1;
  });

  if (!page && !pageSize && limit) {
    const n = Number.parseInt(limit, 10);
    if (!Number.isNaN(n) && n > 0) filtered = filtered.slice(0, n);
  }
  return filtered;
}

/**
 * Helper to determine which jobs to send to the ML service.
 * Can select by specific IDs, explicit job objects, or a filter query.
 */
export function selectJobsForPrediction(req, cachedJobs) {
  const bodyJobs = Array.isArray(req.body?.jobs) ? req.body.jobs : null;
  const bodyJobIds = Array.isArray(req.body?.jobIds) ? req.body.jobIds : null;

  const limitFromBody = parseInt(req.body?.limit, 10);
  const limitFromQuery = parseInt(req.query?.limit, 10);
  
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
        .map((id) => (id == null ? null : String(id)))
        .filter(Boolean)
    );

    jobsToSend = cachedJobs.filter((job) => {
      const id = job?.id;
      return id != null && idSet.has(String(id));
    });
  } else {
    jobsToSend = filterAndSortJobs(cachedJobs, req.query);
  }

  if (Number.isFinite(effectiveLimit) && effectiveLimit > 0) {
    return jobsToSend.slice(0, effectiveLimit);
  }
  
  return jobsToSend;
}

/**
 * Generic handler for proxying requests to an ML service.
 */
export async function handlePredictionRequest(req, res, serviceUrl, endpointName) {
  const cachedJobs = getCachedJobs();
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

/**
 * GET /jobs - Retrieve cached jobs with filtering/pagination
 */
export function handleGetJobs(req, res) {
  const cachedJobs = getCachedJobs();
  const jobs = filterAndSortJobs(cachedJobs, req.query);

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
}

/**
 * POST /predict - Profitability prediction
 */
export async function handleProfitabilityPrediction(req, res) {
  const url = process.env.ML_PROFITABILITY_URL 
    ? `${process.env.ML_PROFITABILITY_URL}/predict` 
    : "http://localhost:8000/predict";
    
  return handlePredictionRequest(req, res, url, "predict-profitability");
}

/**
 * POST /predict_duration - Completion time prediction
 */
export async function handleDurationPrediction(req, res) {
  const url = process.env.ML_DURATION_URL 
    ? `${process.env.ML_DURATION_URL}/predict` 
    : "http://localhost:8001/predict";
    
  return handlePredictionRequest(req, res, url, "predict-duration");
}

/**
 * GET /oauth-test - Diagnostic endpoint to check token health
 */
export async function handleOAuthTest(req, res) {
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
}
