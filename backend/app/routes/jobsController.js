/**
 * jobs controller for listing and ml predictions.
 * handles filtering, paginating, and external calls to ml
 */
import axios from "axios";
import { getCachedJobs } from "./jobsRepo.js";
import { axiosDiag } from "./simproClient.js";
import { getSimproToken } from "./simproClient.js";

// helpers for filtering and sorting jobs

function getRevenue(j) {
  return typeof j?.revenue === "number" ? j.revenue : null;
}

function getSortValue(job, sortField) {
  if (sortField === "revenue") {
    const revenue = job?.revenue;
    return typeof revenue === "number" ? revenue : -Infinity;
  }

  const rawValue = job?.[sortField];

  if (rawValue instanceof Date) {
    return rawValue.toISOString();
  }

  switch (typeof rawValue) {
    case "number":
      return rawValue;
    case "string":
      return rawValue.toLowerCase();
    default:
      return "";
  }
}

// -- Quick sort for job array
function quickSort(arr, sortField, order) {
  if (arr.length <= 1) {
    return arr;
  }

  const pivot = arr[arr.length-1]
  const L = [];
  const R = [];

  const pivotValue = getSortValue(pivot, sortField);

  for (let i =0; i < arr.length-1; i++){
    const currentValue = getSortValue(arr[i], sortField);

    if (currentValue < pivotValue){
      L.push(arr[i]);
    } else {
      R.push(arr[i]);
    }
  }

  let result = [];
  if (L.length > 0 && R.length > 0) {
    result = [...quickSort(L, sortField, order), pivot, ...quickSort(R, sortField, order)];
  } else if (L.length > 0) {
    result = [...quickSort(L, sortField, order), pivot];
  } else { // L !> 0 && R.length > 0
    result = [pivot, ...quickSort(R, sortField, order)]
  }

  if (order === "desc") {
    result.reverse();
  }

  return result;
}

// filters and sorts jobs using query params from the request. Utilises quick sort algorithm.
export function filterAndSortJobs(jobs, query) {
  const {
    sortField = "revenue",
    order = "asc",
    minRevenue,
    maxRevenue,
    limit,
    page,
    pageSize,
  } = query;

  let filtered = [...jobs];

  const revenueList = filtered.map((job) => getRevenue(job));
  const hasAnyRevenue = revenueList.length > 0;

  if (minRevenue && hasAnyRevenue) {
    const minValue = parseFloat(minRevenue);
    filtered = filtered.filter((j) => {
      const r = getRevenue(j);
      return r != null && r >= minValue;
    });
  }

  if (maxRevenue && hasAnyRevenue) {
    const maxValue = parseFloat(maxRevenue);
    filtered = filtered.filter((j) => {
      const r = getRevenue(j);
      return r != null && r <= maxValue;
    });
  }

  filtered = quickSort(filtered, sortField, order);

  if (!page && !pageSize && limit) {
    const n = Number.parseInt(limit);
    if (!Number.isNaN(n) && n > 0) {
      filtered = filtered.slice(0, n);
    }
  }
  return filtered;
}

// decides which jobs are sent to ml
export function selectJobsForPrediction(req, cachedJobs) {
  const bodyJobs = (req.body?.jobs) instanceof Array ? req.body.jobs : null;
  const bodyJobIds = (req.body?.jobIds) instanceof Array ? req.body.jobIds : null;

  const limitFromBody = parseInt(req.body?.limit);
  const limitFromQuery = parseInt(req.query?.limit);

  let usedLimit = null;
  if (Number.isFinite(limitFromBody)) {
    usedLimit = limitFromBody;
  } else if (Number.isFinite(limitFromQuery)) {
    usedLimit = limitFromQuery;
  }

  let jobsToSend = [];

  if (bodyJobs?.length) {
    jobsToSend = bodyJobs;
  } else if (bodyJobIds?.length) {
    const idSet = new Set(
      bodyJobIds
        .map((id) => (id == null ? null : String(id)))
        .filter(Boolean) // removes nulls after mapping
    );

    jobsToSend = cachedJobs.filter((job) => {
      const id = job?.id;
      return id != null && idSet.has(String(id));
    });
  } else {
    // when no explicit list is given, query filters are applied
    jobsToSend = filterAndSortJobs(cachedJobs, req.query);
  }

  if (Number.isFinite(usedLimit) && usedLimit > 0) {
    return jobsToSend.slice(0, usedLimit);
  }
  
  return jobsToSend;
}

// generic handler for requests to an ml service.
export async function handlePredictionRequest(req, res, serviceUrl, endpointName) {
  const cachedJobs = getCachedJobs();
  const jobsToSend = selectJobsForPrediction(req, cachedJobs);

  // empty input just returns an empty prediction list
  if (!jobsToSend || jobsToSend.length === 0) {
    return res.json({ predictions: [], count: 0, model_loaded: false });
  }

  // missing url means ml service is not available
  if (!serviceUrl) {
    return res.status(500).json({
      error: `ML service URL for ${endpointName} not available.`,
    });
  }

  try {
    const response = await axios.post(serviceUrl, {
      data: jobsToSend,
    });

    return res.json(response.data);
  } catch (err) {
    console.error(`ML request failed (${endpointName}):`, axiosDiag(err));

    return res.status(502).json({
      error: `ML Prediction failed (${endpointName})`,
      detail: err.message,
    });
  }
}

// GET /jobs
export function handleGetJobs(req, res) {
  const cachedJobs = getCachedJobs();
  const jobs = filterAndSortJobs(cachedJobs, req.query);
  const totalJobs = jobs.length;

  // page values are pulled from query or defaults if missing
  let candidatePage = parseInt(req.query.page);
  if (candidatePage < 1 || candidatePage === NaN || candidatePage > 99) {
    let page = 1;
  } else {
    let page = candidatePage;
  }
  
  const pageSize = parseInt(req.query.pageSize) || DEFAULT_PAGE_SIZE;

  const start = (page - 1) * pageSize;
  const end = start + pageSize;
  const paged = jobs.slice(start, end);

  res.json({
    jobs: paged,
    total: totalJobs,
    page,
    pageSize: pageSize,
    totalPages: Math.ceil(totalJobs / pageSize),
  });
}

// POST /predict
export async function handleProfitabilityPrediction(req, res) {
  let url = "http://localhost:8000/predict";
  
  return handlePredictionRequest(req, res, url, "predict-profitability");
}

// POST /predict_duration
export async function handleDurationPrediction(req, res) {
  let url = "http://localhost:8001/predict";

  return handlePredictionRequest(req, res, url, "predict-duration");
}

// GET /oauth-test
export async function handleOAuthTest(req, res) {
  try {
    const token = await getSimproToken();
    let tokenPreview = token ? `${token.slice(0, 24)}...` : null;

    return res.status(200).json({
      ok: true,
      tokenPreview,
      base: process.env.SIMPRO_API_BASE,
    });
  } catch (err) {
    const message = err?.message || "no error message given";
    return res.status(500).json({ ok: false, error: message });
  }
}
