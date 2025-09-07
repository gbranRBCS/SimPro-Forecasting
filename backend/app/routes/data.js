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

  try {
    const r = await axios.get(url, {
      headers: { Authorization: `Bearer ${token}` },
      params, // e.g. { DateIssued: "2018-05-29" }
      timeout: 20_000,
    });
    return r.data;
  } catch (err) {
    const d = axiosDiag(err);
    console.error("fetchSimPROJobs() failed:", d);
    const e = new Error("simPRO jobs fetch failed");
    e.detail = d;
    throw e;
  }
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

  try {
    cachedJobs = await fetchSimPROJobs(params);
    lastSyncTime = now;
    res.json({ message: "Sync was successful", jobs: cachedJobs });
  } catch (err) {
    res.status(500).json({
      error: "Failed to fetch simPRO job data.",
      detail: err.detail || { message: err.message },
    });
  }
});

// Helper to filter and sort jobs based on query params
function filterAndSortJobs(jobs, query) {
  let { sortField = "date", order = "asc", minRevenue, maxRevenue } = query;
  let filteredJobs = [...jobs];

  if (minRevenue) filteredJobs = filteredJobs.filter(j => j.Total?.IncTax >= parseFloat(minRevenue));
  if (maxRevenue) filteredJobs = filteredJobs.filter(j => j.Total?.IncTax <= parseFloat(maxRevenue));

  filteredJobs.sort((a, b) => {
    if (order === "desc") return b[sortField] > a[sortField] ? 1 : -1;
    return a[sortField] > b[sortField] ? 1 : -1;
  });

  return filteredJobs;
}

router.get("/jobs", authRequired, (req, res) => {
  const jobs = filterAndSortJobs(cachedJobs, req.query);
  const limit = Number.parseInt(req.query.limit, 10);
  const limited = Number.isFinite(limit) && limit > 0 ? jobs.slice(0, limit) : jobs;
  res.json({ jobs: limited });
});

// Forward cleaned jobs to ML microservice
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

// Token check
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