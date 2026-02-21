/**
 * Main router module for data synchronization and job APIs
Uses:
- simproClient.js: OAuth, API pagination, rate limiting
- dateHelpers.js: Date parsing and calculations
- normaliseJob.js: Job data transformation
- jobsRepo.js: Database operations and cache management
- syncService.js: Sync orchestration and batch processing
- jobsController.js: Route handlers and filtering
 */

import express from "express";
import jwt from "jsonwebtoken";
import { getSyncParams, executeSyncOperation } from "./syncService.js";
import { getCachedJobs } from "./jobsRepo.js";
import {
  handleGetJobs,
  handleProfitabilityPrediction,
  handleDurationPrediction,
  handleOAuthTest,
} from "./jobsController.js";

const router = express.Router();

// Flag to prevent overlapping sync operations.
let syncing = false;

/**
 * Middleware: Protect routes with JWT authentication.
 */
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

/**
 * GET /sync - Synchronize jobs from SimPRO
 *
 * Supports two modes:
 * - Full sync (mode=full): Refetches all jobs from history start
 * - Incremental sync (default): Fetches only recently changed jobs
 *
 * Query parameters:
 * - mode: "full" or "update" (default)
 * - DateIssuedFrom: Start date for custom range (ISO 8601 or YYYY-MM-DD)
 * - DateIssuedTo: End date for custom range
 * - DateIssued: Specific date filter
 */
router.get("/sync", authRequired, async (req, res) => {
  if (syncing) {
    return res.status(409).json({ message: "Sync already in progress" });
  }

  syncing = true;
  try {
    const { params, syncMode, historicalRange, incrementalFrom } = getSyncParams(req);
    const result = await executeSyncOperation(params, syncMode, historicalRange, incrementalFrom);
    
    const cachedJobs = getCachedJobs();
    res.json({
      ...result,
      jobs: cachedJobs,
    });
  } catch (err) {
    console.error("Sync failed:", err.message);
    res.status(502).json({
      error: "Failed to fetch simPRO data",
      detail: err.message,
    });
  } finally {
    syncing = false;
  }
});

/**
 * GET /jobs - Retrieve cached jobs with filtering and pagination
 *
 * Query parameters:
 * - minRevenue: Filter jobs with revenue >= this value
 * - maxRevenue: Filter jobs with revenue <= this value
 * - sortField: Field to sort by (default: "revenue")
 * - order: "asc" or "desc" (default: "asc")
 * - page: Page number for pagination (1-indexed)
 * - pageSize: Items per page
 * - limit: Alternative to pagination (returns first N items)
 */
router.get("/jobs", authRequired, handleGetJobs);

/**
 * POST /predict - Profitability prediction via ML service
 *
 * Request body can include:
 * - jobs: Array of full job objects to predict on
 * - jobIds: Array of job IDs to look up in cache
 * - limit: Max number of jobs to send
 *
 * Falls back to query-based filtering if no body provided.
 */
router.post("/predict", authRequired, handleProfitabilityPrediction);

/**
 * POST /predict_duration - Completion time prediction via ML service
 *
 * Same body/query format as /predict
 */
router.post("/predict_duration", authRequired, handleDurationPrediction);

/**
 * GET /oauth-test - Diagnostic endpoint to verify SimPRO OAuth token health
 */
router.get("/oauth-test", handleOAuthTest);

export default router;
