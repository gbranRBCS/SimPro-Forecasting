/**
 * router for sync and job-related api endpoints.
 * links auth checks with controllers and sync service calls.
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

// prevents two sync runs from happening at the same time
let syncing = false;

// middleware that checks for a valid bearer token.
function authRequired(req, res, next) {
  const authHeader = req.headers["authorization"] ?? "";
  const headerParts = authHeader.split(" ");
  const bearerToken = headerParts[0] === "Bearer" ? headerParts[1] : null;

  if (!bearerToken) {
    return res.sendStatus(401);
  }

  const jwtSecret = process.env.JWT_SECRET;
  if (!jwtSecret) {
    console.error("Auth failed: JWT_SECRET is not set.");
    return res.sendStatus(500);
  }

  jwt.verify(bearerToken, jwtSecret, (err, user) => {
    if (err) {
      return res.sendStatus(403);
    }

    req.user = user;
    next();
  });
}

/**
 * GET /sync
 * runs a full or incremental sync from simPRO into local storage.
 * query params can still override date windows when needed.
 */
router.get("/sync", authRequired, async (req, res) => {
  // when a sync is already running, this request is blocked early
  if (syncing) {
    return res.status(409).json({ message: "Sync already running" });
  }

  syncing = true;
  try {
    // request query is translated into the sync service inputs
    const { params, syncMode, historicalRange, incrementalFrom } = getSyncParams(req);
    const result = await executeSyncOperation(params, syncMode, historicalRange, incrementalFrom);
    
    // response includes latest cached jobs so the ui can refresh immediately
    const cachedJobs = getCachedJobs();
    return res.json({
      ...result,
      jobs: cachedJobs,
    });
  } catch (err) {
    console.error("Sync failed:", err.message);
    return res.status(502).json({
      error: "Failed to fetch simPRO data",
      detail: err.message,
    });
  } finally {
    syncing = false; // ensures syncing can be re-run, regardless of result
  }
});

/**
 * GET /jobs
 * returns cached jobs with filtering, sorting, and pagination from controller logic.
 */
router.get("/jobs", authRequired, handleGetJobs);

/**
 * POST /predict
 * sends jobs to the profitability model and returns predictions.
 */
router.post("/predict", authRequired, handleProfitabilityPrediction);

/**
 * POST /predict_duration
 * sends jobs to the duration model using the same request shape as /predict.
 */
router.post("/predict_duration", authRequired, handleDurationPrediction);

/**
 * GET /oauth-test
 * quick endpoint for checking simPRO oauth token health.
 */
router.get("/oauth-test", handleOAuthTest);

export default router;
