/**
 * Sync service
 * Orchestrates the synchronization process between SimPRO and the local database
 */
import { fetchSimPROJobs, fetchJobDetail, axiosDiag } from "./simproClient.js";
import { normaliseJob } from "./normaliseJob.js";
import { toDate, FULL_HISTORY_START, UPDATE_HISTORY_START, computeIssuedFromOverride, toIsoDate } from "./dateHelpers.js";
import { buildJobRow, getCachedJobs } from "./jobsRepo.js";
import { clearJobs, loadJobs, upsertJobs } from "../db/jobs.js";

const requestGapMs = Math.max(
  0,
  parseInt(process.env.SIMPRO_REQUEST_INTERVAL_MS || "500", 10) || 0
);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Determines the parameters for a sync operation based on the request.
 */
export function getSyncParams(req) {
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

  return { params, syncMode, historicalRange, incrementalFrom };
}

/**
 * Process a batch of Job IDs: fetch details, normalize, and filter them.
 */
export async function processJobBatch(ids, earliestAllowedMs, incrementalFromDate, syncMode) {
  const kept = [];
  let detailFailures = 0;

  const settled = await Promise.allSettled(ids.map((id) => fetchJobDetail(id)));

  for (let i = 0; i < settled.length; i++) {
    const id = ids[i];
    const detail = settled[i].status === "fulfilled" ? settled[i].value : null;
    
    if (settled[i].status === "rejected") {
      const reason = settled[i].reason;
      const diag = axiosDiag(reason) || { message: String(reason) };
      detailFailures++;
      console.warn(`sync: job detail fetch failed for ID ${id}`, diag);
    }

    if (!detail) {
      continue;
    }
    
    const norm = normaliseJob(detail);
    if (!norm) continue;

    const dateCandidate =
      norm.dateIssued ??
      norm.dateDue ??
      norm.dateCompleted ??
      norm.completed_date ??
      null;
      
    const d = toDate(dateCandidate);
    
    if (d) {
      const ms = d.getTime();
      if (ms < earliestAllowedMs) continue;
      if (incrementalFromDate && ms < incrementalFromDate.getTime()) continue;
    } else if (syncMode === "full") {
      continue;
    }

    kept.push(norm);
  }

  return { kept, detailFailures };
}

/**
 * Executes a sync operation: fetches, normalizes, and persists job data.
 * Returns sync results including timing metrics and statistics.
 */
export async function executeSyncOperation(params, syncMode, historicalRange, incrementalFrom) {
  const concurrency = parseInt(process.env.ENRICH_CONCURRENCY || "5", 10);

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

  console.log(`[sync] mode=${syncMode}, params=`, JSON.stringify(params, null, 2));

  const keepMaxRaw = parseInt(process.env.SIMPRO_SYNC_MAX || "500", 10);
  const KEEP_MAX = (Number.isFinite(keepMaxRaw) && keepMaxRaw > 0) ? keepMaxRaw : null;
  
  // 1. Fetch the list of jobs
  timings.fetchStart = Date.now();
  const listRaw = await fetchSimPROJobs(params, {
    historical: historicalRange,
    maxJobs: KEEP_MAX,
  });
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
  
  const getId = (j) => j?.ID ?? j?.Id ?? j?.id ?? null;
  const idsAll = list
    .map((j) => getId(j))
    .filter(Boolean)
    .slice(0, KEEP_MAX ?? undefined);

  const allKept = [];
  let totalDetailFailures = 0;
  let totalDetailAttempts = 0;
  const batchTimings = [];

  // 2. Fetch details in chunks
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
      earliestAllowedMs, 
      incrementalFromDate, 
      syncMode
    );
    
    const batchEnd = Date.now();
    batchTimings.push({
      batchIndex: Math.floor(i / concurrency),
      jobsProcessed: slice.length,
      jobsKept: kept.length,
      detailFailures,
      durationMs: batchEnd - batchStart,
    });
    
    allKept.push(...kept);
    totalDetailFailures += detailFailures;
    totalDetailAttempts += slice.length;

    if (requestGapMs > 0) await sleep(requestGapMs);
  }
  timings.enrichmentEnd = Date.now();

  const filteredCount = allKept.length;
  const excludedCount = list.length - filteredCount;
  
  if (excludedCount > 0) {
    console.info(
      `Sync: excluded ${excludedCount} jobs (kept ${allKept.length})`
    );
  }

  // 3. Prepare for DB
  timings.dbStart = Date.now();
  const rows = allKept.map((job) => buildJobRow(job)).filter(Boolean);

  if (syncMode === "full") {
    clearJobs();
  }

  if (rows.length) {
    upsertJobs(rows);
  }

  // Refresh memory cache
  const cachedJobs = loadJobs();
  timings.dbEnd = Date.now();
  timings.end = Date.now();

  const enrichmentDurationMs = timings.enrichmentEnd - timings.enrichmentStart;
  const dbDurationMs = timings.dbEnd - timings.dbStart;
  const totalDurationMs = timings.end - timings.start;
  const avgBatchDurationMs = batchTimings.length > 0
    ? batchTimings.reduce((sum, b) => sum + b.durationMs, 0) / batchTimings.length
    : 0;
  const detailSuccesses = Math.max(0, totalDetailAttempts - totalDetailFailures);
  const detailFailureRate = totalDetailAttempts > 0
    ? Math.round((totalDetailFailures / totalDetailAttempts) * 10000) / 100
    : 0;

  console.info(
    `[sync] Complete in ${totalDurationMs}ms (fetch: ${fetchDurationMs}ms, enrichment: ${enrichmentDurationMs}ms, db: ${dbDurationMs}ms)`
  );

  return {
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
    diagnostics: {
      requestedListCount: list.length,
      detailAttempts: totalDetailAttempts,
      detailFailures: totalDetailFailures,
      detailSuccesses,
      detailFailureRatePct: detailFailureRate,
      keptAfterEnrichment: filteredCount,
      droppedAfterEnrichment: Math.max(0, totalDetailAttempts - filteredCount),
      keepMaxApplied: KEEP_MAX,
      concurrency,
    },
    timings: {
      totalMs: totalDurationMs,
      fetchMs: fetchDurationMs,
      enrichmentMs: enrichmentDurationMs,
      dbMs: dbDurationMs,
      avgBatchMs: Math.round(avgBatchDurationMs * 100) / 100,
      batchCount: batchTimings.length,
      batches: batchTimings,
    },
  };
}
