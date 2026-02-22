/**
 * sync service
 * fetches jobs from simpro, normalises them, validates dates, and stores in database
 */
import { fetchSimPROJobs, fetchJobDetail, axiosDiag } from "./simproClient.js";
import { normaliseJob } from "./normaliseJob.js";
import { toDate, FULL_HISTORY_START, UPDATE_HISTORY_START, computeIssuedFromOverride, toIsoDate } from "./dateHelpers.js";
import { buildJobRow, getCachedJobs } from "./jobsRepo.js";
import { clearJobs, loadJobs, upsertJobs } from "../db/jobs.js";

const requestGapMs = parseInt(process.env.SIMPRO_REQUEST_INTERVAL_MS);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * parses sync request query parameters
 * selects between full sync or incremental, chooses date filters
 */
export function getSyncParams(req) {
  const {
    DateIssued,
    DateIssuedFrom,
    DateIssuedTo,
    mode: syncModeRaw,
  } = req.query;

  const requestedMode = (syncModeRaw || "").toString().toLowerCase();
  let syncMode = (requestedMode === "full") ? "full" : "update";

  const params = {};
  let historicalRange = false;
  let incrementalFrom = null;

  if (syncMode === "full") {
    params.DateIssuedFrom = toIsoDate(FULL_HISTORY_START);
    historicalRange = true;
  } else {
    if (DateIssued) {
      params.DateIssued = DateIssued;
    }
    if (DateIssuedFrom) {
      params.DateIssuedFrom = DateIssuedFrom;
    }
    if (DateIssuedTo) {
      params.DateIssuedTo = DateIssuedTo;
    }
    
    const hasCustomRange = DateIssuedFrom || DateIssuedTo;
    historicalRange = Boolean(hasCustomRange);

    // if no custom range specified, fetch from computed override
    if (!historicalRange && !DateIssued) {
      incrementalFrom = computeIssuedFromOverride();
      params.DateIssuedFrom = incrementalFrom;
    }
  }

  return { params, syncMode, historicalRange, incrementalFrom };
}

/**
 * fetches job detail for a batch of ids, normalises them, and filters by date
 * returns the jobs that are valid and the number of failures / rejections
 */
export async function processJobBatch(ids, earliestAllowedMs, incrementalFromDate, syncMode) {
  const kept = [];
  let detailFailures = 0;

  // fetch all job details in parallel
  const settled = await Promise.allSettled(ids.map((id) => fetchJobDetail(id)));

  for (let i = 0; i < settled.length; i++) {
    const id = ids[i];
    const result = settled[i];
    
    let detail = null;
    if (result.status === "fulfilled") {
      detail = result.value;
    }
    
    if (result.status === "rejected") {
      const reason = result.reason;
      const diag = axiosDiag(reason) || { message: String(reason) };
      detailFailures++;
      console.warn(`sync: job detail fetch failed for ID ${id}`, diag);
    }

    if (!detail) {
      continue;
    }
    
    const norm = normaliseJob(detail);
    if (!norm) {
      continue;
    }

    const dateCandidate = norm.dateIssued;
    const d = toDate(dateCandidate);
    
    if (d) {
      const ms = d.getTime();
      
      // reject if too old
      if (ms < earliestAllowedMs) {
        continue;
      }
      
      // reject if before incremental sync cutoff
      if (incrementalFromDate && ms < incrementalFromDate.getTime()) {
        continue;
      }
    } else if (syncMode === "full") {
      // full sync rejects jobs with no valid date
      continue;
    }

    kept.push(norm);
  }

  return { kept, detailFailures };
}

// runs the whole sync operation - i.e. fetch, normalise, validate, store
export async function executeSyncOperation(params, syncMode, historicalRange, incrementalFrom) {
  const concurrency = parseInt(process.env.ENRICH_CONCURRENCY);

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

  const keepMaxRaw = parseInt(process.env.SIMPRO_SYNC_MAX);
  let KEEP_MAX = null;
  if (Number.isFinite(keepMaxRaw) && keepMaxRaw > 0) {
    KEEP_MAX = keepMaxRaw;
  }
  
  // 1. fetch
  timings.fetchStart = Date.now();
  const listRaw = await fetchSimPROJobs(params, {
    historical: historicalRange,
    maxJobs: KEEP_MAX,
  });
  const list = Array.isArray(listRaw) ? listRaw : [];
  timings.fetchEnd = Date.now();
  
  const fetchDurationMs = timings.fetchEnd - timings.fetchStart;
  console.info(`[sync] fetched ${list.length} raw jobs in ${fetchDurationMs}ms. Starting enrichment...`);

  let earliestAllowedMs = 0;
  if (syncMode === "full") {
    earliestAllowedMs = FULL_HISTORY_START.getTime();
  } else {
    earliestAllowedMs = UPDATE_HISTORY_START.getTime();
  }
  
  // calculate incremental sync date filter
  let incrementalFromDate = null;
  if (incrementalFrom != null) {
    const dateStr = `${incrementalFrom}T00:00:00Z`;
    incrementalFromDate = toDate(dateStr);
  }
  
  // extract job ids from list
  const getId = (j) => j.ID;
  let idsAll = list
    .map((j) => getId(j))
    .filter(Boolean);
  
  if (KEEP_MAX) {
    idsAll = idsAll.slice(0, KEEP_MAX);
  }

  // 2. enrich and normalise
  const allKept = [];
  let totalDetailFailures = 0;
  let totalDetailAttempts = 0;
  const batchTimings = [];

  timings.enrichmentStart = Date.now();
  
  for (let i = 0; i < idsAll.length; i += concurrency) {
    if (KEEP_MAX && allKept.length >= KEEP_MAX) {
      break;
    }
    
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

  // 3. DB
  timings.dbStart = Date.now();
  
  // build database rows from normalised jobs
  const rows = allKept
    .map((job) => buildJobRow(job))
    .filter(Boolean);

  // clear old data if full sync
  if (syncMode === "full") {
    clearJobs();
  }

  if (rows.length) {
    upsertJobs(rows);
  }

  // reload cache
  const cachedJobs = loadJobs();
  timings.dbEnd = Date.now();
  timings.end = Date.now();

  // final stats
  const enrichmentDurationMs = timings.enrichmentEnd - timings.enrichmentStart;
  const dbDurationMs = timings.dbEnd - timings.dbStart;
  const totalDurationMs = timings.end - timings.start;
  
  let avgBatchDurationMs = 0;
  if (batchTimings.length > 0) {
    const totalBatchMs = batchTimings.reduce((sum, b) => sum + b.durationMs, 0);
    avgBatchDurationMs = totalBatchMs / batchTimings.length;
  }
  
  const detailSuccesses = Math.max(0, totalDetailAttempts - totalDetailFailures);
  if (totalDetailAttempts > 0) {
    let detailFailureRate = Math.round((totalDetailFailures / totalDetailAttempts) * 10000) / 100;
  } else {
    detailFailureRate = 0;
  }

  console.info(
    `[sync] Complete in ${totalDurationMs}ms (fetch: ${fetchDurationMs}ms, enrichment: ${enrichmentDurationMs}ms, db: ${dbDurationMs}ms)`
  );

  let message = "Update sync complete";
  if (syncMode === "full") {
    message = "Full sync complete";
  }

  return {
    message,
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
