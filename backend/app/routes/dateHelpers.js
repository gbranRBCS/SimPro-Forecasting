/**
 * Date parsing and calculation utilities for sync operations
 */
import { getLatestIssuedDate } from "../db/jobs.js";

// --- Configuration ---

const DEFAULT_FULL_HISTORY_START = "2025-01-01T00:00:00Z";
const DEFAULT_UPDATE_HISTORY_START = "2010-01-01T00:00:00Z";

/**
 * Parses a date string into a Date object.
 * Returns the fallback date if parsing fails.
 * Ensures strict ISO formatting for input like "YYYY-MM-DD" with RegEx.
 */
function parseStartDate(raw, fallback) {
  const source = raw || fallback;
  const isoLike = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(source)
    ? `${source}T00:00:00Z`
    : source;
    
  const parsed = new Date(isoLike);
  return Number.isFinite(+parsed) ? parsed : new Date(fallback);
}

export const FULL_HISTORY_START = parseStartDate(
  process.env.SIMPRO_FULL_HISTORY_START || process.env.SIMPRO_HISTORY_START,
  DEFAULT_FULL_HISTORY_START
);

export const UPDATE_HISTORY_START = parseStartDate(
  process.env.SIMPRO_UPDATE_HISTORY_START || process.env.SIMPRO_MIN_ISSUED_DATE,
  DEFAULT_UPDATE_HISTORY_START
);

export const LOOKBACK_HOURS = Math.max(
  0,
  parseInt(process.env.SIMPRO_SYNC_LOOKBACK_HOURS || "24", 10) || 24
);

// --- Date Helpers ---

/**
 * Converts value to a Date object
 */
export function toDate(value) {
  if (!value) return null;
  const d = value instanceof Date ? new Date(value.getTime()) : new Date(value);
  return Number.isFinite(+d) ? d : null;
}

/**
 * Returns ISO 8601 string (full timestamp)
 */
export function toIsoString(value) {
  const d = toDate(value);
  return d ? d.toISOString() : null;
}

/**
 * Returns just the YYYY-MM-DD part.
 */
export function toIsoDate(value) {
  const d = toDate(value);
  if (!d) return null;
  const year = d.getUTCFullYear();
  const month = `${d.getUTCMonth() + 1}`.padStart(2, "0");
  const day = `${d.getUTCDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/**
 * Determines the start date for an incremental sync.
 * Looks at the most recent job, subtracts the lookback window, and clamps it to the minimum history start date.
 */
export function computeIssuedFromOverride() {
  const latestIssued = getLatestIssuedDate();
  const latestDate = toDate(latestIssued);
  const lookbackMs = LOOKBACK_HOURS * 60 * 60 * 1000;
  
  if (!latestDate) {
    return toIsoDate(UPDATE_HISTORY_START);
  }

  const lowerBound = UPDATE_HISTORY_START.getTime();
  const from = new Date(latestDate.getTime() - lookbackMs);
  
  const clamped = new Date(Math.max(lowerBound, from.getTime()));
  return toIsoDate(clamped);
}
