/**
 * date parsing and date math used by sync routes.
 */
import { getLatestIssuedDate } from "../db/jobs.js";

// --- Configuration

const DEFAULT_FULL_HISTORY_START = "2025-01-01T00:00:00Z";
const DEFAULT_UPDATE_HISTORY_START = "2010-01-01T00:00:00Z";

/**
 * parses env date input and falls back to a safe default.
 * yyyy-mm-dd input is converted to full utc format first.
 */
function parseStartDate(raw, fallback) {
  const source = raw ?? fallback;
  const isoLike = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(source)
    ? `${source}T00:00:00Z`
    : source;
    
  const parsed = new Date(isoLike);
  return Number.isFinite(+parsed) ? parsed : new Date(fallback);
}

export const FULL_HISTORY_START = parseStartDate(
  process.env.SIMPRO_FULL_HISTORY_START,
  DEFAULT_FULL_HISTORY_START
);

export const UPDATE_HISTORY_START = parseStartDate(
  process.env.SIMPRO_UPDATE_HISTORY_START,
  DEFAULT_UPDATE_HISTORY_START
);

export const LOOKBACK_HOURS = Math.max(
  0,
  parseInt(process.env.SIMPRO_SYNC_LOOKBACK_HOURS || "24") || 24
);

// helpers shared by sync parameter logic

/**
 * converts input into a valid Date, otherwise returns null.
 */
export function toDate(value) {
  if (!value) {
    return null;
  }

  const dateValue = value instanceof Date ? new Date(value.getTime()) : new Date(value);
  const isValidDate = Number.isFinite(+dateValue);

  return isValidDate ? dateValue : null;
}

/**
 * returns a full iso timestamp like 2025-01-01T13:10:00.000Z.
 */
export function toIsoString(value) {
  const dateValue = toDate(value);
  return dateValue ? dateValue.toISOString() : null;
}

/**
 * returns only the date section in yyyy-mm-dd format.
 */
export function toIsoDate(value) {
  const dateValue = toDate(value);
  if (!dateValue) {
    return null;
  }

  const year = dateValue.getUTCFullYear();
  const month = `${dateValue.getUTCMonth() + 1}`.padStart(2, "0");
  const day = `${dateValue.getUTCDate()}`.padStart(2, "0");

  return `${year}-${month}-${day}`;
}

/**
 * calculates incremental sync start date from the newest cached job.
 * result is restricted so it never goes earlier than update history start.
 */
export function computeIssuedFromOverride() {
  const latestIssued = getLatestIssuedDate();
  const latestDate = toDate(latestIssued);
  const lookbackMs = LOOKBACK_HOURS * 60 * 60 * 1000;
  
  // no cached date means sync should begin from configured minimum
  if (!latestDate) {
    return toIsoDate(UPDATE_HISTORY_START);
  }

  const lowerBound = UPDATE_HISTORY_START.getTime();
  const from = new Date(latestDate.getTime() - lookbackMs);

  // final value stays inside the allowed history window
  const restricted = new Date(Math.max(lowerBound, from.getTime()));
  return toIsoDate(restricted);
}
