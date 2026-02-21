/**
 * Job data normalization
 * Transforms raw SimPRO job data into standardized format for database and ML
 */
import { toDate, toIsoDate } from "./dateHelpers.js";

// --- Utilities ---

/**
 * Strips HTML tags and decodes entities
 */
export function stripHtml(s = " ") {
  if (!s) return "";
  let text = String(s);
  
  // 1. Remove HTML tags (replace with space to prevent word concatenation)
  text = text.replace(/<[^>]*>/g, " ");
  
  // 2. Decode common entities
  const entities = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&copy;": "(c)",
    "&reg;": "(r)"
  };
  
  text = text.replace(/&[a-z0-9#]+;/gi, (match) => {
    return entities[match.toLowerCase()] || " "; 
  });
  
  // 3. Normalize whitespace
  return text.replace(/\s+/g, " ").trim();
}

/**
 * Safely converts value to number, defaulting to `d` if invalid.
 */
export function num(x, d = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : d;
}

/**
 * Calculates days between two dates
 */
export function daysBetween(a, b) {
  const da = a ? new Date(a) : null;
  const db = b ? new Date(b) : null;
  if (!da || !db || Number.isNaN(+da) || Number.isNaN(+db)) return null;
  const ms = db - da;
  return Math.round(ms / 86400000);
}

// --- Financial Calculations ---

/**
 * Calculates financial metrics for a job from raw SimPRO data
 */
export function calculateFinancials(j) {
  const incTax = j?.Total?.IncTax ?? null;
  const revenue = typeof incTax === "number" ? incTax : incTax ? Number(incTax) : null;

  const mats = j?.Totals?.MaterialsCost;
  const resCost = j?.Totals?.ResourcesCost;
  const labor = resCost?.Labor;
  const overhead = resCost?.Overhead;
  const laborHours = resCost?.LaborHours;

  const materials_cost_est = num(mats?.Estimate, num(mats?.Actual, 0));
  const labor_cost_est = num(labor?.Estimate, num(labor?.Actual, 0));
  const overhead_est = num(overhead?.Actual, 0);
  const labor_hours_est = num(laborHours?.Estimate, num(laborHours?.Actual, 0));
  
  const cost_est_total = materials_cost_est + labor_cost_est + overhead_est;

  return {
    revenue,
    materials_cost_est,
    labor_cost_est,
    overhead_est,
    labor_hours_est,
    cost_est_total,
  };
}

// --- Normalization ---

/**
 * Transforms raw SimPRO job data into a standardized format for the frontend and ML models.
 */
export function normaliseJob(j) {
  const financials = calculateFinancials(j);

  const descriptionText = stripHtml(j?.Description ?? " ");

  const dateIssued = j?.DateIssued ?? null;
  const dateDue = j?.DueDate ?? null;
  const dateCompleted = j?.DateCompleted ?? j?.CompletedDate ?? null;
  
  const age_days = dateIssued ? daysBetween(dateIssued, new Date()) : null;
  const due_in_days = dateIssued && dateDue ? daysBetween(dateIssued, dateDue) : null;
  const completion_days = dateIssued && dateCompleted ? daysBetween(dateIssued, dateCompleted) : null;
  
  const isCompleted = dateCompleted != null;
  
  const isOverdue = dateDue ? (
    dateCompleted 
      ? new Date(dateCompleted) > new Date(dateDue)
      : new Date() > new Date(dateDue)
  ) : false;

  const txt = descriptionText.toLowerCase();
  const has_emergency = /emergency|urgent|callout|call-out|call out/.test(txt) ? 1 : 0;

  return {
    id: j?.ID ?? j?.Id ?? j?.id ?? null,
    // Text fields
    descriptionText,
    desc_len: descriptionText.length,
    customerName: j?.Customer?.CompanyName ?? null,
    siteName: j?.Site?.Name ?? null,
    status_name: j?.Status?.Name ?? null,
    stage: j?.Stage ?? null,
    jobType: j?.Type ?? null,

    // Dates & Times
    dateIssued,
    dateDue,
    dateCompleted,
    age_days,
    due_in_days,
    completion_days,
    is_completed: isCompleted,
    is_overdue: isOverdue,
    
    // Financials
    ...financials,
    
    // Flags
    has_emergency,
  };
}
