/**
 * job data normalisation for database and ml models.
 */

import { toDate, toIsoDate } from "./dateHelpers.js";

// utility helpers

// removes html tags and decodes common tags from text.
export function stripHtml(testString = " ") {
  if (!testString) return "";
  let text = String(testString);
  
  // html tags are replaced with spaces so words don't merge
  text = text.replace(/<[^>]*>/g, " ");
  
  // common html tags matched to their readable altermatives
  const tags = {
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
    const lowerMatch = match.toLowerCase();
    return tags[lowerMatch] || " "; 
  });
  
  // multiple whitespace characters are collapsed into one
  text = text.replace(/\s+/g, " ");
  return text.trim();
}

//converts value to number, defaults to 0 or other value if passed
export function num(test_num, default_num = 0) {
  const n = Number(test_num);
  return Number.isFinite(n) ? n : default_num;
}


// calculates whole day difference between two dates.
export function daysBetween(a, b) {
  const da = toDate(a);
  const db = toDate(b);

  if (!da || !db) {
    return null;
  }

  const daValid = !Number.isNaN(+da);
  const dbValid = !Number.isNaN(+db);

  if (!daValid || !dbValid) {
    return null;
  }

  const ms = db - da;
  const days = ms / (1000 * 60 * 60 * 24);
  return Math.round(days);
}
// financials
export function calculateFinancials(job) {
  const incTax = job.Total.IncTax;
  let revenue = (typeof incTax === "number") ? incTax : Number(incTax) ?? null;

  const mats = job.Totals.MaterialsCost;
  const resCost = job.Totals.ResourcesCost;
  const labor = resCost.Labor;
  const overhead = resCost.Overhead;
  const laborHours = resCost.LaborHours;
  
  const materials_cost_est = num(mats.Estimate, num(mats.Actual, 0));
  const labor_cost_est = num(labor.Estimate, num(labor.Actual, 0));
  const overhead_est = num(overhead.Actual, 0);
  const labor_hours_est = num(laborHours.Estimate, num(laborHours.Actual, 0));
  
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

// main normalisation

// transforms raw simpro job into clean format for frontend and ml.
export function normaliseJob(job) {
  const financials = calculateFinancials(job);
  const descriptionText = stripHtml(job.Description);

  const dateIssued = toIsoDate(job.DateIssued);
  const dateDue = toIsoDate(job.DueDate);
  const dateCompleted = toIsoDate(job.DateCompleted);
  
  let age_days = dateIssued ? daysBetween(dateIssued, new Date()) : null;
  let due_in_days = (dateIssued && dateDue) ? daysBetween(dateIssued, dateDue) : null;
  let completion_days = (dateIssued && dateCompleted) ? daysBetween(dateIssued, dateCompleted) : null;
  
  const isCompleted = dateCompleted != null;

  let isOverdue = false;
  if (dateDue) {
    const dueDate = toDate(dateDue);
    if (dateCompleted) {
      const completedDate = toDate(dateCompleted);
      isOverdue = completedDate > dueDate;
    } else {
      const now = new Date();
      isOverdue = now > dueDate;
    }
  }

  const text = descriptionText.toLowerCase();
  const emergencyPattern = /emergency|urgent|callout|call-out|call out/;
  const has_emergency = emergencyPattern.test(text) ? 1 : 0;

  return {
    id: job.ID,
    descriptionText,
    desc_len: descriptionText.length,
    customerName: job.Customer.CompanyName,
    siteName: job.Site.Name,
    status_name: job.Status.Name,
    stage: job.Stage,
    jobType: job.Type,
    dateIssued,
    dateDue,
    dateCompleted,
    age_days,
    due_in_days,
    completion_days,
    is_completed: isCompleted,
    is_overdue: isOverdue,
    ...financials,
    has_emergency
  };
}
