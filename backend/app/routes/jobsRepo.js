/**
 * Jobs repository
 * Handles database operations and in-memory cache management
 */
import { clearJobs, loadJobs, upsertJobs } from "../db/jobs.js";
import { toIsoDate, toIsoString } from "./dateHelpers.js";

// Re-export database functions for convenience
export { clearJobs, loadJobs, upsertJobs } from "../db/jobs.js";

/**
 * Gets all cached jobs
 */
export function getCachedJobs() {
  return loadJobs();
}

/**
 * Reloads the in-memory cache from the database.
 * Used when other processes update the DB directly.
 */
export function refreshCachedJobsFromDb() {
  return loadJobs();
}

// --- Row Building ---

/**
 * Extracts and validates key fields from a normalized job object to prepare it for database storage.
 */
export function buildJobRow(job) {
  if (!job) return null;
  
  const jobId = job?.id;
  if (jobId == null) return null;

  const issuedCandidate = job?.dateIssued ?? null;
  const completedCandidate = job?.dateCompleted ?? null;
  
  const updatedCandidate = completedCandidate ?? issuedCandidate;

  // Validate issued date is not in the future
  let validatedIssuedDate = null;
  if (issuedCandidate) {
    const parsed = new Date(issuedCandidate);
    if (!Number.isNaN(+parsed)) {
      const now = new Date();
      const oneDayAhead = new Date(now.getTime() + 24 * 60 * 60 * 1000);
      
      if (parsed.getTime() <= oneDayAhead.getTime()) {
        validatedIssuedDate = toIsoDate(parsed);
      } else {
        console.warn(
          `Job ${jobId}: issued date ${issuedCandidate} is in the future. Nullifying to protect data integrity.`
        );
      }
    }
  }

  return {
    job_id: String(jobId),
    issued_date: validatedIssuedDate,
    completed_date: toIsoString(completedCandidate),
    updated_at: toIsoString(updatedCandidate),
    payload: JSON.stringify(job),
  };
}
