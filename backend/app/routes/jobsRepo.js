/**
 * jobs repository for db operations and cache access.
 */
import { clearJobs, loadJobs, upsertJobs } from "../db/jobs.js";
import { toIsoDate, toIsoString } from "./dateHelpers.js";

export { clearJobs, loadJobs, upsertJobs } from "../db/jobs.js";

/**
 * returns cached jobs from the database
 */
export function getCachedJobs() {
  return loadJobs();
}

/**
 * reloads the cache when data is updated outside this file
 */
export function refreshCachedJobsFromDb() {
  return loadJobs();
}

// job row building helpers

/**
 * builds a safe db row from a normalised job object.
 * basic validation is done so impossible dates do not get stored.
 */
export function buildJobRow(job) {
  if (!job || job?.id == null) {
    return null;
  }

  const jobId = job.id;
  const issuedCandidate = job?.dateIssued || null;
  const completedCandidate = job?.dateCompleted || null;

  // use completed date if not null, otherwise use issued
  let updatedCandidate = completedCandidate ? completedCandidate : issuedCandidate;

  // issued date is checked to avoid future dates getting saved
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
          `Job ${jobId}: issued date ${issuedCandidate} is in the future. Ignoring to protect dataset.`
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
