import { db } from "./db.js";

const insertJobStmt = db.prepare(`
  INSERT INTO jobs (job_id, updated_at, issued_date, completed_date, payload)
  VALUES (@job_id, @updated_at, @issued_date, @completed_date, @payload)
  ON CONFLICT(job_id) DO UPDATE SET
    updated_at = excluded.updated_at,
    issued_date = excluded.issued_date,
    completed_date = excluded.completed_date,
    payload = excluded.payload
`);

const selectAllStmt = db.prepare(`
  SELECT job_id, updated_at, issued_date, completed_date, payload
  FROM jobs
  ORDER BY
    CASE WHEN updated_at IS NULL THEN 1 ELSE 0 END,
    updated_at DESC,
    job_id ASC
`);

const latestIssuedStmt = db.prepare(`
  SELECT issued_date
  FROM jobs
  WHERE issued_date IS NOT NULL
  ORDER BY issued_date DESC
  LIMIT 1
`);

const clearJobsStmt = db.prepare("DELETE FROM jobs");

export function upsertJobs(rows = []) {
  if (!Array.isArray(rows) || rows.length === 0) return 0;
  const tx = db.transaction((items) => {
    for (const row of items) {
      insertJobStmt.run(row);
    }
  });
  tx(rows);
  return rows.length;
}

export function loadJobs() {
  const rows = selectAllStmt.all();
  const jobs = [];
  for (const row of rows) {
    const { payload } = row;
    if (typeof payload !== "string") continue;
    try {
      const parsed = JSON.parse(payload);
      jobs.push(parsed);
    } catch (err) {
      console.warn("Skipping job with invalid payload JSON", row.job_id, err);
    }
  }
  return jobs;
}

export function getLatestIssuedDate() {
  const result = latestIssuedStmt.get();
  return result?.issued_date ?? null;
}

export function clearJobs() {
  clearJobsStmt.run();
}