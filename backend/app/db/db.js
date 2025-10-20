import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import "dotenv/config"; // Ensure dotenv is loaded

const DB_FILE = process.env.DB_FILE;
if (!DB_FILE) {
  throw new Error("DB_FILE environment variable is not set.");
}
const dir = path.dirname(DB_FILE);
if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

export const db = new Database(DB_FILE, { verbose: null });

db.exec(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
     CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        updated_at TEXT,
        issued_date TEXT,
        completed_date TEXT,
        payload TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at DESC);
`);