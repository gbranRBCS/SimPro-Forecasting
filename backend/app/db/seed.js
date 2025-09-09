import "dotenv/config";
import { db } from "./db.js";
import bcrypt from "bcrypt";

const username = process.env.SEED_USER?.trim();
const password = process.env.SEED_PASS;

if (!username || !password) {
  console.error("SEED_USER and SEED_PASS must be set in your environment.");
  process.exit(1);
}

const hash = await bcrypt.hash(password, 10);

try {
  const stmt = db.prepare(
    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)"
  );
  stmt.run(username, hash, "admin");
  console.log(`Seeded user "${username}" (role=admin).`);
} catch (e) {
  const msg = e?.message || String(e);
  if (msg.includes("UNIQUE")) {
    console.log(`User "${username}" already exists. Skipping.`);
  } else {
    console.error(e);
    process.exit(1);
  }
}