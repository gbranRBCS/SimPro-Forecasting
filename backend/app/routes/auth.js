import express from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { db } from "../db/db.js";

const router = express.Router();

/**
auth routes for register and login.
this file handles password hashing, credential checks, and jwt creation.
 */

// --- Constants ---
const SESSION_DURATION = "2h"; // token stays valid for 2 hours


function getAuthFields(body) {
  const safeBody = body ?? {};

  // username and password are normalised to avoid accidental spaces
  const rawUsername = safeBody.username ?? "";
  const rawPassword = safeBody.password ?? "";
  const rawRole = safeBody.role ?? "user";

  const username = String(rawUsername).trim();
  const password = String(rawPassword);
  const role = String(rawRole).trim() || "user";

  return { username, password, role };
}

function hasMissingCredentials(username, password) {
  return username.length === 0 || password.length === 0;
}

// POST /auth/register
// creates a user acct, stores hashed pw
router.post("/register", async (req, res) => {
  const { username, password, role } = getAuthFields(req.body);

  // missing credentials are rejected before any db or hash work starts
  if (hasMissingCredentials(username, password)) {
    return res.status(400).json({ error: "Username and password are required." });
  }

  try {
    // bcrypt creates the password hash that will go into the users table
    const hash = await bcrypt.hash(password, 10);

    // prevents the storage of plain passwords in DB
    const stmt = db.prepare(
      "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)"
    );
    stmt.run(username, hash, role);

    console.log(`User registered: ${username}`);
    return res.status(201).json({ ok: true, message: "User registered successfully." });

  } catch (err) {
    // handle duplicate usernames
    if (String(err).includes("UNIQUE")) {
      return res.status(409).json({ error: "Username is already taken." });
    }

    // any other db/hash problem is treated as a server-side failure
    console.error("Registration failed:", err);
    return res.status(500).json({ error: "Internal server error during registration." });
  }
});

// POST /auth/login
// checks credentials and returns token + user details
router.post("/login", async (req, res) => {
  const { username, password } = getAuthFields(req.body);

  // invalid input stops here so db access is skipped
  if (hasMissingCredentials(username, password)) {
    return res.status(400).json({ error: "Username and password are required." });
  }

  const jwtSecret = process.env.JWT_SECRET;
  if (!jwtSecret) {
    console.error("Login failed: JWT_SECRET is not set.");
    return res.status(500).json({ error: "Server configuration error." });
  }

  try {
    // user row is loaded from sqlite by username
    const row = db.prepare(
      "SELECT id, username, password_hash, role FROM users WHERE username = ?"
    ).get(username);

    // generic message avoids revealing whether the username exists in db
    if (!row) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // bcrypt compares incoming password with the stored hash
    const passwordsMatch = await bcrypt.compare(password, row.password_hash);
    if (!passwordsMatch) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // jwt token proves identity for further requests
    const token = jwt.sign(
      { 
        sub: row.id,
        username: row.username, 
        role: row.role 
      },
      jwtSecret,
      { expiresIn: SESSION_DURATION }
    );

    console.log(`User logged in: ${username}`);
    return res.json({ 
      token,
      user: { username: row.username, role: row.role }
    });

  } catch (err) {
    console.error("Login process error:", err);
    return res.status(500).json({ error: "Login failed due to server error." });
  }
});

export default router;