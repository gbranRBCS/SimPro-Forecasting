import express from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { db } from "../db/db.js";

const router = express.Router();

/**
AUTHENTICATION ROUTES
Handles user registration and login.
Uses:
- bcrypt for secure password hashing.
- jsonwebtoken (JWT) for session management.
- SQLite (via better-sqlite3) for user storage.
 */

// --- Constants ---
const SALT_ROUNDS = 10;
const SESSION_DURATION = "2h"; // Tokens expire after 2 hours


// POST /auth/register
// Creates a new user account.
router.post("/register", async (req, res) => {
  const { username, password, role = "user" } = req.body || {};

  // 1. Basic Validation
  if (!username || !password) {
    return res.status(400).json({ error: "Username and password are required." });
  }

  try {
    // 2. Hash existing password
    // We never store plain text passwords in the database.
    const hash = await bcrypt.hash(password, SALT_ROUNDS);

    // 3. Insert into DB
    const stmt = db.prepare(
      "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)"
    );
    stmt.run(username, hash, role);

    console.log(`User registered: ${username}`);
    return res.status(201).json({ ok: true, message: "User registered successfully." });

  } catch (err) {
    // 4. Handle duplicates
    if (String(err).includes("UNIQUE")) {
      return res.status(409).json({ error: "Username is already taken." });
    }

    // 5. Build generic error
    console.error("Registration failed:", err);
    return res.status(500).json({ error: "Internal server error during registration." });
  }
});

// POST /auth/login
// Verifies credentials and issues a JWT token.
router.post("/login", async (req, res) => {
  const { username, password } = req.body || {};

  // 1. Basic input check
  if (!username || !password) {
    return res.status(400).json({ error: "Username and password are required." });
  }

  try {
    // 2. Lookup user
    const row = db.prepare(
      "SELECT id, username, password_hash, role FROM users WHERE username = ?"
    ).get(username);

    // If user not found, we return a generic error.
    if (!row) {
      return res.status(401).json({ error: "Invalid credentials" }); // (User not found)
    }

    // 3. Verify password
    const passwordsMatch = await bcrypt.compare(password, row.password_hash);
    if (!passwordsMatch) {
      return res.status(401).json({ error: "Invalid credentials" }); // (Wrong password)
    }

    // 4. Generate Session Token (JWT)
    // This token proves identity for subsequent requests.
    const token = jwt.sign(
      { 
        sub: row.id,        // Subject (User ID)
        username: row.username, 
        role: row.role 
      },
      process.env.JWT_SECRET,
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