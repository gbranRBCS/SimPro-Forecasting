import express from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { db } from "../db/db.js";

const router = express.Router();


// POST /auth/register
router.post("/register", async (req, res) => {
  const { username, password, role = "user" } = req.body || {};
  if (!username || !password) return res.status(400).json({ error: "username and password required" });

  try {
    const hash = await bcrypt.hash(password, 10);
    const stmt = db.prepare("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)");
    stmt.run(username, hash, role);
    return res.status(201).json({ ok: true });
  } catch (e) {
    if (String(e).includes("UNIQUE")) {
      return res.status(409).json({ error: "username already exists" });
    }
    console.error("register error:", e);
    return res.status(500).json({ error: "registration failed" });
  }
});

// POST /auth/login
router.post("/login", async (req, res) => {
  const { username, password } = req.body || {};
  if (!username || !password) return res.status(400).json({ error: "username and password required" });

  try {
    const row = db.prepare("SELECT id, username, password_hash, role FROM users WHERE username = ?").get(username);
    if (!row) return res.status(401).json({ error: "invalid credentials" });

    const ok = await bcrypt.compare(password, row.password_hash);
    if (!ok) return res.status(401).json({ error: "invalid credentials" });

    const token = jwt.sign(
      { sub: row.id, username: row.username, role: row.role },
      process.env.JWT_SECRET,
      { expiresIn: "2h" }
    );

    return res.json({ token });
  } catch (e) {
    console.error("login error:", e);
    return res.status(500).json({ error: "login failed" });
  }
});

export default router;