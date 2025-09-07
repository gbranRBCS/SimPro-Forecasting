
import express from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

const router = express.Router();

// create users
const users = new Map([
  ["george_b",   { passwordHash: bcrypt.hashSync("pw123", 10), isApproved: true }],
  ["director_1", { passwordHash: bcrypt.hashSync("pw456", 10), isApproved: false }],
]);

// POST /auth/register
router.post("/register", async (req, res) => {
  const { username, password } = req.body || {};
  if (!username || !password) return res.status(400).json({ error: "Missing fields" });
  if (users.has(username))   return res.status(400).json({ error: "User exists" });

  const passwordHash = await bcrypt.hash(password, 10);
  users.set(username, { passwordHash, isApproved: false });
  return res.json({ message: "User registered. Awaiting approval." });
});

// POST /auth/login
router.post("/login", async (req, res) => {
  const { username, password } = req.body || {};
  const user = users.get(username);
  if (!user)                return res.status(401).json({ error: "Invalid credentials" });
  if (!user.isApproved)     return res.status(403).json({ error: "User not approved" });

  const ok = await bcrypt.compare(password, user.passwordHash);
  if (!ok) return res.status(401).json({ error: "Invalid credentials" });

  try {
    const token = jwt.sign({ username }, process.env.JWT_SECRET, { expiresIn: "1h" });
    return res.json({ token });
  } catch (e) {
    return res.status(500).json({ error: "Token generation failed" });
  }
});

export default router;