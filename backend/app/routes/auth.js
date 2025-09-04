import express from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";

const router = express.Router();

//In memory users stored for demo purposes
const users = new Map ([
    ["george_b", { password: bcrypt.hashSync("pw123", 10), isApproved : true}]
    ["director", { password: bcrypt.hashSync("pw456", 10), isApproved : false}]
]);

//User registration with manual approval
router.post("/register", async (req, res) => {
    const { username, password } = req.body;
    if (!username || !password) return res.status(400).json({error: "Missing fields"});
    if (users.has(username)) return res.status(400).json({error: "User already exists"});

    const hashed = await bcrypt.hash(password, 10);
    users.set(username, {password: hashed, isApproved: false});
    return res.json({ message: "User registered, awating approval."});
});

// User Login
router.post("/login", async (req, res) => {
    const { username, password } = req.body;
    const user = users.get(username);

    if (!user) return res.status(401).json({ error: "Invalid credentials" });
    if (!user.isApproved) return res.status(403).json({ error: "User not approved" });

    const validPw = await bcrypt.compare(password, user.password);
    if (!validPw) return res.status(401).json({ error: "Invalid credentials" });

    const token = jwt.sign({ username }, process.env.JWT_SECRET, { expiresIn: "1h" });
    return res.json({ token });
});

export default router;
