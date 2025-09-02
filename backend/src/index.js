import express from "express";
import cors from "cors";
import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(express.json());
app.use(cors({ origin: process.env.FRONTEND_URL || "http://localhost:5173" }));

app.get("/health", (_req, res) => res.json({ status: "ok" }));

// mount routers (create files as below)
import authRouter from "./routes/auth.js";
import dataRouter from "./routes/data.js";
app.use("/auth", authRouter);
app.use("/data", dataRouter);

const port = process.env.PORT || 5001;
app.listen(port, () => console.log(`API listening on ${port}`));