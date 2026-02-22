import express from "express";
import cors from "cors";
import dotenv from "dotenv";
dotenv.config();

const app = express();
app.use(express.json());
app.use(cors({ origin: process.env.FRONTEND_URL }));

// mount routers
import authRouter from "../app/routes/auth.js";
import dataRouter from "../app/routes/data.js";
app.use("/auth", authRouter);
app.use("/data", dataRouter);

app.get("/health", (_req, res) => res.json({ status: "ok" }));

const port = process.env.PORT || 5001;
app.listen(port, () => console.log(`API listening on ${port}`));