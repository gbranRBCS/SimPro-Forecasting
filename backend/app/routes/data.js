import express from "express";
import axios from "axios";
import jwt from "jsonwebtoken";

const router = express.Router();

// Store sync state
let lastSyncTime = null;
const syncInterval = parseInt(process.env.SYNC_INTERVAL) || 60;

// Cache current job data
let cachedJobs = [];

// JWT Authentication
function authRequired(req, res, next) {
    const authHeader = req.headers["authorization"];
    const token = authHeader && authHeader.split(" ")[1];
    if (!token) return res.sendStatus(401);

    jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
        if (err) return res.sendStatus(403);
        req.user = user;
        next();
    });
}

// Helper to fetch from simPRO API
async function fetchSimPROJobs() {
    try {
        const response = await axios.get(`${process.env.SIMPRO_API_URL}/jobs`, {
            headers: {
                "Authorization": `Bearer ${process.env.SIMPRO_API_TOKEN}`
            }
        });
        return response.data;
    } catch (error) {
        console.error("Error fetching simPRO jobs:", error.message, error.response?.data);
        throw new Error("Failed to fetch simPRO job data.");
    }
}

// Helper to filter and sort jobs based on query params
function filterAndSortJobs(jobs, query) {
    let { sortField = "date", order = "asc", minRevenue, maxRevenue } = query;
    let filteredJobs = [...jobs];

    // Apply Filters
    if (minRevenue) filteredJobs = filteredJobs.filter(j => j.revenue >= parseFloat(minRevenue));
    if (maxRevenue) filteredJobs = filteredJobs.filter(j => j.revenue <= parseFloat(maxRevenue));

    // Sort
    filteredJobs.sort((a, b) => {
        if (order === "desc") return b[sortField] > a[sortField] ? 1 : -1;
        return a[sortField] > b[sortField] ? 1 : -1;
    });

    return filteredJobs;
}

// Sync functionality
router.get("/sync", authRequired, async (req, res) => {
    const now = Date.now();
    if (lastSyncTime && now - lastSyncTime < syncInterval * 60 * 1000) {
        return res.json({ message: "Already recently synced", jobs: cachedJobs });
    }

    try {
        cachedJobs = await fetchSimPROJobs();
        lastSyncTime = now;
        res.json({ message: "Sync was successful", jobs: cachedJobs });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Filtering and Sorting
router.get("/jobs", authRequired, (req, res) => {
    const jobs = filterAndSortJobs(cachedJobs, req.query);
    res.json({ jobs });
});

// Forward cleaned jobs to ML microservice
router.post("/predict", authRequired, async (req, res) => {
    const jobsToSend = filterAndSortJobs(cachedJobs, req.query);
    try {
        const response = await axios.post(`${process.env.ML_URL}/predict`, { data: jobsToSend });
        res.json(response.data);
    } catch (err) {
        console.error("Error forwarding jobs to ML service:", err.message, err.response?.data);
        res.status(500).json({ error: "ML Prediction failed" });
    }
});

export default router;