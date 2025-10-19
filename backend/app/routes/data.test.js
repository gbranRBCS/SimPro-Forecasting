import request from "supertest";
import express from "express";
import dataRouter from "./data.js";
import jwt from "jsonwebtoken";

// mock environment variables
process.env.JWT_SECRET = "testsecret";
process.env.SIMPRO_API_URL = "http://mock-simpro";
process.env.SIMPRO_API_TOKEN = "mocktoken";
process.env.ML_URL = "http://mock-ml";
process.env.SYNC_INTERVAL_MINUTES = "1"; // 1 minute for testing

// mock axios
import axios from "axios";
jest.mock("axios");

const app = express();
app.use(express.json());
app.use("/data", dataRouter);

function getToken(payload = { user: "test" }) {
    return jwt.sign(payload, process.env.JWT_SECRET);
}

describe("Data Routes", () => {
    let jobsMock = [
        { id: 1, revenue: 100, date: "2024-01-01" },
        { id: 2, revenue: 200, date: "2024-01-02" },
        { id: 3, revenue: 300, date: "2024-01-03" }
    ];

    beforeEach(() => {
        axios.get.mockReset();
        axios.post.mockReset();
    });

    test("/sync caches jobs and does not re-fetch if interval not elapsed", async () => {
        axios.get.mockResolvedValueOnce({ data: jobsMock });

        // First sync
        const res1 = await request(app)
            .get("/data/sync")
            .set("Authorization", `Bearer ${getToken()}`);
        expect(res1.body.jobs).toEqual(jobsMock);
        expect(res1.body.message).toMatch(/successful/i);

        // Second sync (should not call axios.get again)
        const res2 = await request(app)
            .get("/data/sync")
            .set("Authorization", `Bearer ${getToken()}`);
        expect(res2.body.jobs).toEqual(jobsMock);
        expect(res2.body.message).toMatch(/recently synced/i);
        expect(axios.get).toHaveBeenCalledTimes(1);
    });

    test("/jobs applies filtering and sorting", async () => {
        // Seed cache via sync
        axios.get.mockResolvedValueOnce({ data: jobsMock });
        await request(app)
            .get("/data/sync")
            .set("Authorization", `Bearer ${getToken()}`);

        // Filter minRevenue
        const res1 = await request(app)
            .get("/data/jobs?minRevenue=200")
            .set("Authorization", `Bearer ${getToken()}`);
        expect(res1.body.jobs).toEqual([
            { id: 2, revenue: 200, date: "2024-01-02" },
            { id: 3, revenue: 300, date: "2024-01-03" }
        ]);

        // Filter maxRevenue
        const res2 = await request(app)
            .get("/data/jobs?maxRevenue=200")
            .set("Authorization", `Bearer ${getToken()}`);
        expect(res2.body.jobs).toEqual([
            { id: 1, revenue: 100, date: "2024-01-01" },
            { id: 2, revenue: 200, date: "2024-01-02" }
        ]);

        // Sort desc
        const res3 = await request(app)
            .get("/data/jobs?sortField=revenue&order=desc")
            .set("Authorization", `Bearer ${getToken()}`);
        expect(res3.body.jobs[0].revenue).toBe(300);
    });

    test("/predict forwards filtered jobs to ML and handles errors", async () => {
        // Seed cache via sync
        axios.get.mockResolvedValueOnce({ data: jobsMock });
        await request(app)
            .get("/data/sync")
            .set("Authorization", `Bearer ${getToken()}`);

        // Mock ML response
        axios.post.mockResolvedValueOnce({ data: { prediction: "ok" } });

        // Only jobs with revenue >= 200
        const res1 = await request(app)
            .post("/data/predict?minRevenue=200")
            .set("Authorization", `Bearer ${getToken()}`);
        expect(axios.post).toHaveBeenCalledWith(
            `${process.env.ML_URL}/predict`,
            { data: [
                { id: 2, revenue: 200, date: "2024-01-02" },
                { id: 3, revenue: 300, date: "2024-01-03" }
            ]}
        );
        expect(res1.body.prediction).toBe("ok");

        // ML error handling
        axios.post.mockRejectedValueOnce(new Error("ML down"));
        const res2 = await request(app)
            .post("/data/predict")
            .set("Authorization", `Bearer ${getToken()}`);
        expect(res2.body.error).toMatch(/ML Prediction failed/i);
        expect(res2.status).toBe(502);
    });

    test("/predict accepts jobIds in the request body", async () => {
        axios.get.mockResolvedValueOnce({ data: jobsMock });
        await request(app)
            .get("/data/sync")
            .set("Authorization", `Bearer ${getToken()}`);

        axios.post.mockResolvedValueOnce({ data: { predictions: [{ jobId: 2, class: "High" }] } });

        const res = await request(app)
            .post("/data/predict")
            .send({ jobIds: [2] })
            .set("Authorization", `Bearer ${getToken()}`);

        expect(axios.post).toHaveBeenCalledWith(
            `${process.env.ML_URL}/predict`,
            { data: [{ id: 2, revenue: 200, date: "2024-01-02" }] }
        );
        expect(res.status).toBe(200);
        expect(res.body.predictions).toEqual([{ jobId: 2, class: "High" }]);
    });
});