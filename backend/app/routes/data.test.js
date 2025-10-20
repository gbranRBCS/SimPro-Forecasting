import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import request from "supertest";
import express from "express";
import jwt from "jsonwebtoken";
import { jest } from "@jest/globals";

// mock environment variables (must be set before importing the router)
process.env.JWT_SECRET = "testsecret";
process.env.SIMPRO_API_BASE = "http://mock-simpro";
process.env.ML_URL = "http://mock-ml";
process.env.SYNC_INTERVAL_MINUTES = "1"; // unused but retained for compatibility
process.env.TOKEN_URL = "http://mock-auth/token";
process.env.SIMPRO_CLIENT_ID = "cid";
process.env.SIMPRO_CLIENT_SECRET = "secret";

const tmpDir = path.join(os.tmpdir(), "simpro-tests");
if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
process.env.DB_FILE = path.join(tmpDir, "jobs-test.sqlite");

import axios from "axios";

const app = express();
app.use(express.json());

let dataRouter;
let refreshCachedJobsFromDb;
let clearJobs;
let getSpy;
let postSpy;

function getToken(payload = { user: "test" }) {
  return jwt.sign(payload, process.env.JWT_SECRET);
}

function mockJobsEndpoints(jobs) {
  getSpy.mockImplementation((url, config = {}) => {
    if (url.includes("/jobs/") && !config?.params) {
      const id = url.split("/").filter(Boolean).pop();
      const job = jobs.find((j) => String(j.ID ?? j.id) === id) || {};
      return Promise.resolve({ data: job });
    }
    return Promise.resolve({ data: jobs, headers: { "result-pages": "1" } });
  });
}

const jobsMock = [
  {
    ID: "1",
    id: 1,
    Total: { IncTax: 100 },
    DateIssued: "2025-01-02T00:00:00Z",
    DateUpdated: "2025-01-03T00:00:00Z",
    Description: "Job 1",
  },
  {
    ID: "2",
    id: 2,
    Total: { IncTax: 200 },
    DateIssued: "2025-01-03T00:00:00Z",
    DateUpdated: "2025-01-04T00:00:00Z",
    Description: "Job 2",
  },
  {
    ID: "3",
    id: 3,
    Total: { IncTax: 300 },
    DateIssued: "2025-01-04T00:00:00Z",
    DateUpdated: "2025-01-05T00:00:00Z",
    Description: "Job 3",
  },
];

beforeAll(async () => {
  ({ default: dataRouter, refreshCachedJobsFromDb } = await import("./data.js"));
  ({ clearJobs } = await import("../db/jobs.js"));
  app.use("/data", dataRouter);
});

beforeEach(() => {
  jest.restoreAllMocks();
  getSpy = jest.spyOn(axios, "get");
  postSpy = jest.spyOn(axios, "post");
  postSpy.mockImplementation((url) => {
    if (url === process.env.TOKEN_URL) {
      return Promise.resolve({ data: { access_token: "mocktoken", expires_in: 3600 } });
    }
    return Promise.reject(new Error(`Unexpected axios.post call to ${url}`));
  });
  if (clearJobs) clearJobs();
  if (refreshCachedJobsFromDb) refreshCachedJobsFromDb();
});

afterAll(() => {
  try {
    fs.unlinkSync(process.env.DB_FILE);
  } catch (err) {
    // ignore if already removed
  }
});

describe("Data Routes", () => {
  test("/sync persists jobs and returns cached payloads", async () => {
    mockJobsEndpoints(jobsMock);

    const res1 = await request(app)
      .get("/data/sync")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res1.status).toBe(200);
    expect(res1.body.count).toBe(3);
    expect(res1.body.upserted).toBe(3);
    expect(res1.body.jobs.map((j) => j.ID)).toEqual(["3", "2", "1"]);

    // ensure data returned without hitting API again
    getSpy.mockReset();
    const res2 = await request(app)
      .get("/data/jobs")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res2.status).toBe(200);
    expect(res2.body.total).toBe(3);
    expect(res2.body.jobs.map((j) => j.ID)).toEqual(["1", "2", "3"]);
    expect(getSpy).not.toHaveBeenCalled();
  });

  test("/sync performs incremental lookback and skips stale records", async () => {
    const firstBatch = [
      {
        ID: "1",
        id: 1,
        Total: { IncTax: 100 },
        DateIssued: "2025-01-02T00:00:00Z",
        DateUpdated: "2025-01-03T00:00:00Z",
        Description: "First job",
      },
      {
        ID: "2",
        id: 2,
        Total: { IncTax: 200 },
        DateIssued: "2025-01-03T00:00:00Z",
        DateUpdated: "2025-01-04T00:00:00Z",
        Description: "Second job",
      },
      {
        ID: "3",
        id: 3,
        Total: { IncTax: 300 },
        DateIssued: "2025-01-04T00:00:00Z",
        DateUpdated: "2025-01-05T00:00:00Z",
        Description: "Third job",
      },
    ];

    const incrementalBatch = [
      {
        ID: "4",
        id: 4,
        Total: { IncTax: 400 },
        DateIssued: "2025-01-05T00:00:00Z",
        DateUpdated: "2025-01-06T00:00:00Z",
        Description: "New job",
      },
      {
        ID: "legacy",
        id: "legacy",
        Total: { IncTax: 50 },
        DateIssued: "2016-06-01T00:00:00Z",
        DateUpdated: "2016-06-02T00:00:00Z",
        Description: "Ancient job",
      },
    ];

    const detailById = new Map(
      [...firstBatch, ...incrementalBatch].map((job) => [String(job.ID), job]),
    );

    const listParams = [];
    const detailRequests = [];

    getSpy.mockImplementation((url, config = {}) => {
      if (url.includes("/jobs/") && !config?.params) {
        const id = url.split("/").filter(Boolean).pop();
        detailRequests.push(id);
        return Promise.resolve({ data: detailById.get(id) || {} });
      }

      const from = config?.params?.DateIssuedFrom || null;
      listParams.push(from);
      const payload =
        from === "2025-01-01"
          ? firstBatch
          : from === "2025-01-03"
          ? incrementalBatch
          : [];
      return Promise.resolve({ data: payload, headers: { "result-pages": "1" } });
    });

    const res1 = await request(app)
      .get("/data/sync")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res1.status).toBe(200);
    expect(res1.body.count).toBe(3);
    expect(listParams).toEqual(["2025-01-01"]);
    expect(res1.body.jobs.map((j) => j.ID)).toEqual(["3", "2", "1"]);
    expect([...detailRequests].sort()).toEqual(["1", "2", "3"]);

    const res2 = await request(app)
      .get("/data/sync")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res2.status).toBe(200);
    expect(res2.body.upserted).toBe(1);
    expect(res2.body.count).toBe(4);
    expect(res2.body.jobs.map((j) => j.ID)).toEqual(["4", "3", "2", "1"]);
    expect(listParams).toEqual(["2025-01-01", "2025-01-03"]);
    expect(detailRequests).toContain("4");
    expect(detailRequests).not.toContain("legacy");
  });

  test("/sync full mode replaces stored jobs with 2025-onward data", async () => {
    const initialBatch = [
      {
        ID: "alpha",
        id: "alpha",
        Total: { IncTax: 150 },
        DateIssued: "2025-01-02T10:00:00Z",
        DateUpdated: "2025-01-03T00:00:00Z",
        Description: "Alpha job",
      },
      {
        ID: "beta",
        id: "beta",
        Total: { IncTax: 175 },
        DateIssued: "2025-01-03T09:00:00Z",
        DateUpdated: "2025-01-04T00:00:00Z",
        Description: "Beta job",
      },
    ];

    const fullBatch = [
      {
        ID: "gamma",
        id: "gamma",
        Total: { IncTax: 220 },
        DateIssued: "2025-02-10T09:00:00Z",
        DateUpdated: "2025-02-11T00:00:00Z",
        Description: "Gamma job",
      },
      {
        ID: "legacy",
        id: "legacy",
        Total: { IncTax: 80 },
        DateIssued: "2024-12-31T12:00:00Z",
        DateUpdated: "2025-01-01T00:00:00Z",
        Description: "Legacy job",
      },
    ];

    const detailById = new Map(
      [...initialBatch, ...fullBatch].map((job) => [String(job.ID), job]),
    );

    let currentBatch = initialBatch;
    const listCalls = [];

    getSpy.mockImplementation((url, config = {}) => {
      if (url.includes("/jobs/") && !config?.params) {
        const id = url.split("/").filter(Boolean).pop();
        return Promise.resolve({ data: detailById.get(id) || {} });
      }

      listCalls.push({
        from: config?.params?.DateIssuedFrom,
        page: config?.params?.page,
      });

      return Promise.resolve({ data: currentBatch, headers: { "result-pages": "1" } });
    });

    const res1 = await request(app)
      .get("/data/sync")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res1.status).toBe(200);
    expect(res1.body.count).toBe(2);
    expect(res1.body.mode).toBe("update");
    expect(res1.body.jobs.map((j) => j.ID).sort()).toEqual(["alpha", "beta"].sort());

    currentBatch = fullBatch;

    const res2 = await request(app)
      .get("/data/sync?mode=full")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res2.status).toBe(200);
    expect(res2.body.mode).toBe("full");
    expect(res2.body.count).toBe(1);
    expect(res2.body.jobs.map((j) => j.ID)).toEqual(["gamma"]);
    expect(res2.body.upserted).toBe(1);

    const res3 = await request(app)
      .get("/data/jobs")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res3.status).toBe(200);
    expect(res3.body.total).toBe(1);
    expect(res3.body.jobs[0].ID).toBe("gamma");

    const fromValues = listCalls.map((c) => c.from);
    expect(fromValues.every((from) => from === "2025-01-01" || from == null)).toBe(true);
  });

  test("/jobs applies filtering and sorting", async () => {
    mockJobsEndpoints(jobsMock);

    await request(app)
      .get("/data/sync")
      .set("Authorization", `Bearer ${getToken()}`);

    const res1 = await request(app)
      .get("/data/jobs?minRevenue=200")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res1.status).toBe(200);
    expect(res1.body.jobs.map((j) => j.revenue)).toEqual([200, 300]);

    const res2 = await request(app)
      .get("/data/jobs?maxRevenue=200")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res2.body.jobs.map((j) => j.revenue)).toEqual([100, 200]);

    const res3 = await request(app)
      .get("/data/jobs?sortField=revenue&order=desc")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res3.body.jobs[0].revenue).toBe(300);
  });

  test("/predict forwards filtered jobs to ML and handles errors", async () => {
    mockJobsEndpoints(jobsMock);

    await request(app)
      .get("/data/sync")
      .set("Authorization", `Bearer ${getToken()}`);

    postSpy.mockImplementation((url, body) => {
      if (url === process.env.TOKEN_URL) {
        return Promise.resolve({ data: { access_token: "mocktoken", expires_in: 3600 } });
      }
      if (url === `${process.env.ML_URL}/predict`) {
        return Promise.resolve({ data: { prediction: "ok" } });
      }
      return Promise.reject(new Error(`Unexpected axios.post call to ${url}`));
    });

    const res1 = await request(app)
      .post("/data/predict?minRevenue=200")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res1.status).toBe(200);
    expect(res1.body.prediction).toBe("ok");

    const mlCall = postSpy.mock.calls.find(
      ([url]) => url === `${process.env.ML_URL}/predict`,
    );
    expect(mlCall).toBeTruthy();
    expect(mlCall[1].data.map((j) => j.ID)).toEqual(["2", "3"]);

    postSpy.mockImplementation((url, body) => {
      if (url === process.env.TOKEN_URL) {
        return Promise.resolve({ data: { access_token: "mocktoken", expires_in: 3600 } });
      }
      if (url === `${process.env.ML_URL}/predict`) {
        return Promise.reject(new Error("ML down"));
      }
      return Promise.reject(new Error(`Unexpected axios.post call to ${url}`));
    });

    const res2 = await request(app)
      .post("/data/predict")
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res2.status).toBe(502);
    expect(res2.body.error).toMatch(/ML Prediction failed/i);
  });

  test("/predict accepts jobIds in the request body", async () => {
    mockJobsEndpoints(jobsMock);

    await request(app)
      .get("/data/sync")
      .set("Authorization", `Bearer ${getToken()}`);

    postSpy.mockImplementation((url, body) => {
      if (url === process.env.TOKEN_URL) {
        return Promise.resolve({ data: { access_token: "mocktoken", expires_in: 3600 } });
      }
      if (url === `${process.env.ML_URL}/predict`) {
        return Promise.resolve({ data: { predictions: [{ jobId: "2", class: "High" }] } });
      }
      return Promise.reject(new Error(`Unexpected axios.post call to ${url}`));
    });

    const res = await request(app)
      .post("/data/predict")
      .send({ jobIds: [2] })
      .set("Authorization", `Bearer ${getToken()}`);

    expect(res.status).toBe(200);
    expect(res.body.predictions).toEqual([{ jobId: "2", class: "High" }]);

    const mlCall = postSpy.mock.calls.find(
      ([url]) => url === `${process.env.ML_URL}/predict`,
    );
    expect(mlCall[1].data).toHaveLength(1);
    expect(mlCall[1].data[0].ID).toBe("2");
  });
});