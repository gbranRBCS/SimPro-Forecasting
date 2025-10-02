import { useState, useEffect } from "react";
import { syncJobs, getJobs, predict } from "../features/jobs/api";
import { toJobRowView, formatCurrency, formatDate, classBadgeProps } from "../utils/jobs";

type ApiJob = Record<string, any>;

export default function Dashboard() {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [minRev, setMinRev] = useState<number | undefined>(undefined);
  const [maxRev, setMaxRev] = useState<number | undefined>(undefined);
  const [order, setOrder] = useState<"asc" | "desc">("asc");
  const [limit, setLimit] = useState<number | undefined>(undefined);
  const [jobs, setJobs] = useState<ApiJob[]>([]);
  type Prediction = {
    jobId: string | number;
    class?: string;
    confidence?: number;
    probability?: number;
    margin_est?: number;
    [key: string]: any;
  };

  const [pred, setPred] = useState<{ highCount: number; mediumCount: number; lowCount: number; count: number; avgConfidence?: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [totalPages, setTotalPages] = useState(1);

  async function doSync() {
    setLoading(true);
    setMsg(null);
    try {
      const r = await syncJobs({ from, to });
      setMsg(r?.message || "Synced");
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Sync failed");
    } finally {
      setLoading(false);
    }
  }

  async function load() {
    setLoading(true);
    setMsg(null);
    try {
      const data = await getJobs({
        minRevenue: minRev,
        maxRevenue: maxRev,
        sortField: "revenue",
        order,
        limit,
        page,
        pageSize,
      });
      setJobs(data?.jobs ?? []);
      setTotalPages(data?.totalPages ?? 1);
      setPred(null);
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Load failed");
    } finally {
      setLoading(false);
    }
  }

  // auto-load when page or pageSize changes
  useEffect(() => {
    if (typeof pageSize === "number" && pageSize > 0) {
      // @ts-ignore - load is defined below
      load();
    }
  }, [page, pageSize]);

  async function runPredict() {
    setLoading(true);
    setMsg(null);
    try {
      const r = await predict({ jobs });
      const preds: Prediction[] = r?.predictions ?? [];

      // index predictions by jobId
      const byId = new Map(preds.map((p) => [p.jobId, p]));

      const updated = jobs.map((j) => {
        const p = byId.get(j.ID) as Prediction | undefined;
        if (!p) return j;

        // default to trained model output (class + confidence), fallback to legacy fields
        const klass: string | null =
          (p.class as string) ??
          (typeof p.margin_est === "number"
            ? p.margin_est >= 0.10
              ? "High"
              : p.margin_est >= 0.03
              ? "Medium"
              : "Low"
            : null);

        const scoreType =
          typeof p.confidence === "number"
            ? "confidence"
            : typeof p.probability === "number"
            ? "probability"
            : typeof p.margin_est === "number"
            ? "margin"
            : null;

        const score =
          (typeof p.confidence === "number" && p.confidence) ||
          (typeof p.probability === "number" && p.probability) ||
          (typeof p.margin_est === "number" && p.margin_est) ||
          null;

        return {
          ...j,
          profitability: {
            class: klass ?? j.profitability?.class ?? j.profitability_class ?? null,
            score: score,
            scoreType,
          },
        };
      });
      setJobs(updated);

      // summary by class
      const getClass = (p: any): string | null => {
        if (p?.class) return p.class;
        if (typeof p?.margin_est === "number") {
          return p.margin_est >= 0.10 ? "High" : p.margin_est >= 0.03 ? "Medium" : "Low";
        }
        return null;
      };
      const classes = preds.map((p: any) => getClass(p)).filter(Boolean) as string[];
      const highCount = classes.filter((c) => c === "High").length;
      const mediumCount = classes.filter((c) => c === "Medium").length;
      const lowCount = classes.filter((c) => c === "Low").length;
      const count = classes.length;

      const confidences = preds
        .map((p: any) =>
          typeof p.confidence === "number"
            ? p.confidence
            : typeof p.probability === "number"
            ? p.probability
            : null,
        )
        .filter((x: number | null) => typeof x === "number") as number[];
      const avgConfidence =
        confidences.length > 0
          ? confidences.reduce((a, b) => a + b, 0) / confidences.length
          : undefined;

      setPred({ highCount, mediumCount, lowCount, count, avgConfidence });
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Predict failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen pb-24">
      {/* Pinned top toolbar: inputs + sync/load/predict */}
      <div className="sticky top-0 z-30 bg-white/95 backdrop-blur-sm border-b shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-3 flex flex-wrap gap-3 items-center">
          {/* Controls */}
          <div className="grid gap-3 md:grid-cols-6">
            <input
              className="border p-2"
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              aria-label="From date"
            />
            <input
              className="border p-2"
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              aria-label="To date"
            />
            <input
              className="border p-2"
              placeholder="Min revenue (IncTax)"
              onChange={(e) => setMinRev(e.target.value ? Number(e.target.value) : undefined)}
              aria-label="Min revenue"
            />
            <input
              className="border p-2"
              placeholder="Max revenue (IncTax)"
              onChange={(e) => setMaxRev(e.target.value ? Number(e.target.value) : undefined)}
              aria-label="Max revenue"
            />
            <select
              className="border p-2"
              value={order}
              onChange={(e) => setOrder(e.target.value as "asc" | "desc")}
              aria-label="Order"
            >
              <option value="asc">asc</option>
              <option value="desc">desc</option>
            </select>
            <input
              className="border p-2"
              placeholder="Limit"
              value={limit ?? ""}
              onChange={(e) => setLimit(e.target.value ? Number(e.target.value) : undefined)}
              aria-label="Limit"
            />
          </div>

          <div className="flex gap-3">
            <button
              className="px-3 py-2 bg-blue-600 text-white rounded"
              onClick={doSync}
              disabled={loading}
            >
              Sync
            </button>
            <button
              className="px-3 py-2 bg-gray-700 text-white rounded"
              onClick={load}
              disabled={loading}
            >
              Load Jobs
            </button>
            <button
              className="px-3 py-2 bg-green-600 text-white rounded"
              onClick={runPredict}
              disabled={loading || jobs.length === 0}
            >
              Predict
            </button>
          </div>
        </div>
      </div>

      {/* content area */}
      <div className="max-w-6xl mx-auto px-4 pt-4 pb-16">
        {msg && <div className="text-sm text-red-700">{msg}</div>}

        {/* Prediction */}
        {pred && (
          <div className="p-4 rounded border">
            <div className="font-semibold">Profitability Summary</div>
            <div>High: {pred.highCount}</div>
            <div>Medium: {pred.mediumCount}</div>
            <div>Low: {pred.lowCount}</div>
            <div>Total predicted: {pred.count}</div>
            {typeof pred.avgConfidence === "number" && (
              <div>Avg confidence: {(pred.avgConfidence * 100).toFixed(1)}%</div>
            )}
          </div>
        )}

        {/* Jobs table */}
        <div className="overflow-auto border rounded">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left p-2">ID</th>
                <th className="text-left p-2">Name</th>
                <th className="text-left p-2">Customer</th>
                <th className="text-left p-2">Site</th>
                <th className="text-left p-2">Status</th>
                <th className="text-left p-2">Issued</th>
                <th className="text-left p-2">Due</th>
                <th className="text-right p-2">Revenue (IncTax)</th>
                <th className="text-right p-2">Profitability</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => {
                const v = toJobRowView(j);
                const badge = classBadgeProps(v.profitClass);
                return (
                  <tr key={v.id} className="border-t">
                    <td className="p-2">{v.id}</td>
                    <td className="p-2">{v.name}</td>
                    <td className="p-2">{v.customer}</td>
                    <td className="p-2">{v.site}</td>
                    <td className="p-2">{v.status}</td>
                    <td className="p-2">{formatDate(v.issued)}</td>
                    <td className="p-2">{formatDate(v.due)}</td>
                    <td className="p-2 text-right">{formatCurrency(v.revenue ?? 0)}</td>
                    <td className="p-2 text-right">
                      <span
                        className={`inline-block rounded px-2 py-0.5 text-xs border ${
                          badge.tone === "success"
                            ? "bg-green-50 border-green-300 text-green-800"
                            : badge.tone === "warning"
                            ? "bg-yellow-50 border-yellow-300 text-yellow-800"
                            : badge.tone === "destructive"
                            ? "bg-red-50 border-red-300 text-red-800"
                            : "bg-gray-50 border-gray-300 text-gray-700"
                        }`}
                        title={
                          v.profitScore != null
                            ? `${v.profitScoreType || "score"}: ${(v.profitScore * 100).toFixed(1)}%`
                            : undefined
                        }
                      >
                        {badge.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
              {jobs.length === 0 && (
                <tr>
                  <td className="p-3 text-center text-gray-500" colSpan={9}>
                    No jobs
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        
      </div>

      {/* Sticky pagination footer */}
      <div className="fixed bottom-0 left-0 right-0 z-40 border-t bg-white/95 backdrop-blur-sm shadow-lg">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-center gap-3 px-4 py-3">
          <button
            className="rounded border bg-gray-800 px-3 py-1.5 text-white hover:bg-gray-700 disabled:opacity-50"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            Previous
          </button>

          <div className="text-sm font-medium">
            Page {page} of {totalPages}
          </div>

          <label className="flex items-center gap-2 text-sm">
            <span>Page size:</span>
            <select
              id="page-size"
              className="w-20 rounded border px-2 py-1.5"
              value={pageSize}
              onChange={(e) => {
                const newSize = Number(e.target.value);
                setPageSize(newSize);
                setPage(1); // Reset to first page when changing page size
              }}
            >
              {[10, 20, 50, 100].map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>

          <button
            className="rounded border bg-gray-800 px-3 py-1.5 text-white hover:bg-gray-700 disabled:opacity-50"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
