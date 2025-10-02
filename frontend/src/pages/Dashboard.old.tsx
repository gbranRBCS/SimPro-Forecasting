import { useState, useEffect } from "react";
import { syncJobs, getJobs, predict } from "../features/jobs/api";
import { toJobRowView, formatCurrency, formatDate, classBadgeProps } from "../utils/jobs";

type ApiJob = Record<string, any>;

function Spinner({ size = 20 }: { size?: number }) {
  const px = `${size}px`;
  return (
    <span
      className="inline-block rounded-full animate-spin border-2 border-slate-500 border-t-blue-500"
      style={{ width: px, height: px }}
      aria-label="Loading"
      role="status"
    />
  );
}

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
  const [syncing, setSyncing] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [totalPages, setTotalPages] = useState(1);

  async function doSync() {
    setMsg(null);
    setSyncing(true);
    try {
      const r = await syncJobs({ from, to });
      setMsg(r?.message || "Synced");
      // after sync completes, refresh current page automatically
      await load();
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Sync failed");
    } finally {
      setSyncing(false);
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

  // auto-load on page or pageSize change
  useEffect(() => {
    if (typeof pageSize === "number" && pageSize > 0) {
      load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, pageSize]);

  async function runPredict() {
    setLoading(true);
    setMsg(null);
    try {
      const r = await predict({ jobs });
      const preds: Prediction[] = r?.predictions ?? [];
      const byId = new Map(preds.map((p) => [p.jobId, p]));
      const updated = jobs.map((j) => {
        const p = byId.get(j.ID) as Prediction | undefined;
        if (!p) return j;
        const klass: string | null =
          (p.class as string) ??
          (typeof p.margin_est === "number"
            ? p.margin_est >= 0.1
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
            score,
            scoreType,
          },
        };
      });
      setJobs(updated);
      const getClass = (p: any): string | null => {
        if (p?.class) return p.class;
        if (typeof p?.margin_est === "number") {
          return p.margin_est >= 0.1 ? "High" : p.margin_est >= 0.03 ? "Medium" : "Low";
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
        confidences.length > 0 ? confidences.reduce((a, b) => a + b, 0) / confidences.length : undefined;
      setPred({ highCount, mediumCount, lowCount, count, avgConfidence });
    } catch (e: any) {
      const respData = e?.response?.data;
      if (respData) {
        const parts: string[] = [];
        const baseMsg = respData.error || respData.message || "Predict failed";
        if (baseMsg) parts.push(baseMsg);
        if (respData.mlStatus) parts.push(`ML status ${respData.mlStatus}`);
        if (respData.mlBody) {
          if (typeof respData.mlBody === "string") {
            parts.push(respData.mlBody);
          } else if (respData.mlBody?.error || respData.mlBody?.message) {
            parts.push(respData.mlBody.error || respData.mlBody.message);
          } else {
            try {
              parts.push(JSON.stringify(respData.mlBody));
            } catch (_) {
              // ignore stringify errors
            }
          }
        }
        setMsg(parts.join(" - "));
      } else {
        setMsg(e?.message || "Predict failed");
      }
      setPred(null);
    } finally {
      setLoading(false);
    }
  }

  // dark theme + layout
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Sticky top toolbar */}
      <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="grid gap-3 grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6">
            <input
              className="bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-400 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-transparent focus:ring-2 focus:ring-blue-600/60 transition-colors"
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              aria-label="From date"
            />
            <input
              className="bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-400 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-transparent focus:ring-2 focus:ring-blue-600/60 transition-colors"
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              aria-label="To date"
            />
            <input
              className="bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-400 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-transparent focus:ring-2 focus:ring-blue-600/60 transition-colors"
              type="number"
              placeholder="Min revenue"
              onChange={(e) => setMinRev(e.target.value ? Number(e.target.value) : undefined)}
              aria-label="Min revenue"
            />
            <input
              className="bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-400 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-transparent focus:ring-2 focus:ring-blue-600/60 transition-colors"
              type="number"
              placeholder="Max revenue"
              onChange={(e) => setMaxRev(e.target.value ? Number(e.target.value) : undefined)}
              aria-label="Max revenue"
            />
            <select
              className="bg-slate-800 border border-slate-700 text-slate-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-transparent focus:ring-2 focus:ring-blue-600/60 transition-colors"
              value={order}
              onChange={(e) => setOrder(e.target.value as "asc" | "desc")}
              aria-label="Order"
            >
              <option value="asc">Asc</option>
              <option value="desc">Desc</option>
            </select>
            <input
              className="bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-400 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-transparent focus:ring-2 focus:ring-blue-600/60 transition-colors"
              type="number"
              placeholder="Limit"
              value={limit ?? ""}
              onChange={(e) => setLimit(e.target.value ? Number(e.target.value) : undefined)}
              aria-label="Limit"
            />
          </div>

          <div className="flex gap-3 mt-3 flex-wrap">
            <button
              className="px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-600/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
              onClick={doSync}
              disabled={syncing}
            >
              {syncing ? (
                <span className="inline-flex items-center gap-2"><Spinner size={16} /> Syncing…</span>
              ) : (
                "Sync"
              )}
            </button>
            <button
              className="px-4 py-2.5 bg-slate-700 text-white rounded-lg hover:bg-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-600/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
              onClick={load}
              disabled={loading || syncing}
            >
              Load Jobs
            </button>
            <button
              className="px-4 py-2.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 focus:outline-none focus:ring-2 focus:ring-blue-600/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
              onClick={runPredict}
              disabled={loading || syncing || jobs.length === 0}
            >
              Predict
            </button>
          </div>
        </div>
      </div>

      {/* Page title */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <h1 className="text-2xl font-bold text-slate-100">
          Jobs Dashboard
        </h1>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 pb-6">
        {/* Status message */}
        {msg && (
          <div className="mb-4 px-4 py-3 rounded-xl border bg-blue-900/20 border-blue-800/50 text-blue-300 text-sm">
            {msg}
          </div>
        )}

        {/* Sync progress panel */}
        {syncing && (
          <div className="mb-4 px-6 py-4 rounded-xl border border-slate-800 bg-slate-900">
            <div className="flex items-center gap-3">
              <Spinner />
              <div>
                <div className="font-medium text-slate-100">Sync in progress…</div>
                <div className="text-xs text-slate-400 mt-1">This can take a few minutes depending on date range and API rate limits.</div>
              </div>
            </div>
          </div>
        )}

        {/* Prediction Summary */}
        {pred && (
          <div className="mb-4 px-6 py-4 rounded-xl border border-slate-800 bg-slate-900">
            <div className="text-sm font-medium text-slate-100 mb-3">Profitability Summary</div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <div className="text-xs text-slate-400">High</div>
                <div className="text-lg font-medium text-green-300">{pred.highCount}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400">Medium</div>
                <div className="text-lg font-medium text-yellow-300">{pred.mediumCount}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400">Low</div>
                <div className="text-lg font-medium text-red-300">{pred.lowCount}</div>
              </div>
              <div>
                <div className="text-xs text-slate-400">Total</div>
                <div className="text-lg font-medium text-slate-300">{pred.count}</div>
              </div>
              {typeof pred.avgConfidence === "number" && (
                <div>
                  <div className="text-xs text-slate-400">Avg Confidence</div>
                  <div className="text-lg font-medium text-slate-300">{(pred.avgConfidence * 100).toFixed(1)}%</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Jobs table with loading overlay */}
        <div className="relative border border-slate-800 rounded-xl overflow-hidden">
          {loading && (
            <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center z-10">
              <div className="flex items-center gap-3 text-slate-100">
                <Spinner /> 
                <span className="text-sm">Loading…</span>
              </div>
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-900">
                <tr className="border-b border-slate-800">
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">ID</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Name</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Customer</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Site</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Status</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Issued</th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Due</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Revenue</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-slate-300 uppercase tracking-wider">Profitability</th>
                </tr>
              </thead>
              <tbody className="bg-slate-950">
                {jobs.map((j) => {
                  const v = toJobRowView(j);
                  const badge = classBadgeProps(v.profitClass);
                  return (
                    <tr key={v.id} className="border-t border-slate-800 hover:bg-slate-800/50 transition-colors">
                      <td className="px-4 py-3 text-slate-300 whitespace-nowrap">{v.id}</td>
                      <td className="px-4 py-3 text-slate-300" title={v.name}>{v.name}</td>
                      <td className="px-4 py-3 text-slate-300" title={v.customer}>{v.customer}</td>
                      <td className="px-4 py-3 text-slate-300" title={v.site}>{v.site}</td>
                      <td className="px-4 py-3 text-slate-400">{v.status}</td>
                      <td className="px-4 py-3 text-slate-400 whitespace-nowrap">{formatDate(v.issued)}</td>
                      <td className="px-4 py-3 text-slate-400 whitespace-nowrap">{formatDate(v.due)}</td>
                      <td className="px-4 py-3 text-right text-slate-300 whitespace-nowrap">{formatCurrency(v.revenue ?? 0)}</td>
                      <td className="px-4 py-3 text-right">
                        <span
                          className={`inline-block rounded-md px-2 py-0.5 text-xs font-medium border ${
                            badge.tone === "success"
                              ? "bg-green-900/30 text-green-300 border-green-800/50"
                              : badge.tone === "warning"
                              ? "bg-yellow-900/30 text-yellow-300 border-yellow-800/50"
                              : badge.tone === "destructive"
                              ? "bg-red-900/30 text-red-300 border-red-800/50"
                              : "bg-slate-800 border-slate-700 text-slate-300"
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
                    <td className="px-4 py-6 text-center text-slate-500 text-sm" colSpan={9}>
                      No jobs found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Pagination */}
        <div className="mt-4 flex items-center justify-between flex-wrap gap-3">
          <button
            className="px-4 py-2 bg-slate-800 text-slate-100 rounded-lg hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-600/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1 || loading || syncing}
          >
            Previous
          </button>

          <div className="flex items-center gap-4">
            <div className="text-sm text-slate-300">
              Page <span className="font-medium text-slate-100">{page}</span> of <span className="font-medium text-slate-100">{totalPages}</span>
            </div>

            <div className="flex items-center gap-2">
              <label htmlFor="page-size" className="text-xs text-slate-400 whitespace-nowrap">Page size:</label>
              <select
                id="page-size"
                className="bg-slate-800 border border-slate-700 text-slate-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-transparent focus:ring-2 focus:ring-blue-600/60 transition-colors"
                value={pageSize}
                onChange={(e) => {
                  const newSize = Number(e.target.value);
                  setPageSize(newSize);
                  setPage(1);
                }}
              >
                {[10, 20, 50, 100].map((size) => (
                  <option key={size} value={size}>{size}</option>
                ))}
              </select>
            </div>
          </div>

          <button
            className="px-4 py-2 bg-slate-800 text-slate-100 rounded-lg hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-600/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages || loading || syncing}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
