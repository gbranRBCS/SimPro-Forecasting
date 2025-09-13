import { useState } from "react";
import { syncJobs, getJobs, predict } from "../features/jobs/api";
import { toJobRowView, formatCurrency, formatDate, classBadgeProps } from "../utils/jobs";

type ApiJob = Record<string, any>;

export default function Dashboard() {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [minRev, setMinRev] = useState<number | undefined>(undefined);
  const [maxRev, setMaxRev] = useState<number | undefined>(undefined);
  const [order, setOrder] = useState<"asc" | "desc">("asc");
  const [limit, setLimit] = useState<number | undefined>(20);
  const [jobs, setJobs] = useState<ApiJob[]>([]);
  const [pred, setPred] = useState<{ avgProbability: number; profitableCount: number; count: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

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
      });
      setJobs(data?.jobs ?? []);
      setPred(null);
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Load failed");
    } finally {
      setLoading(false);
    }
  }

  async function runPredict() {
    setLoading(true);
    setMsg(null);
    try {
      const r = await predict({
        minRevenue: minRev,
        maxRevenue: maxRev,
        sortField: "revenue",
        order,
        limit,
      });
      const preds = r?.predictions ?? [];

      // attach profitability info to jobs
      const byId = new Map(preds.map((p: any) => [p.jobId, p]));
      const probToClass = (p: number, profitable: boolean) => {
        if (!profitable) return "Low";
        if (p >= 0.7) return "High";
        if (p >= 0.5) return "Medium";
        return "Low";
      };
      const updated = jobs.map((j) => {
        const p = byId.get(j.ID);
        if (!p) return j;
        return {
          ...j,
          profitability: {
            class: probToClass(p.probability ?? 0, !!p.profitable),
            score: p.probability ?? null,
          },
        };
      });
      setJobs(updated);

      const count = preds.length;
      const profitableCount = preds.filter((p: any) => p.profitable).length;
      const avgProbability = count ? preds.reduce((s: number, p: any) => s + (p.probability || 0), 0) / count : 0;
      setPred({ avgProbability, profitableCount, count });
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Predict failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6 space-y-4">
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
          defaultValue={20}
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

      {msg && <div className="text-sm text-red-700">{msg}</div>}

      {/* Prediction */}
      {pred && (
        <div className="p-4 rounded border">
          <div className="font-semibold">Profitability Summary</div>
          <div>profitable: {pred.profitableCount} / {pred.count}</div>
          <div>avg probability: {(pred.avgProbability * 100).toFixed(1)}%</div>
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
                      title={v.profitScore != null ? `score: ${v.profitScore}` : undefined}
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
  );
}
