import { useState } from "react";
import { syncJobs, getJobs, predict } from "../../src/features/jobs/api";

export default function Dashboard() {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [minRev, setMinRev] = useState<number | undefined>(undefined);
  const [maxRev, setMaxRev] = useState<number | undefined>(undefined);
  const [order, setOrder] = useState<"asc"|"desc">("asc");
  const [limit, setLimit] = useState<number | undefined>(20);
  const [jobs, setJobs] = useState<any[]>([]);
  const [pred, setPred] = useState<{score:number; count:number} | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function doSync() {
    setLoading(true);
    try {
      const r = await syncJobs({ from, to });
      setMsg(r.message || "Synced");
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Sync failed");
    } finally { setLoading(false); }
  }

  async function load() {
    setLoading(true);
    try {
      const data = await getJobs({
        minRevenue: minRev, maxRevenue: maxRev,
        sortField: "revenue", order, limit
      });
      setJobs(data);
      setPred(null);
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Load failed");
    } finally { setLoading(false); }
  }

  async function runPredict() {
    setLoading(true);
    try {
      const r = await predict({
        minRevenue: minRev, maxRevenue: maxRev,
        sortField: "revenue", order, limit
      });
      const score = r?.predictions?.[0]?.score ?? 0;
      setPred({ score, count: r?.count ?? 0 });
    } catch (e: any) {
      setMsg(e?.response?.data?.error || "Predict failed");
    } finally { setLoading(false); }
  }

  return (
    <div className="p-6 space-y-4">
      {/* Controls */}
      <div className="grid gap-3 md:grid-cols-6">
        <input className="border p-2" type="date" value={from} onChange={e=>setFrom(e.target.value)} />
        <input className="border p-2" type="date" value={to} onChange={e=>setTo(e.target.value)} />
        <input className="border p-2" placeholder="Min IncTax" onChange={e=>setMinRev(e.target.value?Number(e.target.value):undefined)} />
        <input className="border p-2" placeholder="Max IncTax" onChange={e=>setMaxRev(e.target.value?Number(e.target.value):undefined)} />
        <select className="border p-2" value={order} onChange={e=>setOrder(e.target.value as any)}>
          <option value="asc">asc</option>
          <option value="desc">desc</option>
        </select>
        <input className="border p-2" placeholder="Limit" defaultValue={20} onChange={e=>setLimit(e.target.value?Number(e.target.value):undefined)} />
      </div>

      <div className="flex gap-3">
        <button className="px-3 py-2 bg-blue-600 text-white rounded" onClick={doSync} disabled={loading}>Sync</button>
        <button className="px-3 py-2 bg-gray-700 text-white rounded" onClick={load} disabled={loading}>Load Jobs</button>
        <button className="px-3 py-2 bg-green-600 text-white rounded" onClick={runPredict} disabled={loading || jobs.length===0}>Predict</button>
      </div>

      {msg && <div className="text-sm text-red-700">{msg}</div>}

      {/* Prediction */}
      {pred && (
        <div className="p-4 rounded border">
          <div className="font-semibold">Mean IncTax prediction</div>
          <div>score: {pred.score.toFixed(2)}</div>
          <div>count: {pred.count}</div>
        </div>
      )}

      {/* Jobs table */}
      <div className="overflow-auto border rounded">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left p-2">ID</th>
              <th className="text-left p-2">Description</th>
              <th className="text-right p-2">IncTax</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map(j => (
              <tr key={j.ID} className="border-t">
                <td className="p-2">{j.ID}</td>
                <td className="p-2">{(j.Description || "").replace(/<[^>]+>/g,"").slice(0,120)}</td>
                <td className="p-2 text-right">{j?.Total?.IncTax ?? 0}</td>
              </tr>
            ))}
            {jobs.length===0 && (
              <tr><td className="p-3 text-center text-gray-500" colSpan={3}>No jobs</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}