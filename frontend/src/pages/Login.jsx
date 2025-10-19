import { useState } from "react";
import api from "../lib/api";

export default function Login() {
  const [u, setU] = useState(""); const [p, setP] = useState("");
  const [err, setErr] = useState(null);

  async function submit(e) {
    e.preventDefault();
    setErr(null);
    try {
      const { data } = await api.post("/auth/login", { username: u, password: p });
      localStorage.setItem("token", data.token);
      window.location.href = "/dashboard";
    } catch (e2) {
      setErr(e2?.response?.data?.error || "Login failed");
    }
  }

  return (
    <form
      onSubmit={submit}
      className="min-h-screen bg-slate-950 flex flex-col items-center justify-center px-6"
    >
      <div className="w-full max-w-sm space-y-4 bg-slate-900/70 border border-slate-800 rounded-xl shadow-lg p-8">
        <div className="space-y-1 text-center">
          <h1 className="text-2xl font-semibold text-slate-100">SimPRO Forecasting</h1>
          <p className="text-sm text-slate-400">Please sign in to continue to your dashboard:</p>
        </div>
        <input
          className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="username"
          value={u}
          onChange={e=>setU(e.target.value)}
        />
        <input
          className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="password"
          type="password"
          value={p}
          onChange={e=>setP(e.target.value)}
        />
        {err && <div className="rounded-lg border border-red-700/40 bg-red-900/20 px-3 py-2 text-sm text-red-300">{err}</div>}
        <button className="w-full rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-slate-950">
          Login
        </button>
      </div>
    </form>
  );
}