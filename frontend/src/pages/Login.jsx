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
    <form onSubmit={submit} className="p-6 space-y-3 max-w-sm">
      <input className="border p-2 w-full" placeholder="username" value={u} onChange={e=>setU(e.target.value)} />
      <input className="border p-2 w-full" placeholder="password" type="password" value={p} onChange={e=>setP(e.target.value)} />
      {err && <div className="text-red-700 text-sm">{err}</div>}
      <button className="px-3 py-2 bg-blue-600 text-white rounded w-full">Login</button>
    </form>
  );
}