import { useState } from "react";
import api from "../../lib/api";

export function useLogin() {
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);

  async function login(username: string, password: string) {
    setLoading(true); setError(null);
    try {
      const { data } = await api.post("/auth/login", { username, password });
      localStorage.setItem("token", data.token);
      return true;
    } catch (e: any) {
      setError(e?.response?.data?.error || "Login failed");
      return false;
    } finally { setLoading(false); }
  }

  return { login, loading, error };
}