import { useState } from "react";
import { useLogin } from "../features/auth/useLogin";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  
  // hook handles api call and keeps track of loading/errors logic
  const { login, loading, error } = useLogin();

  async function handleSubmit(event) {
    // stop the default form submission which refreshes the page
    event.preventDefault();
    
    // attempt to log in using the hook
    const success = await login(username, password);

    if (success) {
      // direct browser to dashboard on success
      window.location.href = "/dashboard";
    }
  }

  // handler for username input changes
  function handleUsernameChange(event) {
    setUsername(event.target.value);
  }

  // handler for password input changes
  function handlePasswordChange(event) {
    setPassword(event.target.value);
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center px-6">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm space-y-6 bg-slate-900/70 border border-slate-800 rounded-xl shadow-lg p-8"
      >
        <div className="space-y-2 text-center">
          <h1 className="text-2xl font-semibold text-slate-100">
            SimPRO Forecasting
          </h1>
          <p className="text-sm text-slate-400">
            Please sign in to continue to your dashboard:
          </p>
        </div>

        <div className="space-y-4">
          <input
            className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            placeholder="Username"
            type="text"
            value={username}
            onChange={handleUsernameChange}
            disabled={loading}
          />
          
          <input
            className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
            placeholder="Password"
            type="password"
            value={password}
            onChange={handlePasswordChange}
            disabled={loading}
          />
        </div>

        {error && (
          <div className="rounded-lg border border-red-900/40 bg-red-900/20 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Signing in..." : "Login"}
        </button>
      </form>
    </div>
  );
}