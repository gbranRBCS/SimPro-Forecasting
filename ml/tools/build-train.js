import fs from 'fs';
import path from 'path';
import process from 'process';
import axios from 'axios';

function arg(name, def=null) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i+1] : def;
}

const BASE = arg('base', 'http://localhost:5001');
const USER = arg('user');
const PASS = arg('pass');
const TOKEN = arg('token');
const OUT  = arg('out', 'ml/train.json');
const LOW = Number(arg('low', '0.44'));
const HIGH = Number(arg('high', '0.64'));

// date-range + limit arguments for historical data sync
const FROM = arg('from', '2010-01-01');           // default start (very old)
const TO   = arg('to', '2024-03-31');             // default end: end of 2024 Q1
const LIMIT = Number(arg('limit', '50'));         // how many jobs to read after syncing

function toNum(x) {
  if (x === null || x === undefined) return null;
  const n = typeof x === 'number' ? x : parseFloat(String(x).replace(/[,£$]/g,''));
  return Number.isFinite(n) ? n : null;
}

function computeProfitClassFromMargin(p) {
  if (p == null) return null;
  if (p > HIGH) return "High";
  if (p >= LOW) return "Medium";
  return "Low";
}

function deriveRow(j) {
  // revenue fallback: revenue || Total.IncTax
  const rev = toNum(j.revenue ?? j.Total?.IncTax);

  // costs: prefer *_cost_est and cost_est_total; else fall back to legacy keys
  const mat = toNum(j.materials_cost_est ?? j.materials) ?? 0;
  const lab = toNum(j.labor_cost_est ?? j.labour) ?? 0;
  const ovh = toNum(j.overhead_est ?? j.overhead) ?? 0;
  const cost_total = toNum(j.cost_est_total ?? j.cost_total) ?? (mat + lab + ovh || null);

  // label: try netMarginPct; else compute if possible; else keep class
  let netMarginPct = j.netMarginPct;
  if (netMarginPct == null && rev && cost_total != null && rev > 0) {
    netMarginPct = (rev - cost_total) / rev;
  }

  // if still no margin, accept profitability_class 
  const profitability_class_backend = j.profitability_class ?? null;

  // prefer computed class (ML thresholds) - fall back to backend-provided class only if compute returns null
  const profitability_class = computeProfitClassFromMargin(netMarginPct) ?? profitability_class_backend;

  // keep only if we have margin || class
  if (netMarginPct == null && !profitability_class) return null;

  return {
    ID: j.ID ?? j.id ?? null,
    revenue: rev ?? null,
    materials: mat,
    labour: lab,
    overhead: ovh,
    cost_total,
    statusName: j.statusName ?? null,
    jobType: j.jobType ?? null,
    customerName: j.customerName ?? null,
    siteName: j.siteName ?? null,
    job_age_days: j.job_age_days ?? null,
    lead_time_days: j.lead_time_days ?? null,
    is_overdue: j.is_overdue ?? 0,
    descriptionText: j.descriptionText ?? "",
    dateIssued: j.dateIssued ?? null,
    ...(netMarginPct != null ? { netMarginPct } : {}),
    ...(profitability_class ? { profitability_class } : {})
  };
}

async function main() {
  let jwt = TOKEN;
  if (!jwt) {
    if (!USER || !PASS) {
      console.error('Provide --token OR (--user and --pass).');
      process.exit(1);
    }
    const auth = await axios.post(`${BASE}/auth/login`, { username: USER, password: PASS });
    jwt = auth.data.token;
  }
  const h = { Authorization: `Bearer ${jwt}` };

  // ask backend to sync historical range
  console.log(`Requesting sync for DateIssuedFrom=${FROM} DateIssuedTo=${TO}`);
  await axios.get(`${BASE}/data/sync`, { headers: h, params: { DateIssuedFrom: FROM, DateIssuedTo: TO, force: 1 } });


  const jobsRes = await axios.get(`${BASE}/data/jobs`, {
    headers: h,
    params: { order: 'date', limit: LIMIT }
  });

  const jobs = jobsRes.data?.jobs ?? jobsRes.data ?? [];
  const rows = [];
  for (const j of jobs) {
    const r = deriveRow(j);
    if (r) rows.push(r);
  }

  const payload = {
    data: rows,
    test_size: 0.2,
    random_state: 42,
    max_tfidf_features: 500,
    thresholds: { low: LOW, high: HIGH }
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(payload, null, 2));
  console.log(`Wrote ${rows.length} rows to ${OUT}`);
  const haveMargin = rows.filter(r => r.netMarginPct != null).length;
  const haveClass  = rows.filter(r => r.profitability_class).length;
  console.log({ haveMargin, haveClass });
}

main().catch(e => {
  console.error('build-train failed:', e?.response?.data ?? e);
  process.exit(1);
});